"""Tendon-driven underactuated finger on Newton's massless routed tendons (XPBD).

Layout: finger straight along local +X, flexion joints about local +Y, palmar
side is local -Z (positive joint angle curls the tip toward -Z).

Route: ANCHOR(slider) -> [PINHOLE(proximal body), PINHOLE(distal body)] per joint,
the last distal guide being the terminating ANCHOR. Friction lives only on the
passive pinholes. The actuator is a prismatic slider driven to a position target.

Return springs and the slider's position servo are applied as explicit body
forces (``DriveRig.apply``) rather than through SolverXPBD's joint targets: the
XPBD joint drive does not accumulate its multiplier across iterations, so its
effective stiffness depends on the iteration count (newton#4315; docs/results.md).
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton
from newton import TendonGuide, TendonGuideType

Y = (0.0, 1.0, 0.0)
STOP_KE = 3.0  # joint end-stop stiffness [N m/rad]; explicit, so STOP_KE * dt^2 / I must stay below 1
THETA_MIN = -0.3  # hyperextension stop [rad]


@dataclass
class FingerParams:
    L: tuple = (0.045, 0.030, 0.025)  # phalanx lengths [m]
    a: float = 0.006  # pinhole setback from each joint axis [m]
    h: tuple = (0.008, 0.008, 0.008)  # pinhole depth = moment arm at theta=0 [m]
    k: tuple = (0.10, 0.10, 0.10)  # return-spring stiffness [N m/rad]
    theta0: tuple = (0.0, 0.0, 0.0)  # spring rest angles [rad]
    joint_kd: float = 1.0e-4  # joint damping [N m s/rad]
    mu: float = 0.0  # capstan friction at every pinhole
    pretension: float = 0.0  # tendon shortening applied through rest_length [m]
    link_density: float = 1200.0  # [kg/m^3]
    half_w: float = 0.008  # phalanx half width (y) [m]
    half_t: float = 0.006  # phalanx half thickness (z) [m]
    compliance: float = 1.0e-5  # tendon span compliance [m/N]
    tendon_damping: float = 0.0  # [N s/m]
    slider_ke: float = 2.0e4  # actuator position gain [N/m]
    slider_kd: float = 100.0  # [N s/m]
    slider_mass: float = 0.5
    slider_fmax: float = 1.0e9  # actuator force limit [N]
    slider_lead: float = 0.05  # straight run between slider anchor and first pinhole [m]
    theta_max: float = 1.75
    stop_margin: float = 0.2  # flexion stop sits this far before the pinhole-crossing angle [rad]
    soft_stop: bool = True  # compliant explicit stops; False uses SolverXPBD's hard joint limits
    attach_tendon: bool = True  # negative control: False leaves the finger unrouted


@dataclass
class FingerHandles:
    links: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    springs: list = field(default_factory=list)
    slider_body: int = -1
    slider_joint: int = -1
    servo: int = -1
    tendon: int = -1


class DriveRig:
    """Collects explicit springs (revolute) and position servos (prismatic)."""

    def __init__(self):
        self.sp = dict(child=[], parent=[], prot=[], k=[], th0=[], kd=[], lo=[], hi=[])
        self.ac = dict(child=[], parent=[], ppos=[], prot=[], ke=[], kd=[], fmax=[])

    def add_spring(self, child, parent, parent_rot, k, th0, kd, lo=-10.0, hi=10.0):
        for key, v in zip(self.sp, (child, parent, parent_rot, k, th0, kd, lo, hi), strict=True):
            self.sp[key].append(v)
        return len(self.sp["child"]) - 1

    def add_servo(self, child, parent, parent_pos, parent_rot, ke, kd, fmax=1.0e9):
        for key, v in zip(self.ac, (child, parent, parent_pos, parent_rot, ke, kd, fmax), strict=True):
            self.ac[key].append(v)
        return len(self.ac["child"]) - 1

    def finalize(self, device, worlds=1, bodies_per_world=0):
        """Tile the single-world template over ``worlds`` replicated worlds."""

        def tile_idx(v):
            v = np.asarray(v, dtype=np.int32)
            out = np.concatenate([np.where(v >= 0, v + w * bodies_per_world, -1) for w in range(worlds)])
            return wp.array(out, dtype=int, device=device)

        def tile(v, dtype):
            v = np.asarray(v, dtype=np.float32)
            return wp.array(np.tile(v, (worlds,) + (1,) * (v.ndim - 1)), dtype=dtype, device=device)

        s, a = self.sp, self.ac
        self.n_sp, self.n_ac = len(s["child"]) * worlds, len(a["child"]) * worlds
        self.sp_child, self.sp_parent = tile_idx(s["child"]), tile_idx(s["parent"])
        self.sp_prot = tile(s["prot"], wp.quat)
        self.sp_k, self.sp_th0, self.sp_kd = tile(s["k"], float), tile(s["th0"], float), tile(s["kd"], float)
        self.sp_lo, self.sp_hi = tile(s["lo"], float), tile(s["hi"], float)
        self.sp_theta = wp.zeros(self.n_sp, dtype=float, device=device)
        self.ac_child, self.ac_parent = tile_idx(a["child"]), tile_idx(a["parent"])
        self.ac_ppos, self.ac_prot = tile(a["ppos"], wp.vec3), tile(a["prot"], wp.quat)
        self.ac_ke, self.ac_kd, self.ac_fmax = tile(a["ke"], float), tile(a["kd"], float), tile(a["fmax"], float)
        self.ac_target = wp.zeros(self.n_ac, dtype=float, device=device)
        self.ac_x = wp.zeros(self.n_ac, dtype=float, device=device)
        self.ac_force = wp.zeros(self.n_ac, dtype=float, device=device)
        return self

    def apply(self, state):
        dev = state.body_q.device
        wp.launch(
            spring_kernel, dim=self.n_sp, device=dev,
            inputs=[state.body_q, state.body_qd, self.sp_child, self.sp_parent, self.sp_prot, self.sp_k, self.sp_th0, self.sp_kd, self.sp_lo, self.sp_hi],
            outputs=[self.sp_theta, state.body_f],
        )  # fmt: skip
        wp.launch(
            servo_kernel, dim=self.n_ac, device=dev,
            inputs=[state.body_q, state.body_qd, self.ac_child, self.ac_parent, self.ac_ppos, self.ac_prot, self.ac_ke, self.ac_kd, self.ac_fmax, self.ac_target],
            outputs=[self.ac_x, self.ac_force, state.body_f],
        )  # fmt: skip


@wp.kernel
def spring_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    child: wp.array[int],
    parent: wp.array[int],
    parent_rot: wp.array[wp.quat],
    k: wp.array[float],
    th0: wp.array[float],
    kd: wp.array[float],
    lo: wp.array[float],
    hi: wp.array[float],
    theta_out: wp.array[float],
    body_f: wp.array[wp.spatial_vector],
):
    """Torsion spring + damper about the joint's local +Y axis."""
    i = wp.tid()
    c = child[i]
    p = parent[i]
    q_c = wp.transform_get_rotation(body_q[c])
    w_c = wp.spatial_bottom(body_qd[c])
    q_pj = parent_rot[i]
    w_p = wp.vec3(0.0)
    if p >= 0:
        q_pj = wp.transform_get_rotation(body_q[p]) * parent_rot[i]
        w_p = wp.spatial_bottom(body_qd[p])
    q_rel = wp.quat_inverse(q_pj) * q_c
    theta = 2.0 * wp.atan2(q_rel[1], q_rel[3])  # rotation about local Y
    if theta > wp.pi:
        theta -= 2.0 * wp.pi
    if theta < -wp.pi:
        theta += 2.0 * wp.pi
    axis = wp.quat_rotate(q_pj, wp.vec3(0.0, 1.0, 0.0))
    tau = -k[i] * (theta - th0[i]) - kd[i] * wp.dot(w_c - w_p, axis)
    # compliant end stops: a hard XPBD limit loaded by a saturated tendon diverged under shaking
    if theta > hi[i]:
        tau -= STOP_KE * (theta - hi[i])
    if theta < lo[i]:
        tau -= STOP_KE * (theta - lo[i])
    theta_out[i] = theta
    wp.atomic_add(body_f, c, wp.spatial_vector(wp.vec3(0.0), axis * tau))
    if p >= 0:
        wp.atomic_add(body_f, p, wp.spatial_vector(wp.vec3(0.0), -axis * tau))


@wp.kernel
def servo_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    child: wp.array[int],
    parent: wp.array[int],
    parent_pos: wp.array[wp.vec3],
    parent_rot: wp.array[wp.quat],
    ke: wp.array[float],
    kd: wp.array[float],
    fmax: wp.array[float],
    target: wp.array[float],
    x_out: wp.array[float],
    f_out: wp.array[float],
    body_f: wp.array[wp.spatial_vector],
):
    """PD position servo along the slider's local -X axis (positive x pulls tendon)."""
    i = wp.tid()
    c = child[i]
    p = parent[i]
    pos_c = wp.transform_get_translation(body_q[c])
    v_c = wp.spatial_top(body_qd[c])
    origin = parent_pos[i]
    q = parent_rot[i]
    v_p = wp.vec3(0.0)
    if p >= 0:
        origin = wp.transform_point(body_q[p], parent_pos[i])
        q = wp.transform_get_rotation(body_q[p]) * parent_rot[i]
        v_p = wp.spatial_top(body_qd[p])  # parent COM velocity (palm is driven slowly)
    axis = wp.quat_rotate(q, wp.vec3(-1.0, 0.0, 0.0))
    x = wp.dot(pos_c - origin, axis)
    f = ke[i] * (target[i] - x) - kd[i] * wp.dot(v_c - v_p, axis)
    f = wp.clamp(f, -fmax[i], fmax[i])  # actuator force limit
    x_out[i] = x
    f_out[i] = f
    wp.atomic_add(body_f, c, wp.spatial_vector(axis * f, wp.vec3(0.0)))
    if p >= 0:
        wp.atomic_add(body_f, p, wp.spatial_vector(-axis * f, wp.vec3(0.0)))


def joint_stop(p: FingerParams, i: int) -> float:
    """Flexion stop [rad]. The two pinholes of joint i meet at theta = pi - 2 atan2(h, a), where the
    crossing span has zero length; the stop sits ``stop_margin`` before that (the standoffs would collide)."""
    return min(p.theta_max, float(np.pi - 2.0 * np.arctan2(p.h[i], p.a) - p.stop_margin))


def add_finger(builder, rig: DriveRig, p: FingerParams, parent=-1, parent_world=None, mount=None, shape_cfg=None, key=""):
    """Add one finger. ``mount`` is the joint-1 frame in the parent body frame
    (world frame if ``parent == -1``); ``parent_world`` is the parent's pose."""
    parent_world = parent_world if parent_world is not None else wp.transform_identity()
    mount = mount if mount is not None else wp.transform_identity()
    root = wp.transform_multiply(parent_world, mount)  # joint-1 frame in world
    cfg = shape_cfg or newton.ModelBuilder.ShapeConfig(density=p.link_density)
    hnd = FingerHandles()
    ident = wp.quat_identity()
    mrot = wp.transform_get_rotation(mount)

    s_local = wp.vec3(-p.a - p.slider_lead, 0.0, -p.h[0])
    s_mount = wp.transform_multiply(mount, wp.transform(s_local, ident))
    sb = builder.add_link(
        xform=wp.transform_multiply(root, wp.transform(s_local, ident)),
        mass=p.slider_mass,
        inertia=wp.mat33(np.eye(3) * p.slider_mass * 1.0e-4),
    )
    sj = builder.add_joint_prismatic(
        parent, sb, parent_xform=s_mount, child_xform=wp.transform_identity(), axis=(-1.0, 0.0, 0.0),
        target_ke=0.0, target_kd=0.0, limit_lower=-0.05, limit_upper=0.2, label=f"{key}slider",
    )  # fmt: skip
    hnd.slider_body, hnd.slider_joint = sb, sj
    hnd.servo = rig.add_servo(sb, parent, tuple(wp.transform_get_translation(s_mount)), tuple(mrot), p.slider_ke, p.slider_kd, p.slider_fmax)

    prev, x0 = parent, 0.0
    for i, Li in enumerate(p.L):
        link = builder.add_link(xform=wp.transform_multiply(root, wp.transform(wp.vec3(x0, 0.0, 0.0), ident)))
        builder.add_shape_box(
            link, xform=wp.transform(wp.vec3(0.5 * Li, 0.0, 0.0), ident), hx=0.5 * Li - 0.002, hy=p.half_w, hz=p.half_t, cfg=cfg
        )  # fmt: skip
        pj = mount if i == 0 else wp.transform(wp.vec3(p.L[i - 1], 0.0, 0.0), ident)
        j = builder.add_joint_revolute(
            prev, link, parent_xform=pj, child_xform=wp.transform_identity(), axis=Y,
            target_ke=0.0, target_kd=0.0, limit_lower=-3.0 if p.soft_stop else THETA_MIN,
            limit_upper=3.0 if p.soft_stop else joint_stop(p, i), label=f"{key}flex{i}",
        )  # fmt: skip
        hnd.springs.append(rig.add_spring(link, prev, tuple(mrot) if i == 0 else tuple(ident), p.k[i], p.theta0[i], p.joint_kd,
                                          THETA_MIN if p.soft_stop else -10.0, joint_stop(p, i) if p.soft_stop else 10.0))
        hnd.links.append(link)
        hnd.joints.append(j)
        prev, x0 = link, x0 + Li
    if parent == -1:
        builder.add_articulation([sj], label=f"{key}act")
        builder.add_articulation(hnd.joints, label=f"{key}finger")

    if p.attach_tendon:
        span = dict(compliance=p.compliance, damping=p.tendon_damping)
        route = [TendonGuide(body=sb, offset=(0.0, 0.0, 0.0), axis=Y)]
        n = len(p.L)
        frame = root if parent == -1 else mount  # world-attached guides take world coordinates
        for i in range(n):
            if i == 0:
                prox = parent
                off = tuple(wp.transform_point(frame, wp.vec3(-p.a, 0.0, -p.h[0])))
                ax = tuple(wp.transform_vector(frame, wp.vec3(*Y)))
                rest = p.slider_lead - p.pretension if p.pretension != 0.0 else -1.0
            else:
                prox, off, ax, rest = hnd.links[i - 1], (p.L[i - 1] - p.a, 0.0, -p.h[i]), Y, -1.0
            route.append(TendonGuide(body=prox, guide_type=TendonGuideType.PINHOLE, offset=off, axis=ax, mu=p.mu, rest_length=rest, **span))
            last = i == n - 1
            route.append(
                TendonGuide(
                    body=hnd.links[i],
                    guide_type=TendonGuideType.ANCHOR if last else TendonGuideType.PINHOLE,
                    offset=(p.a, 0.0, -p.h[i]), axis=Y, mu=0.0 if last else p.mu, **span,
                )  # fmt: skip
            )
        hnd.tendon = builder.add_tendon(route, label=f"{key}tendon")
    return hnd


@contextlib.contextmanager
def capture_tendon_errors(out: list):
    """Collect Warp-printed 'ERROR: Tendon' diagnostics (they go to C stdout)."""
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            yield
        finally:
            wp.synchronize()
            sys.stdout.flush()
            libc.fflush(None)  # kernel printf uses C stdio: flush it before the descriptor is restored
            os.dup2(saved, 1)
            os.close(saved)
            tmp.seek(0)
            lines = tmp.read().decode(errors="replace").splitlines()
            out.extend(ln for ln in lines if "ERROR" in ln and "endon" in ln)
            rest = [ln for ln in lines if not ("ERROR" in ln and "endon" in ln)]
            if rest:
                print("\n".join(rest))


@dataclass
class SolverCfg:
    """Numerical knobs. The sensitivity study perturbs each one."""

    fps: int = 60
    substeps: int = 16
    iterations: int = 16
    tendon_max_sweeps: int = 256
    tendon_settle_tol: float = 1.0e-3
    tendon_settle_atol: float = 1.0e-6
    joint_relaxation: float = 0.8  # linear AND angular; unequal values bias XPBD dynamics (newton#4314)
    joint_angular_relaxation: float | None = None  # None: same as joint_relaxation; set only to test Newton's split defaults
    rigid_contact_relaxation: float = 0.8  # SolverXPBD default
    angular_damping: float = 0.0  # SolverXPBD default [1/s]

    @property
    def dt(self):
        return 1.0 / (self.fps * self.substeps)

    def make_solver(self, model):
        return newton.solvers.SolverXPBD(
            model,
            iterations=self.iterations,
            joint_linear_relaxation=self.joint_relaxation,
            joint_angular_relaxation=self.joint_relaxation if self.joint_angular_relaxation is None else self.joint_angular_relaxation,
            tendon_max_sweeps=self.tendon_max_sweeps,
            tendon_settle_tol=self.tendon_settle_tol,
            tendon_settle_atol=self.tendon_settle_atol,
            rigid_contact_relaxation=self.rigid_contact_relaxation,
            angular_damping=self.angular_damping,
        )


class FingerSim:
    """Single finger bolted to the world, gravity off: the statics/hysteresis rig."""

    def __init__(self, p: FingerParams, cfg: SolverCfg | None = None, device=None, obstacle=None):
        self.p, self.cfg = p, cfg or SolverCfg()
        b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
        self.rig = DriveRig()
        self.h = add_finger(b, self.rig, p)
        if obstacle is not None:
            obstacle(b)
        self.model = b.finalize(device=device)
        self.rig.finalize(self.model.device)
        self.solver = self.cfg.make_solver(self.model)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.use_contacts = obstacle is not None
        self.pipeline = newton.CollisionPipeline(self.model) if self.use_contacts else None
        self.contacts = self.pipeline.contacts() if self.use_contacts else None
        self.errors: list[str] = []
        self.time = 0.0

    def set_target(self, x):
        self.rig.ac_target.fill_(x)

    def step(self, n=1):
        with capture_tendon_errors(self.errors):
            for _ in range(n):
                self.s0.clear_forces()
                self.rig.apply(self.s0)
                if self.use_contacts:
                    self.pipeline.collide(self.s0, self.contacts)
                self.solver.step(self.s0, self.s1, self.control, self.contacts, self.cfg.dt)
                self.s0, self.s1 = self.s1, self.s0
        self.time += n * self.cfg.dt

    def read(self):
        """Joint angles [rad], slider displacement [m], servo force [N], span tensions [N].

        ``tension`` is the solver diagnostic (stretch / compliance); ``lam_force`` is the
        XPBD multiplier over dt; ``F_act`` is the servo's own force, an independent
        measurement of the base tension when the slider is at rest.
        """
        self.s1.clear_forces()
        self.rig.apply(self._probe())
        out = dict(theta=self.rig.sp_theta.numpy().copy(), x=float(self.rig.ac_x.numpy()[0]), F_act=float(self.rig.ac_force.numpy()[0]))
        if self.p.attach_tendon:
            mt = self.solver.tendon_segment_material_tension.numpy()
            dt = self.solver.tendon_segment_damping_tension.numpy()
            out["tension"] = np.maximum(mt + dt, 0.0)
            out["lam_force"] = -self.solver.tendon_segment_lambda.numpy() / self.cfg.dt
        return out

    def _probe(self):
        """Scratch state holding the current pose, so reading never touches s0.body_f."""
        self.s1.body_q.assign(self.s0.body_q)
        self.s1.body_qd.assign(self.s0.body_qd)
        return self.s1

    def ramp(self, x_from, x_to, seconds, record_every=8):
        n = max(1, round(seconds / self.cfg.dt))
        rows = []
        for i in range(1, n + 1):
            self.set_target(x_from + (x_to - x_from) * i / n)
            self.step()
            if i % record_every == 0 or i == n:
                r = self.read()
                r["t"] = self.time
                rows.append(r)
        return rows
