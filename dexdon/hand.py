"""Parallel grasp harness: a 3-finger tendon hand (2 fingers opposing 1 thumb), one design per world.

Palm frame formulation. The palm is bolted to the world and the test moves everything else:
  * lift  = the support table under the object retracts downward,
  * shake = every body receives the inertial force -m a(t) of a palm shaken with a(t).
Fingers hang from the palm (world -Z) and flex toward the hand centre. Each finger has its
own slider; fingers that share an actuator receive the same displacement command.

Trial timeline [s]: settle, close (ramp), hold, retract table, hold, shake -> grasp check,
open (ramp), wait -> release check.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton

from .finger import DriveRig, FingerParams, SolverCfg, add_finger, capture_tendon_errors, spring_kernel

PALM_Z, PALM_W, FINGER_GAP = 0.16, 0.090, 0.036  # open-finger gap 78 mm: every object fits with its offset
PALM_PLATE_Z = 0.128  # underside of the palm plate
OBJ_Z = 0.092  # object centre height; fingertips of the straight finger reach z = 0.06
TABLE_DROP = 0.08
CONTACT_MU = 0.5
PHASES = dict(settle=0.2, close=1.0, hold=0.3, retract=0.5, hold2=0.3, shake=1.0, open=1.0, wait=0.6)
SHAKE_ACC, SHAKE_HZ = 15.0, 3.0  # m/s^2 peak (about 1.5 g), Hz
F_LIMIT = 15.0  # actuator force limit per finger tendon [N]; identical in every configuration
ACT_SPEED = 0.0375  # actuator no-load speed limit [m/s]: servo damping = F_LIMIT / ACT_SPEED
SLIDER_MASS = 0.3  # [kg]; keeps the explicit servo damper stable (kd * dt / m = 0.35)
HAND_A = 0.006  # pinhole setback in the hand fingers [m], same as the finger rig
HAND_STOP_MARGIN = 0.2  # flexion stop this far before the pinhole-crossing angle [rad]
HAND_JOINT_RELAXATION = 0.6  # equal linear/angular joint relaxation for hand batches; 0.8 diverges (docs/results.md)
BLOWUP_SPEED = 10.0  # [m/s] no hand body moves this fast in a valid trial (slider ramps at 0.03 m/s)
TAU_LIMIT = 0.12  # fully actuated baseline: joint torque limit = F_LIMIT x 8 mm nominal moment arm [N m]

OBJECTS = {  # name: (kind, size [m], mass [kg])
    "cyl40": ("capsule", 0.020, 0.050),
    "sphere50": ("sphere", 0.025, 0.050),
    "box40": ("box", 0.020, 0.060),
    "cyl30": ("capsule", 0.015, 0.035),  # held-out below
    "sphere60": ("sphere", 0.030, 0.080),
    "box50": ("box", 0.025, 0.090),
    "cyl50": ("capsule", 0.025, 0.070),
}
TRAIN_OBJECTS = ("cyl40", "sphere50", "box40")
HELDOUT_OBJECTS = ("cyl30", "sphere60", "box50", "cyl50")


@dataclass
class Trial:
    """One world: a hand design, a friction level, an object and a command vector."""

    h: tuple = (0.008, 0.008, 0.008)
    k: tuple = (0.10, 0.10, 0.10)
    theta0: tuple = (0.0, 0.0, 0.0)
    pretension: float = 0.0
    x_close: tuple = (0.02, 0.02, 0.02)  # per finger: A, B, thumb [m]
    mu: float = 0.0
    obj: str = "cyl40"
    obj_dx: float = 0.0  # lateral object offset [m]
    mass_scale: float = 1.0  # object mass multiplier
    f_limit: float = F_LIMIT  # actuator force limit per finger [N]
    link_density: float = 1200.0  # phalanx density [kg/m^3]
    soft_stop: bool = False  # True: compliant explicit joint stops; False: hard XPBD limits
    pin_a: float = HAND_A  # pinhole setback [m]
    stop_margin: float = HAND_STOP_MARGIN
    act_speed: float = ACT_SPEED  # actuator no-load speed limit [m/s]; 0 = legacy fast servo
    shake_scale: float = 1.0  # shake amplitude multiplier
    attach_tendon: bool = True
    joint_cmd: tuple | None = None  # fully actuated baseline: 9 joint targets [rad]
    meta: dict = field(default_factory=dict)


@wp.kernel
def inertial_kernel(mass: wp.array[float], accel: wp.array[wp.vec3], comp: wp.array[wp.vec3], body_f: wp.array[wp.spatial_vector]):
    """Inertial load of the shaken palm, plus gravity compensation of the actuator sliders."""
    i = wp.tid()
    body_f[i] = body_f[i] + wp.spatial_vector(accel[0] * mass[i] + comp[i], wp.vec3(0.0))


@wp.kernel
def joint_servo_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    child: wp.array[int],
    parent: wp.array[int],
    parent_rot: wp.array[wp.quat],
    gain: wp.array[float],
    cmd: wp.array[float],
    rest: wp.array[float],
    blend: wp.array[float],
    tau_max: float,
    kd: float,
    body_f: wp.array[wp.spatial_vector],
):
    """Fully actuated baseline: torque-limited PD on every flexion joint (explicit)."""
    i = wp.tid()
    if gain[i] == 0.0:
        return
    c = child[i]
    p = parent[i]
    q_pj = parent_rot[i]
    w_p = wp.vec3(0.0)
    if p >= 0:
        q_pj = wp.transform_get_rotation(body_q[p]) * parent_rot[i]
        w_p = wp.spatial_bottom(body_qd[p])
    q_rel = wp.quat_inverse(q_pj) * wp.transform_get_rotation(body_q[c])
    theta = 2.0 * wp.atan2(q_rel[1], q_rel[3])
    axis = wp.quat_rotate(q_pj, wp.vec3(0.0, 1.0, 0.0))
    target = rest[i] + (cmd[i] - rest[i]) * blend[0]
    tau = -gain[i] * (theta - target) - kd * wp.dot(wp.spatial_bottom(body_qd[c]) - w_p, axis)
    tau = wp.clamp(tau, -tau_max, tau_max)
    wp.atomic_add(body_f, c, wp.spatial_vector(wp.vec3(0.0), axis * tau))
    if p >= 0:
        wp.atomic_add(body_f, p, wp.spatial_vector(wp.vec3(0.0), -axis * tau))


N_CODES = 5  # 0 slider, 1 tendon tension, 2 contact, 3 finger link, 4 object or table
CODE_NAMES = ("slider", "tendon_tension", "contact", "link", "object_or_table")
NEVER = 2147483647
LINK_W_MAX, TENSION_MAX = 500.0, 150.0  # [rad/s], [N]: far above anything a valid trial reaches


@wp.kernel
def tick_kernel(step: wp.array[int]):
    step[0] = step[0] + 1


@wp.kernel
def record_bodies_kernel(
    body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], B: int, obj_local: int, v_max: float, w_max: float,
    step: wp.array[int], first: wp.array2d[int],
):
    """First substep at which a body is non-finite or impossibly fast, per world and category. Runs in-graph."""
    i = wp.tid()
    w = i / B
    loc = i - w * B
    p = wp.transform_get_translation(body_q[i])
    r = wp.transform_get_rotation(body_q[i])
    v = wp.spatial_top(body_qd[i])
    om = wp.spatial_bottom(body_qd[i])
    s = wp.length_sq(p) + wp.length_sq(v) + wp.length_sq(om) + r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]
    bad = not wp.isfinite(s)
    code = 3
    if loc == 0 or loc == 4 or loc == 8:
        code = 0
    if loc >= obj_local - 1:
        code = 4
    if loc != obj_local:  # a dropped object legitimately falls fast
        if wp.length_sq(v) > v_max * v_max or wp.length_sq(om) > w_max * w_max:
            bad = True
    if bad:
        wp.atomic_min(first, w, code, step[0])


@wp.kernel
def record_tendon_kernel(
    tension: wp.array[float], guide_l: wp.array[int], guide_r: wp.array[int], guide_body: wp.array[int], B: int, t_max: float,
    step: wp.array[int], first: wp.array2d[int],
):
    i = wp.tid()
    b = wp.max(guide_body[guide_l[i]], guide_body[guide_r[i]])
    if b < 0:
        return
    if not wp.isfinite(tension[i]) or tension[i] > t_max:
        wp.atomic_min(first, b / B, 1, step[0])


@wp.kernel
def record_contact_kernel(
    count: wp.array[int], shape0: wp.array[int], shape1: wp.array[int], shape_body: wp.array[int],
    point0: wp.array[wp.vec3], point1: wp.array[wp.vec3], normal: wp.array[wp.vec3], B: int,
    step: wp.array[int], first: wp.array2d[int],
):
    i = wp.tid()
    if i >= count[0]:
        return
    b = wp.max(shape_body[shape0[i]], shape_body[shape1[i]])
    if b < 0:
        return
    s = wp.length_sq(point0[i]) + wp.length_sq(point1[i]) + wp.length_sq(normal[i])
    if not wp.isfinite(s):
        wp.atomic_min(first, b / B, 2, step[0])


@wp.kernel
def servo_joint_f_kernel(
    body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], child: wp.array[int], parent_pos: wp.array[wp.vec3],
    parent_rot: wp.array[wp.quat], ke: wp.array[float], kd: wp.array[float], fmax: wp.array[float], target: wp.array[float],
    dof: wp.array[int], x_out: wp.array[float], f_out: wp.array[float], joint_f: wp.array[float],
):
    """Same clamped PD as ``servo_kernel`` (world-mounted sliders), delivered through Control.joint_f."""
    i = wp.tid()
    c = child[i]
    axis = wp.quat_rotate(parent_rot[i], wp.vec3(-1.0, 0.0, 0.0))
    x = wp.dot(wp.transform_get_translation(body_q[c]) - parent_pos[i], axis)
    f = wp.clamp(ke[i] * (target[i] - x) - kd[i] * wp.dot(wp.spatial_top(body_qd[c]), axis), -fmax[i], fmax[i])
    x_out[i] = x
    f_out[i] = f
    joint_f[dof[i]] = f


def _mounts():
    qy = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)  # finger +X -> world -Z, palmar -> -X
    qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi)
    return [
        wp.transform(wp.vec3(0.5 * PALM_W, 0.5 * FINGER_GAP, PALM_Z), qy),
        wp.transform(wp.vec3(0.5 * PALM_W, -0.5 * FINGER_GAP, PALM_Z), qy),
        wp.transform(wp.vec3(-0.5 * PALM_W, 0.0, PALM_Z), qz * qy),  # thumb: palmar -> +X
    ]


def build_world(t: Trial):
    """Returns (builder, rig-with-local-indices, info) for one trial."""
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
    rig = DriveRig()
    cfg = newton.ModelBuilder.ShapeConfig(density=t.link_density, mu=CONTACT_MU)
    hand_shapes, sliders, servo_joints = [], [], []
    for f, mount in enumerate(_mounts()):
        p = FingerParams(
            a=t.pin_a, stop_margin=t.stop_margin, soft_stop=t.soft_stop, h=t.h, k=t.k, theta0=t.theta0, mu=t.mu, pretension=t.pretension, attach_tendon=t.attach_tendon,
            slider_mass=SLIDER_MASS if t.act_speed > 0 else 0.1, slider_ke=2.0e4, slider_kd=F_LIMIT / t.act_speed if t.act_speed > 0 else 60.0, slider_fmax=t.f_limit, link_density=t.link_density,
        )  # fmt: skip
        n0 = b.shape_count
        hnd = add_finger(b, rig, p, mount=mount, shape_cfg=cfg, key=f"f{f}_")
        sliders.append(hnd.slider_body)
        servo_joints.append(hnd.slider_joint)
        hand_shapes += list(range(n0, b.shape_count))
    kind, size, mass = OBJECTS[t.obj]
    mass *= t.mass_scale
    # support table on a vertical slider (servo index 3): positive command lowers it
    q_down = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.5 * math.pi)  # local -X -> world -Z
    table_top = OBJ_Z - size
    table = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, table_top - 0.01), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3) * 1e-3))
    ts = b.add_shape_box(table, hx=0.09, hy=0.09, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
    tj = b.add_joint_prismatic(
        -1, table, parent_xform=wp.transform(wp.vec3(0.0, 0.0, table_top - 0.01), q_down), child_xform=wp.transform(wp.vec3(0.0), q_down),
        axis=(-1.0, 0.0, 0.0), target_ke=0.0, target_kd=0.0, limit_lower=-0.02, limit_upper=0.3,
    )  # fmt: skip
    b.add_articulation([tj])
    servo_joints.append(tj)
    rig.add_servo(table, -1, (0.0, 0.0, table_top - 0.01), tuple(q_down), 4.0e4, 400.0)
    # palm plate (static): stops the object from being squeezed up between the fingers
    ps = b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, PALM_PLATE_Z + 0.008), wp.quat_identity()), hx=0.030, hy=0.04, hz=0.008,
                         cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))  # fmt: skip
    for s in hand_shapes:  # table and palm plate only touch the object; fingers pass through them and each other
        b.add_shape_collision_filter_pair(s, ts)
        b.add_shape_collision_filter_pair(s, ps)
    for i, si in enumerate(hand_shapes):
        for sj in hand_shapes[i + 1 :]:
            b.add_shape_collision_filter_pair(si, sj)
    ocfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=CONTACT_MU)
    inertia = wp.mat33(np.eye(3) * 0.4 * mass * size * size)
    obj = b.add_body(xform=wp.transform(wp.vec3(t.obj_dx, 0.0, OBJ_Z), wp.quat_identity()), mass=mass, inertia=inertia)
    if kind == "sphere":
        b.add_shape_sphere(obj, radius=size, cfg=ocfg)
    elif kind == "box":
        b.add_shape_box(obj, hx=size, hy=size, hz=size, cfg=ocfg)
    else:  # cylinder-like: capsule with its axis along world Y
        qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
        b.add_shape_capsule(obj, xform=wp.transform(wp.vec3(0.0), qx), radius=size, half_height=0.03, cfg=ocfg)
    return b, rig, dict(obj=obj, table=table, sliders=sliders, servo_joints=servo_joints, joints=b.joint_count)


class HandBatch:
    def __init__(self, trials: list[Trial], cfg: SolverCfg | None = None, device=None, collide_every=4, servo_via_joint_f=False):
        self.trials, self.collide_every = trials, collide_every
        self.cfg = cfg or SolverCfg(substeps=64, iterations=32, joint_relaxation=HAND_JOINT_RELAXATION)
        self.servo_via_joint_f = servo_via_joint_f
        main = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
        rigs = []
        for t in trials:
            wb, rig, info = build_world(t)
            main.add_world(wb)
            rigs.append(rig)
        self.B = wb.body_count  # bodies per world (identical topology)
        self.obj_local, self.n = info["obj"], len(trials)
        self.model = main.finalize(device=device)
        dev = self.model.device
        # merge per-world rigs into one, offsetting local body indices
        self.rig = DriveRig()
        for w, r in enumerate(rigs):
            for key in r.sp:
                vals = r.sp[key]
                if key in ("child", "parent"):
                    vals = [v + w * self.B if v >= 0 else -1 for v in vals]
                self.rig.sp[key] += vals
            for key in r.ac:
                vals = r.ac[key]
                if key in ("child", "parent"):
                    vals = [v + w * self.B if v >= 0 else -1 for v in vals]
                self.rig.ac[key] += vals
        self.rig.finalize(dev)
        self.n_sp, self.n_ac = 9, 4  # per world
        gain = np.zeros((self.n, 9), np.float32)
        cmd = np.zeros((self.n, 9), np.float32)
        for w, t in enumerate(trials):
            if t.joint_cmd is not None:
                gain[w], cmd[w] = 1.0, t.joint_cmd
        self.full_gain = wp.array(gain.ravel(), dtype=float, device=dev)
        self.full_cmd = wp.array(cmd.ravel(), dtype=float, device=dev)
        self.blend = wp.zeros(1, dtype=float, device=dev)
        self.any_full = bool(gain.any())
        self.accel = wp.zeros(1, dtype=wp.vec3, device=dev)
        scale = np.repeat(np.array([t.shake_scale for t in trials], np.float32), self.B)
        self.shake_mass = wp.array(self.model.body_mass.numpy() * scale, dtype=float, device=dev)
        comp = np.zeros((self.model.body_count, 3), np.float32)
        for w in range(self.n):
            idx = [w * self.B + sl for sl in info["sliders"]]
            comp[idx, 2] = self.model.body_mass.numpy()[idx] * 9.81  # the slider is the actuator's moving mass, not a payload
        self.grav_comp = wp.array(comp, dtype=wp.vec3, device=dev)
        self.solver = self.cfg.make_solver(self.model)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.errors: list[str] = []
        self.graph = None
        # in-graph first-failure recorder (no host syncs): first[w, code] = first bad substep
        self.step_count = wp.zeros(1, dtype=int, device=dev)
        self.first_bad = wp.full((self.n, N_CODES), NEVER, dtype=int, device=dev)
        qd_start = self.model.joint_qd_start.numpy()
        dofs = [int(qd_start[w * info["joints"] + j]) for w in range(self.n) for j in info["servo_joints"]]
        self.servo_dof = wp.array(dofs, dtype=int, device=dev)
        self.x_close = np.array([list(t.x_close) for t in trials], np.float32)

    def _frame(self):
        r = self.rig
        for i in range(self.cfg.substeps):
            self.s0.clear_forces()
            if self.servo_via_joint_f:
                wp.launch(spring_kernel, dim=r.n_sp, device=self.model.device,
                          inputs=[self.s0.body_q, self.s0.body_qd, r.sp_child, r.sp_parent, r.sp_prot, r.sp_k, r.sp_th0, r.sp_kd, r.sp_lo, r.sp_hi],
                          outputs=[r.sp_theta, self.s0.body_f])  # fmt: skip
                wp.launch(servo_joint_f_kernel, dim=r.n_ac, device=self.model.device,
                          inputs=[self.s0.body_q, self.s0.body_qd, r.ac_child, r.ac_ppos, r.ac_prot, r.ac_ke, r.ac_kd, r.ac_fmax, r.ac_target, self.servo_dof],
                          outputs=[r.ac_x, r.ac_force, self.control.joint_f])  # fmt: skip
            else:
                r.apply(self.s0)
            wp.launch(inertial_kernel, dim=self.model.body_count, inputs=[self.shake_mass, self.accel, self.grav_comp], outputs=[self.s0.body_f], device=self.model.device)
            if self.any_full:
                wp.launch(
                    joint_servo_kernel, dim=r.n_sp, device=self.model.device,
                    inputs=[self.s0.body_q, self.s0.body_qd, r.sp_child, r.sp_parent, r.sp_prot, self.full_gain, self.full_cmd, r.sp_th0, self.blend, TAU_LIMIT, 5.0e-5],
                    outputs=[self.s0.body_f],
                )  # fmt: skip
            if i % self.collide_every == 0:
                self.pipeline.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, self.cfg.dt)
            self.s0, self.s1 = self.s1, self.s0
            self._record()

    def _record(self):
        dev, m, c = self.model.device, self.model, self.contacts
        wp.launch(tick_kernel, dim=1, inputs=[self.step_count], device=dev)
        wp.launch(record_bodies_kernel, dim=m.body_count, device=dev,
                  inputs=[self.s0.body_q, self.s0.body_qd, self.B, self.obj_local, BLOWUP_SPEED, LINK_W_MAX, self.step_count], outputs=[self.first_bad])  # fmt: skip
        if m.tendon_segment_count:
            sv = self.solver
            wp.launch(record_tendon_kernel, dim=m.tendon_segment_count, device=dev,
                      inputs=[sv.tendon_segment_material_tension, sv.tendon_segment_active_guide_l, sv.tendon_segment_active_guide_r, m.tendon_guide_body, self.B, TENSION_MAX, self.step_count],
                      outputs=[self.first_bad])  # fmt: skip
        wp.launch(record_contact_kernel, dim=c.rigid_contact_max, device=dev,
                  inputs=[c.rigid_contact_count, c.rigid_contact_shape0, c.rigid_contact_shape1, m.shape_body, c.rigid_contact_point0, c.rigid_contact_point1,
                          c.rigid_contact_normal, self.B, self.step_count], outputs=[self.first_bad])  # fmt: skip

    def frame(self):
        if self.model.device.is_cuda and self.graph is None and self._warm:
            with wp.ScopedCapture(device=self.model.device) as cap:
                self._frame()
            self.graph = cap.graph
        elif self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._frame()
            self._warm = True

    def run(self, record_frames=False):
        """Runs the full timeline. Returns per-world result arrays."""
        self._warm = False
        fps = self.cfg.fps
        n, tgt = self.n, np.zeros((self.n, 4), np.float32)
        obj_idx = np.arange(n) * self.B + self.obj_local
        z0 = None
        peak_force = np.zeros(n)
        peak_speed = np.zeros(n)
        work = np.zeros(n)
        prev_x = np.zeros((n, 3))
        out, frames = {}, []
        t = 0.0
        with capture_tendon_errors(self.errors):
            for phase, dur in PHASES.items():
                nf = round(dur * fps)
                for f in range(nf):
                    s = (f + 1) / nf
                    if phase == "close":
                        tgt[:, :3] = self.x_close * s
                        self.blend.fill_(s)
                    elif phase == "retract":
                        tgt[:, 3] = TABLE_DROP * s
                    elif phase == "open":
                        tgt[:, :3] = self.x_close * (1.0 - s)
                        self.blend.fill_(1.0 - s)
                    a = (0.0, 0.0, 0.0)
                    if phase == "shake":
                        ph = 2.0 * math.pi * SHAKE_HZ * f / fps
                        a = (SHAKE_ACC * math.sin(ph), SHAKE_ACC * math.sin(ph + 0.5 * math.pi), 0.0)
                    self.accel.assign(np.array([a], np.float32))
                    self.rig.ac_target.assign(tgt.ravel())
                    self.frame()
                    t += 1.0 / fps
                    if phase in ("close", "hold", "retract", "hold2", "shake"):
                        F = self.rig.ac_force.numpy().reshape(n, 4)[:, :3]
                        x = self.rig.ac_x.numpy().reshape(n, 4)[:, :3]
                        peak_force = np.maximum(peak_force, F.sum(1))
                        work += (np.maximum(F, 0.0) * np.maximum(x - prev_x, 0.0)).sum(1)
                        prev_x = x
                    if record_frames:
                        frames.append(self.s0.body_q.numpy().copy())
                oz = self.s0.body_q.numpy()[obj_idx, 2]
                if phase == "hold":
                    z0 = oz.copy()
                    out["theta_closed"] = self.rig.sp_theta.numpy().reshape(n, 9).copy()
                elif phase == "shake":
                    out["z_after_shake"] = oz.copy()
                elif phase == "wait":
                    out["z_final"] = oz.copy()
                    out["theta_final"] = self.rig.sp_theta.numpy().reshape(n, 9).copy()
        th0 = np.array([list(tr.theta0) * 3 for tr in self.trials])
        finite = np.isfinite(out["z_final"]) & np.isfinite(out["z_after_shake"])
        out["grasp"] = finite & (out["z_after_shake"] > z0 - 0.03)
        opened = np.abs(out["theta_final"] - th0).max(1) < 0.15
        out["opened"] = opened
        out["release"] = out["grasp"] & finite & (out["z_final"] < z0 - 0.06) & opened
        out["peak_force"], out["work"], out["z0"] = peak_force, work, z0
        bad = np.zeros(n, bool)  # any tendon ERROR invalidates that world's trial
        for ln in self.errors:
            m = re.search(r"Tendon (\d+)", ln)
            if m:
                bad[int(m.group(1)) // 3] = True
        out["tendon_error"] = bad
        # a world is invalid if anything went non-finite or any hand body exceeded BLOWUP_SPEED:
        # some blow-ups renormalize back to finite values and would otherwise pass as physics
        first = self.first_bad.numpy()  # the only host read of the recorder, after the run
        out["first_bad_step"] = first
        out["peak_link_speed"] = peak_speed
        out["invalid"] = ~finite | (first.min(1) < NEVER) | bad
        out["grasp"] &= ~out["invalid"]
        out["release"] &= ~out["invalid"]
        if record_frames:
            out["frames"] = np.array(frames)
        return out
