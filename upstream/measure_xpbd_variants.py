"""Pendulum period and joint convergence of SolverXPBD for whichever Newton tree is on PYTHONPATH.

    PYTHONPATH=/path/to/newton python upstream/measure_xpbd_variants.py label

Prints one JSON object:
  * pendulum: period / theory for a hinged rod at the default relaxation factors,
  * for three of Newton's articulated examples run with ``--solver xpbd`` at their own default
    iteration counts: the positional error of every joint whose linear axes are all locked
    (distance between the parent and child anchors) and the hinge-axis misalignment of
    revolute joints, as maximum over the run and at the last frame.
"""

from __future__ import annotations

import importlib
import json
import sys

import numpy as np
import warp as wp

import newton
import newton.examples

wp.config.quiet = True


def pendulum(dt=1.0 / 600.0, half=0.15, theta0=0.1):
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))
    rod = b.add_link(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
    b.add_shape_box(rod, xform=wp.transform((0.0, 0.0, -half), wp.quat_identity()), hx=0.01, hy=0.01, hz=half)
    b.add_articulation([b.add_joint_revolute(-1, rod, axis=(0.0, 1.0, 0.0), target_ke=0.0, target_kd=0.0)])
    b.joint_q[0] = theta0
    m = b.finalize()
    s0, s1, c = m.state(), m.state(), m.control()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
    solver = newton.solvers.SolverXPBD(m)
    th, sep = [], 0.0
    for _ in range(int(2.2 / dt)):
        s0.clear_forces()
        solver.step(s0, s1, c, None, dt)
        s0, s1 = s1, s0
        q = s0.body_q.numpy()[0]
        th.append(2.0 * np.arctan2(q[4], q[6]))
        sep = max(sep, float(np.linalg.norm(q[:3])))  # the hinge is at the body origin and the world origin
    th = np.array(th)
    zc = [(i - th[i] / (th[i + 1] - th[i])) * dt for i in range(len(th) - 1) if th[i] > 0.0 >= th[i + 1]]
    mass, inertia = float(m.body_mass.numpy()[0]), float(m.body_inertia.numpy()[0][1, 1])
    theory = 2 * np.pi * np.sqrt((inertia + mass * half**2) / (mass * 9.81 * half)) * (1 + theta0**2 / 16)
    return dict(period_ratio=float(np.mean(np.diff(zc)) / theory), hinge_separation_max_mm=sep * 1e3)


def rot(q, v):
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def joint_errors(model, state):
    """Anchor separation [m] of linear-locked joints and hinge-axis misalignment [rad] of revolute joints."""
    jt = model.joint_type.numpy()
    par, chi = model.joint_parent.numpy(), model.joint_child.numpy()
    Xp, Xc = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    qd_start, axis = model.joint_qd_start.numpy(), model.joint_axis.numpy()
    bq = state.body_q.numpy()
    locked = {int(newton.JointType.REVOLUTE), int(newton.JointType.FIXED), int(newton.JointType.BALL)}
    pos, ang = [], []
    for j in range(model.joint_count):
        if int(jt[j]) not in locked or chi[j] < 0:
            continue
        pp, pq = (Xp[j, :3], Xp[j, 3:]) if par[j] < 0 else (bq[par[j], :3] + rot(bq[par[j], 3:], Xp[j, :3]), None)
        cp = bq[chi[j], :3] + rot(bq[chi[j], 3:], Xc[j, :3])
        pos.append(float(np.linalg.norm(pp - cp)))
        if int(jt[j]) == int(newton.JointType.REVOLUTE):
            a = axis[qd_start[j]]
            ap = rot(Xp[j, 3:], a) if par[j] < 0 else rot(bq[par[j], 3:], rot(Xp[j, 3:], a))
            ac = rot(bq[chi[j], 3:], rot(Xc[j, 3:], a))
            ang.append(float(np.arccos(np.clip(np.dot(ap, ac) / (np.linalg.norm(ap) * np.linalg.norm(ac)), -1.0, 1.0))))
        _ = pq
    return np.array(pos), np.array(ang)


def run_example(name, frames):
    mod = importlib.import_module(f"newton.examples.{name}")
    args = newton.examples.create_parser().parse_args([])
    args.solver = "xpbd"
    args.world_count = 4  # used by basic_urdf (replicated quadrupeds); ignored by the others
    ex = mod.Example(newton.viewer.ViewerNull(), args)
    solver = ex.solver
    pmax, amax, pos, ang = 0.0, 0.0, np.zeros(0), np.zeros(0)
    for _ in range(frames):
        ex.step()
        pos, ang = joint_errors(ex.model, ex.state_0)
        if pos.size:
            pmax = max(pmax, float(pos.max()))
        if ang.size:
            amax = max(amax, float(ang.max()))
    finite = bool(np.isfinite(ex.state_0.body_q.numpy()).all())
    return dict(
        solver=type(solver).__name__, iterations=int(getattr(solver, "iterations", -1)), substeps=int(getattr(ex, "sim_substeps", -1)),
        joints_measured=int(pos.size), frames=frames, finite=finite,
        anchor_error_max_mm=pmax * 1e3, anchor_error_final_rms_mm=float(np.sqrt((pos**2).mean()) * 1e3) if pos.size else None,
        hinge_misalignment_max_mrad=amax * 1e3,
    )  # fmt: skip


if __name__ == "__main__":
    out = dict(label=sys.argv[1] if len(sys.argv) > 1 else "", newton=newton.__version__, warp=wp.__version__, pendulum=pendulum())
    for name, frames in (("basic.example_basic_pendulum", 300), ("basic.example_basic_joints", 300), ("basic.example_basic_urdf", 300)):
        try:
            out[name] = run_example(name, frames)
        except Exception as e:  # report, do not hide
            out[name] = dict(error=f"{type(e).__name__}: {e}"[:200])
    print("RESULT " + json.dumps(out))
