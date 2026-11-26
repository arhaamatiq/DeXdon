"""Two closed-form checks of SolverXPBD joint handling (Newton main @ 7ee5484, CPU or CUDA).

Both effects were first reported by pulipakaa24: A is newton#4314, B is newton#4315.

A) Physical pendulum period. A uniform rod hinged at one end swings under gravity;
   theory: T = 2 pi sqrt(I_hinge / (m g d)) (1 + theta0^2 / 16).
   With the default joint_linear_relaxation=0.7 / joint_angular_relaxation=0.4 the
   period is about 9 % long, independent of the iteration count. With equal
   relaxations it matches theory.

B) Joint position drive stiffness. A hinged rod, no gravity, drive target 0 with
   target_ke = k, constant external torque tau about the hinge. Theory: theta = tau / k.
   The deflection shrinks as the iteration count grows.

    python upstream/repro_xpbd_joint_accuracy.py
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

wp.config.quiet = True
L, HX, DT = 0.3, 0.15, 1.0 / 3840.0


def rod(gravity, target_ke=0.0):
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, gravity))
    link = b.add_link(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
    b.add_shape_box(link, xform=wp.transform((0.0, 0.0, -HX), wp.quat_identity()), hx=0.01, hy=0.01, hz=HX)
    j = b.add_joint_revolute(-1, link, axis=(0.0, 1.0, 0.0), target_ke=target_ke, target_kd=0.0, target_q=0.0)
    b.add_articulation([j])
    return b, link


def pendulum(lin, ang, iterations, theta0=0.1, seconds=3.0):
    b, _ = rod(-9.81)
    b.joint_q[0] = theta0
    m = b.finalize()
    s0, s1, c = m.state(), m.state(), m.control()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
    solver = newton.solvers.SolverXPBD(m, iterations=iterations, joint_linear_relaxation=lin, joint_angular_relaxation=ang)
    jq, jqd = wp.zeros(1, dtype=float, device=m.device), wp.zeros(1, dtype=float, device=m.device)
    th = []
    for _ in range(int(seconds / DT)):
        s0.clear_forces()
        solver.step(s0, s1, c, None, DT)
        s0, s1 = s1, s0
        newton.eval_ik(m, s0, jq, jqd)
        th.append(float(jq.numpy()[0]))
    th = np.array(th)
    t = np.arange(len(th)) * DT
    zc = [t[i] - th[i] * DT / (th[i + 1] - th[i]) for i in range(len(th) - 1) if th[i] > 0.0 >= th[i + 1]]
    mass, inertia = float(m.body_mass.numpy()[0]), float(m.body_inertia.numpy()[0][1, 1])
    theory = 2 * np.pi * np.sqrt((inertia + mass * HX * HX) / (mass * 9.81 * HX)) * (1 + theta0**2 / 16)
    return float(np.mean(np.diff(zc))), float(theory)


def drive(iterations, k=2.0, tau=0.2, seconds=3.0, ang=0.4):
    b, link = rod(0.0, target_ke=k)
    m = b.finalize()
    s0, s1, c = m.state(), m.state(), m.control()
    solver = newton.solvers.SolverXPBD(m, iterations=iterations, joint_angular_relaxation=ang, angular_damping=0.5)
    wrench = np.zeros((m.body_count, 6), np.float32)
    wrench[link, 4] = tau
    w = wp.array(wrench, dtype=wp.spatial_vector, device=m.device)
    jq, jqd = wp.zeros(1, dtype=float, device=m.device), wp.zeros(1, dtype=float, device=m.device)
    tail = []
    n = int(seconds / DT)
    for i in range(n):
        s0.body_f.assign(w)
        solver.step(s0, s1, c, None, DT)
        s0, s1 = s1, s0
        if i > n * 0.8:
            newton.eval_ik(m, s0, jq, jqd)
            tail.append(float(jq.numpy()[0]))
    return float(np.mean(tail)), tau / k


if __name__ == "__main__":
    print(f"newton {newton.__version__}, warp {wp.__version__}, dt = 1/{round(1 / DT)} s")
    print("A) pendulum period [s]")
    for lin, ang, it in ((0.7, 0.4, 2), (0.7, 0.4, 16), (0.7, 0.7, 2), (1.0, 1.0, 2), (0.4, 0.7, 2)):
        T, th = pendulum(lin, ang, it)
        print(f"   linear_relaxation={lin} angular_relaxation={ang} iterations={it:2d}: sim {T:.4f}  theory {th:.4f}  ratio {T / th:.4f}")
    print("B) drive deflection under constant torque [rad] (theory tau/k)")
    for it in (1, 2, 4, 8, 16):
        th, ref = drive(it)
        print(f"   iterations={it:2d}: sim {th:.5f}  theory {ref:.5f}  effective stiffness / target_ke = {ref / th:.3f}")
