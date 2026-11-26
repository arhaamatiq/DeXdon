"""Pure-Newton repro for newton-physics/newton#5596 (no Isaac Lab).

4 replicated worlds, 8 free boxes each, dropped on a ground plane. SolverMuJoCo with
MuJoCo contacts and run_to_run determinism. Three separate processes: reference A,
reference B (no reset), and a treatment that resets only one world's boxes at step 100.
Worlds that were not reset should stay bitwise identical to the reference.

    python upstream/repro_5596.py --njmax 300
    python upstream/repro_5596.py --sweep            # njmax in 256 300 320 384 600
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

STEPS, RESET_AT, N_WORLDS, N_BOXES, DT = 200, 100, 4, 8, 1.0 / 120.0
NCONMAX = 96  # per world; this scene peaks at 64 narrowphase contacts per world


def simulate(out: Path, reset_world: int, njmax: int, device: str) -> None:
    import warp as wp

    import newton

    wp.config.quiet = True
    wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN  # global, as Isaac Lab's deterministic_mode does
    world = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    shapes = []
    for i in range(N_BOXES):
        pos = (0.06 * (i % 3) - 0.06, 0.06 * ((i // 3) % 3) - 0.06, 0.3 + 0.12 * i)
        body = world.add_body(xform=wp.transform(pos, wp.quat(0.2588, 0.0, 0.0, 0.9659)), mass=0.5)
        shapes.append(world.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg))
    # The issue's scene has ground contacts only (nefc peaks at 8 boxes x 4 contacts x 4 rows):
    # boxes of one world do not collide with each other.
    for i in range(N_BOXES):
        for j in range(i + 1, N_BOXES):
            world.add_shape_collision_filter_pair(shapes[i], shapes[j])
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(world, N_WORLDS, spacing=(2.0, 2.0, 0.0))
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverMuJoCo(
        model, njmax=njmax, nconmax=NCONMAX, disable_sensors=True, use_mujoco_contacts=True, deterministic=wp.DeterministicMode.RUN_TO_RUN
    )
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    q_init, qd_init = s0.joint_q.numpy().copy(), s0.joint_qd.numpy().copy()
    nq, nv = model.joint_coord_count // N_WORLDS, model.joint_dof_count // N_WORLDS

    def step():
        s0.clear_forces()
        solver.step(s0, s1, control, None, DT)

    graph = None
    if model.device.is_cuda and not os.environ.get("REPRO_NO_GRAPH"):
        with wp.ScopedCapture(device=model.device) as cap:
            step()
            s0.assign(s1)
        graph = cap.graph
        s0.joint_q.assign(q_init)
        s0.joint_qd.assign(qd_init)
        newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
        solver.reset(s0) if hasattr(solver, "reset") else None

    pos = np.zeros((STEPS, N_WORLDS, N_BOXES, 3), np.float32)
    nefc = np.zeros((STEPS, N_WORLDS), np.int32)
    nacon = np.zeros(STEPS, np.int32)
    ovf = 0
    for t in range(STEPS):
        if reset_world >= 0 and t == RESET_AT:
            q, qd = s0.joint_q.numpy(), s0.joint_qd.numpy()
            q[reset_world * nq : (reset_world + 1) * nq] = q_init[reset_world * nq : (reset_world + 1) * nq]
            qd[reset_world * nv : (reset_world + 1) * nv] = 0.0
            s0.joint_q.assign(q)
            s0.joint_qd.assign(qd)
            newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
        if graph is not None:
            wp.capture_launch(graph)
        else:
            step()
            s0.assign(s1)
        pos[t] = s0.body_q.numpy()[:, :3].reshape(N_WORLDS, N_BOXES, 3)
        nefc[t] = solver.mjw_data.nefc.numpy()[:N_WORLDS]
        nacon[t] = int(solver.mjw_data.nacon.numpy()[0])
        ovf |= int(np.bitwise_or.reduce(solver.mjw_data.overflow.numpy().ravel()))
    np.savez(out, pos=pos, nefc=nefc, nacon=nacon, overflow=np.array([ovf]), naconmax=np.array([solver.mjw_data.naconmax]))


def compare(njmax: int, reset_world: int, device: str) -> dict:
    tmp = Path(tempfile.mkdtemp())
    for name, w in (("ref_a", -1), ("ref_b", -1), ("treat", reset_world)):
        subprocess.run(
            [sys.executable, __file__, "--_child", str(tmp / f"{name}.npz"), str(w), str(njmax), device],
            check=True, stdout=subprocess.DEVNULL,
        )  # fmt: skip
    a, b, tr = (np.load(tmp / f"{n}.npz") for n in ("ref_a", "ref_b", "treat"))
    pa, pt = a["pos"], tr["pos"]

    def first(e):
        return next((t for t in range(STEPS) if not np.array_equal(pa[t, e], pt[t, e])), None)

    return {
        "njmax": njmax,
        "reset_world": reset_world,
        "reference_bitwise_reproducible": bool(np.array_equal(pa, b["pos"])),
        "first_divergent_step_per_world": [first(e) for e in range(N_WORLDS)],
        "max_position_diff_m_per_world": [float(np.abs(pa[:, e] - pt[:, e]).max()) for e in range(N_WORLDS)],
        "peak_nefc_per_world": a["nefc"].max(0).tolist(),
        "treat_first_nefc_diff_step_per_world": [
            next((t for t in range(STEPS) if a["nefc"][t, e] != tr["nefc"][t, e]), None) for e in range(N_WORLDS)
        ],
        "peak_nacon": int(max(a["nacon"].max(), tr["nacon"].max())),
        "naconmax": int(a["naconmax"][0]),
        "overflow_nonzero": bool(np.any(a["overflow"]) or np.any(tr["overflow"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--njmax", type=int, default=300)
    ap.add_argument("--reset-world", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--_child", nargs=4, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._child:
        simulate(Path(a._child[0]), int(a._child[1]), int(a._child[2]), a._child[3])
        return
    import importlib.metadata as md

    vers = {}
    for p in ("newton", "mujoco-warp", "mujoco-core", "mujoco", "warp-lang"):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    print(json.dumps(vers), flush=True)
    for nj in (256, 300, 320, 384, 600) if a.sweep else (a.njmax,):
        print(json.dumps(compare(nj, a.reset_world, a.device)), flush=True)


if __name__ == "__main__":
    main()
