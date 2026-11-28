"""System identification of the tendon finger: recover pinhole friction mu and joint
stiffness k from joint angles measured against tendon pull over one close-open cycle.

    python scripts/sysid.py                 # everything, RTX 4090
    python scripts/sysid.py --dev-only      # development pairs only (the held-out pairs are not simulated)

Ground truth comes from the Newton XPBD finger (dexdon/finger.py) at known (mu, k); the
"measurement" is its joint angles plus 0.5 degrees of Gaussian noise, with the actuator
pull force as the independent variable.

  Method A  gradient fit of the closed-form finger model (a Warp kernel) with wp.Tape.
            Newton's routed tendons are not differentiable, which the script demonstrates.
  Method B  brute force: a grid of (mu, k) simulated in parallel Newton worlds; the best
            trajectory match wins, refined by a quadratic fitted to the losses around it.

Writes results/sysid.json and prints the table that docs/results.md quotes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOISE_DEG = 0.5
DEV_PAIRS = [(0.20, 0.100), (0.35, 0.080), (0.12, 0.140)]  # used while developing
MU_RANGE, K_RANGE = (0.02, 0.60), (0.05, 0.20)  # brute-force grid
FD_STEPS = (1.0e-5, 1.0e-6)  # relative finite-difference steps for the gradient check


def heldout_pairs(seed=4090, n=5):
    """Five pairs drawn once from a fixed seed; never simulated with --dev-only."""
    rng = np.random.default_rng(seed)
    return [(round(float(rng.uniform(0.10, 0.50)), 4), round(float(rng.uniform(0.06, 0.16)), 4)) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--grid", type=int, default=64, help="grid points per axis (grid x grid worlds)")
    ap.add_argument("--t-ramp", type=float, default=4.0, help="seconds per half cycle")
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "sysid.json"))
    a = ap.parse_args()

    import warp as wp

    wp.config.quiet = True
    from dexdon.sysid import FingerBatch, FingerModel, brute_force, check_xpbd_tendon_gradient, finite_difference_grad, trajectory_at

    device = wp.get_device(a.device)
    truths = [("dev", p) for p in DEV_PAIRS] + ([] if a.dev_only else [("held-out", p) for p in heldout_pairs()])

    # ---- does the simulator give gradients? -------------------------------------------
    xpbd = check_xpbd_tendon_gradient(device=device)
    print("gradient through SolverXPBD routed tendons:", xpbd.get("tape_error") or xpbd.get("tape_gradient"))
    print(f"  finite-difference d(tip height)/d(mu) through the simulator: {xpbd['finite_difference']:.3e} m")

    # ---- one batch: the brute-force grid plus the ground-truth worlds -------------------
    g = a.grid
    mu_axis, k_axis = np.linspace(*MU_RANGE, g), np.linspace(*K_RANGE, g)
    mu_grid, k_grid = [v.ravel() for v in np.meshgrid(mu_axis, k_axis, indexing="ij")]
    mus = np.concatenate([mu_grid, [p[0] for _, p in truths]])
    ks = np.concatenate([k_grid, [p[1] for _, p in truths]])
    t0 = time.time()
    batch = FingerBatch(mus, ks, device=device)
    build_s = time.time() - t0
    t0 = time.time()
    cyc = batch.cycle(t_ramp=a.t_ramp)
    sim_s = time.time() - t0
    n_grid = g * g
    opening = cyc["opening"]
    bad = ~np.isfinite(cyc["theta"]).all(axis=(0, 2))
    print(f"simulated {len(mus)} worlds ({n_grid} grid + {len(truths)} ground truth) on {device}: build {build_s:.0f} s, cycle {sim_s:.0f} s, "
          f"tendon ERROR lines {len(batch.errors)}, non-finite worlds {int(bad.sum())}")  # fmt: skip

    rows = []
    for t_idx, (split, (mu_true, k_true)) in enumerate(truths):
        w = n_grid + t_idx
        pull, clean = cyc["pull"][:, w], cyc["theta"][:, w]
        rng = np.random.default_rng(1000 + int(round(mu_true * 1e4)) * 7 + int(round(k_true * 1e4)))
        meas = clean + rng.normal(0.0, np.radians(NOISE_DEG), clean.shape)

        # Method A: gradient fit of the closed-form model
        t0 = time.time()
        model = FingerModel(pull, opening.astype(np.int32), meas, device=device)
        mu_a, k_a, loss_a, _ = model.fit()
        fit_s = time.time() - t0
        pred_a = model.predict(mu_a, k_a)
        _, tape_grad = model.loss_and_grad(mu_a * 1.1, k_a * 0.9)
        # The model has a kink where a joint switches from sticking to reverse slip, so the
        # comparison is made at two finite-difference steps.
        grad_rel = []
        for rel in FD_STEPS:
            fd_grad = finite_difference_grad(model, mu_a * 1.1, k_a * 0.9, rel=rel)
            grad_rel.append(float(np.max(np.abs(tape_grad - fd_grad) / np.abs(fd_grad))))

        # Method B: brute force over the simulated grid
        t0 = time.time()
        bf = brute_force(cyc["pull"][:, :n_grid], cyc["theta"][:, :n_grid], opening, mu_grid, k_grid, (g, g), pull, meas)
        match_s = time.time() - t0
        pred_b = trajectory_at(cyc["pull"][:, bf["world"]], cyc["theta"][:, bf["world"]], opening, pull)

        rmse = lambda p: float(np.degrees(np.sqrt(((p - clean) ** 2).mean())))  # noqa: E731
        rows.append(dict(
            split=split, mu_true=mu_true, k_true=k_true,
            A=dict(mu=mu_a, k=k_a, mu_err_pct=100 * (mu_a / mu_true - 1), k_err_pct=100 * (k_a / k_true - 1), traj_rmse_deg=rmse(pred_a),
                   wall_s=fit_s, grad_vs_fd_max_rel=dict(zip([f"{s:g}" for s in FD_STEPS], grad_rel, strict=True))),
            B=dict(mu=bf["mu"], k=bf["k"], mu_err_pct=100 * (bf["mu"] / mu_true - 1), k_err_pct=100 * (bf["k"] / k_true - 1),
                   mu_grid=bf["mu_grid"], k_grid=bf["k_grid"], traj_rmse_deg=rmse(pred_b), match_s=match_s),
        ))  # fmt: skip

    print(f"\n{'split':8s} {'mu':>6s} {'k':>6s} | A: mu err   k err   rmse   wall | B: mu err   k err   rmse   match | grad vs FD (step 1e-5 / 1e-6)")
    for r in rows:
        A, B = r["A"], r["B"]
        print(f"{r['split']:8s} {r['mu_true']:6.4f} {r['k_true']:6.4f} | {A['mu_err_pct']:+7.2f}% {A['k_err_pct']:+6.2f}% {A['traj_rmse_deg']:5.3f}° {A['wall_s']:5.1f}s | "
              f"{B['mu_err_pct']:+7.2f}% {B['k_err_pct']:+6.2f}% {B['traj_rmse_deg']:5.3f}° {B['match_s']:5.1f}s | {' / '.join(f'{v:.1e}' for v in A['grad_vs_fd_max_rel'].values())}")  # fmt: skip
    for split in ("dev", "held-out"):
        sel = [r for r in rows if r["split"] == split]
        if not sel:
            continue
        for m in ("A", "B"):
            mu_e = max(abs(r[m]["mu_err_pct"]) for r in sel)
            k_e = max(abs(r[m]["k_err_pct"]) for r in sel)
            print(f"{split} method {m}: worst |mu error| {mu_e:.2f} %, worst |k error| {k_e:.2f} %, within 5 %: {mu_e < 5 and k_e < 5}")

    out = dict(
        device=str(device), noise_deg=NOISE_DEG, grid=g, grid_worlds=n_grid, t_ramp=a.t_ramp, frames=int(len(opening)),
        mu_range=MU_RANGE, k_range=K_RANGE, build_s=build_s, sim_s=sim_s, tendon_error_lines=len(batch.errors),
        nonfinite_worlds=int(bad.sum()), xpbd_gradient=xpbd, rows=rows,
    )  # fmt: skip
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
