"""Statics check and hysteresis metrics from one close-open cycle file.

    python experiments/analyze_cycle.py results/cycle_base.npz --tag base

Writes figures/statics.png and figures/hysteresis.png (with a _<tag> suffix unless the tag is "base") and
results/metrics_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dexdon import analytic as A
from dexdon.finger import FingerParams

JOINT = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical: joint identity
MU_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # sequential: friction magnitude
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e9e8e3", "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "lines.linewidth": 2.0,
    }
)  # fmt: skip


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    runs = {}
    for mu in z["mu"]:
        key = f"mu{mu:g}/"
        runs[float(mu)] = {k[len(key) :]: z[k] for k in z.files if k.startswith(key)}
    return meta, runs


def branch(p, mu, xs, s):
    th, T0 = [], []
    for x in xs:
        t, _, t0 = A.solve_from_displacement(x, p.k, p.theta0, p.a, p.h, mu, s, p.compliance)
        th.append(t)
        T0.append(t0)
    return np.array(th), np.array(T0)


def interp_branch(x, y, grid):
    o = np.argsort(x)
    return np.interp(grid, x[o], y[o])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    meta, runs = load(a.npz)
    p = FingerParams(**meta["params"])
    x_max = meta["x_max"]
    root = Path(__file__).resolve().parents[1]
    (root / "figures").mkdir(exist_ok=True)
    mus = sorted(runs)
    metrics = {"source": a.npz, "meta": meta, "e1": {}, "e2": {}}
    grid = np.linspace(0.1 * x_max, 0.9 * x_max, 200)
    ana = {}
    for mu in mus:
        r = runs[mu]
        ld = r["loading"]
        x, th, F = r["x"], r["theta"], r["F"]
        # ---- loading branch vs closed-form statics --------------------------------
        m = ld & (x > 0.05 * x_max)
        th_a, T_a = branch(p, mu, x[m], +1.0)
        err = th[m] - th_a
        metrics["e1"][f"{mu:g}"] = dict(
            n_samples=int(m.sum()),
            angle_err_max_rad=float(np.abs(err).max()),
            angle_err_rms_rad=float(np.sqrt((err**2).mean())),
            angle_err_max_deg=float(np.degrees(np.abs(err).max())),
            angle_range_rad=float(th[m].max()),
            tension_err_max_rel=float(np.abs(F[m] / T_a - 1.0).max()),
            tension_err_rms_rel=float(np.sqrt(((F[m] / T_a - 1.0) ** 2).mean())),
            tendon_errors=int(r["n_errors"]),
        )
        # ---- loop metrics ---------------------------------------------------------
        xu, thu, Fu = x[~ld], th[~ld], F[~ld]
        W = float(np.trapezoid(F[ld], x[ld]) + np.trapezoid(Fu, xu))  # closed loop, J
        th_l_g = np.stack([interp_branch(x[ld], th[ld][:, i], grid) for i in range(3)], 1)
        th_u_g = np.stack([interp_branch(xu, thu[:, i], grid) for i in range(3)], 1)
        F_l_g, F_u_g = interp_branch(x[ld], F[ld], grid), interp_branch(xu, Fu, grid)
        tip_l = np.array([A.fingertip(t, p.L) for t in th_l_g])
        tip_u = np.array([A.fingertip(t, p.L) for t in th_u_g])
        ga_l, Ta_l = branch(p, mu, grid, +1.0)
        ga_u, Ta_u = branch(p, mu, grid, -1.0)
        ana[mu] = (ga_l, Ta_l, ga_u, Ta_u)
        late = grid < 0.45 * x_max  # unloading branch after the turnaround transition has finished
        metrics["e2"][f"{mu:g}"] = dict(
            loop_work_mJ=W * 1e3,
            loop_work_analytic_mJ=float(np.trapezoid(Ta_l - Ta_u, grid)) * 1e3,
            loop_work_sim_same_window_mJ=float(np.trapezoid(F_l_g - F_u_g, grid)) * 1e3,
            force_gap_max_N=float(np.abs(F_l_g - F_u_g).max()),
            joint_gap_max_rad=[float(v) for v in np.abs(th_l_g - th_u_g).max(0)],
            total_flexion_gap_max_rad=float(np.abs(th_l_g.sum(1) - th_u_g.sum(1)).max()),
            fingertip_gap_max_mm=float(np.linalg.norm(tip_l - tip_u, axis=1).max() * 1e3),
            unload_angle_err_max_rad=float(np.abs(th_u_g[late] - ga_u[late]).max()),
            unload_tension_err_max_rel=float(np.abs(F_u_g[late] / Ta_u[late] - 1.0).max()),
            tendon_errors=int(r["n_errors"]),
        )
    out = root / "results" / f"metrics_{a.tag}.json"
    out.write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: metrics[k] for k in ("e1", "e2")}, indent=1))
    if a.no_fig:
        return
    suffix = "" if a.tag == "base" else f"_{a.tag}"

    # ---------------- Figure: joint angles and base tension, sim vs analytic -----------
    show = [m for m in (0.0, 0.2, 0.5) if m in runs] or mus[:3]
    fig, axs = plt.subplots(1, len(show) + 1, figsize=(3.3 * (len(show) + 1), 3.3), constrained_layout=True)
    for ax, mu in zip(axs, show, strict=False):
        r = runs[mu]
        ld = r["loading"]
        xs = np.linspace(0, r["x"][ld].max(), 120)
        th_a, _ = branch(p, mu, xs, +1.0)
        step = max(1, ld.sum() // 22)
        for i in range(3):
            ax.plot(xs * 1e3, th_a[:, i], color=JOINT[i], label=f"joint {i + 1}")
            ax.plot(r["x"][ld][::step] * 1e3, r["theta"][ld][::step, i], "o", ms=4.2, mfc=SURFACE, mec=JOINT[i], mew=1.3)
        ax.set_title(f"μ = {mu:g}", loc="left")
        ax.set_xlabel("actuator travel x [mm]")
    axs[0].set_ylabel("joint angle [rad]")
    axs[0].plot([], [], color=MUTED, label="closed form")
    axs[0].plot([], [], "o", ms=4.2, mfc=SURFACE, mec=MUTED, mew=1.3, label="Newton XPBD")
    axs[0].legend(loc="upper left")
    ax = axs[-1]
    for mu, c in zip(mus, MU_RAMP, strict=False):
        r = runs[mu]
        ld = r["loading"]
        xs = np.linspace(0, r["x"][ld].max(), 120)
        _, T_a = branch(p, mu, xs, +1.0)
        step = max(1, ld.sum() // 16)
        ax.plot(xs * 1e3, T_a, color=c)
        ax.plot(r["x"][ld][::step] * 1e3, r["F"][ld][::step], "o", ms=4.2, mfc=SURFACE, mec=c, mew=1.3)
        ax.annotate(f"μ={mu:g}", (xs[-1] * 1e3, T_a[-1]), xytext=(3, 0), textcoords="offset points", va="center", color=MUTED, fontsize=8)
    ax.set_title("base tension, loading", loc="left")
    ax.set_xlabel("actuator travel x [mm]")
    ax.set_ylabel("tension [N]")
    ax.set_xlim(right=x_max * 1e3 * 1.16)
    fig.suptitle("Finger statics: simulated joint angles and tension follow the closed-form capstan model", x=0.01, ha="left", fontsize=11)
    fig.savefig(root / "figures" / f"statics{suffix}.png", dpi=170)

    # ---------------- Figure: hysteresis loops ----------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)
    for mu, c in zip(mus, MU_RAMP, strict=False):
        r = runs[mu]
        ga_l, Ta_l, ga_u, Ta_u = ana[mu]
        axs[0].plot(r["x"] * 1e3, r["F"], color=c, label=f"μ = {mu:g}")
        axs[0].plot(grid * 1e3, Ta_l, color=INK, lw=0.8, ls=(0, (4, 3)))
        axs[0].plot(grid * 1e3, Ta_u, color=INK, lw=0.8, ls=(0, (4, 3)))
        d = r["theta"][:, 0] - r["theta"][:, 2]
        axs[1].plot(r["x"] * 1e3, d, color=c)
        axs[1].plot(grid * 1e3, ga_l[:, 0] - ga_l[:, 2], color=INK, lw=0.8, ls=(0, (4, 3)))
        axs[1].plot(grid * 1e3, ga_u[:, 0] - ga_u[:, 2], color=INK, lw=0.8, ls=(0, (4, 3)))
        axs[2].plot(r["x"] * 1e3, r["theta"].sum(1), color=c)
    axs[0].plot([], [], color=INK, lw=0.8, ls=(0, (4, 3)), label="closed form (full slip)")
    axs[0].legend(loc="upper left")
    axs[0].set_title("actuator force: loading above, unloading below", loc="left")
    axs[0].set_ylabel("actuator force [N]")
    axs[1].set_title("pose: proximal minus distal joint angle", loc="left")
    axs[1].set_ylabel("θ₁ − θ₃ [rad]")
    axs[2].set_title("total flexion θ₁ + θ₂ + θ₃", loc="left")
    axs[2].set_ylabel("total flexion [rad]")
    for ax in axs:
        ax.set_xlabel("actuator travel x [mm]")
    fig.suptitle("Close-open cycle: pinhole friction opens a loop in force and in how flexion is shared between joints", x=0.01, ha="left", fontsize=11)
    fig.savefig(root / "figures" / f"hysteresis{suffix}.png", dpi=170)
    print("figures written")


if __name__ == "__main__":
    main()
