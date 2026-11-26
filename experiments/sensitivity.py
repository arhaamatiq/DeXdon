"""Sensitivity table: does the hysteresis loop survive changes to the numerical knobs?

    python experiments/sensitivity.py results/cycle_base.npz results/cycle_dt2.npz ...

For every variant and friction value, prints the loop metrics and the largest
difference of each logged curve from the baseline (first file), interpolated on
actuator travel, then writes results/sensitivity.json and figures/sensitivity.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_cycle import INK, MUTED, interp_branch, load

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def curves(r, grid):
    ld = r["loading"]
    x, th, F = r["x"], r["theta"], r["F"]
    out = {}
    for name, m in (("load", ld), ("unload", ~ld)):
        out[name] = dict(
            F=interp_branch(x[m], F[m], grid),
            th=np.stack([interp_branch(x[m], th[m][:, i], grid) for i in range(3)], 1),
        )
    out["work_mJ"] = float(np.trapezoid(F[ld], x[ld]) + np.trapezoid(F[~ld], x[~ld])) * 1e3
    out["spread_gap"] = float(np.abs((out["load"]["th"][:, 0] - out["load"]["th"][:, 2]) - (out["unload"]["th"][:, 0] - out["unload"]["th"][:, 2])).max())
    out["force_gap"] = float(np.abs(out["load"]["F"] - out["unload"]["F"]).max())
    return out


def main():
    files = sys.argv[1:]
    root = Path(__file__).resolve().parents[1]
    base_meta, base = load(files[0])
    grid = np.linspace(0.1, 0.9, 200) * base_meta["x_max"]
    table = {}
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for fi, f in enumerate(files):
        meta, runs = load(f)
        tag = Path(f).stem.replace("cycle_", "")
        table[tag] = {"cfg": meta["cfg"]}
        for mu in sorted(runs):
            if mu not in base:
                continue
            c, b = curves(runs[mu], grid), curves(base[mu], grid)
            table[tag][f"{mu:g}"] = dict(
                loop_work_mJ=c["work_mJ"],
                force_gap_max_N=c["force_gap"],
                spread_gap_max_rad=c["spread_gap"],
                max_angle_diff_vs_base_rad=float(max(np.abs(c[k]["th"] - b[k]["th"]).max() for k in ("load", "unload"))),
                max_force_diff_vs_base_N=float(max(np.abs(c[k]["F"] - b[k]["F"]).max() for k in ("load", "unload"))),
                tendon_errors=int(runs[mu]["n_errors"]),
            )
            if mu == 0.3:
                r = runs[mu]
                kw = dict(color=SERIES[fi % 8], lw=2.4 if fi == 0 else 1.2, label=tag, zorder=3 if fi == 0 else 2)
                axs[0].plot(r["x"] * 1e3, r["F"], **kw)
                axs[1].plot(r["x"] * 1e3, r["theta"][:, 0] - r["theta"][:, 2], **kw)
    axs[0].set_title("actuator force loop, μ = 0.3", loc="left")
    axs[0].set_ylabel("actuator force [N]")
    axs[1].set_title("pose loop θ₁ − θ₃, μ = 0.3", loc="left")
    axs[1].set_ylabel("θ₁ − θ₃ [rad]")
    for ax in axs:
        ax.set_xlabel("actuator travel x [mm]")
    axs[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("The μ = 0.3 loop under halved dt, doubled iterations, more sweeps, tighter tolerances, other relaxations", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(root / "figures" / "sensitivity.png", dpi=170)
    (root / "results" / "sensitivity.json").write_text(json.dumps(table, indent=1))
    hdr = f"{'variant':10s} {'mu':>4s} {'work mJ':>9s} {'Fgap N':>8s} {'spread rad':>10s} {'d_angle':>8s} {'d_force':>8s} err"
    print(hdr)
    for tag, t in table.items():
        for mu, v in t.items():
            if mu == "cfg":
                continue
            print(f"{tag:10s} {mu:>4s} {v['loop_work_mJ']:9.3f} {v['force_gap_max_N']:8.3f} {v['spread_gap_max_rad']:10.4f} {v['max_angle_diff_vs_base_rad']:8.4f} {v['max_force_diff_vs_base_N']:8.3f} {v['tendon_errors']}")
    _ = MUTED


if __name__ == "__main__":
    main()
