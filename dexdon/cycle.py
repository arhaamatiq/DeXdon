"""Close-open actuator cycle on the single finger. One run gives both the statics check and the hysteresis loop.

    python -m dexdon.cycle --mu 0 0.1 0.2 0.3 0.5 --out results/cycle_base.npz

The actuator target ramps 0 -> x_max -> 0 at constant speed (quasi-static), and
joint angles, slider travel, servo force and span tensions are logged throughout.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np


def run_cycle(job):
    import warp as wp

    wp.config.quiet = True
    from dexdon.finger import FingerParams, FingerSim, SolverCfg

    p = FingerParams(**job["params"])
    sim = FingerSim(p, SolverCfg(**job["cfg"]))
    t0 = time.time()
    n_rec = job.get("samples", 300)
    every = max(1, round(job["t_ramp"] / sim.cfg.dt / n_rec))
    rows = sim.ramp(0.0, job["x_max"], job["t_ramp"], every)
    n_load = len(rows)
    rows += sim.ramp(job["x_max"], 0.0, job["t_ramp"], every)
    out = dict(
        t=np.array([r["t"] for r in rows]),
        x=np.array([r["x"] for r in rows]),
        F=np.array([r["F_act"] for r in rows]),
        theta=np.array([r["theta"] for r in rows]),
        loading=np.arange(len(rows)) < n_load,
        n_errors=len(sim.errors),
        wall_s=time.time() - t0,
    )
    if p.attach_tendon:
        out["tension"] = np.array([r["tension"] for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--x-max", type=float, default=0.020)
    ap.add_argument("--t-ramp", type=float, default=4.0)
    ap.add_argument("--params", type=json.loads, default={}, help="FingerParams overrides (JSON)")
    ap.add_argument("--cfg", type=json.loads, default={}, help="SolverCfg overrides (JSON)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [dict(params={**a.params, "mu": mu}, cfg=a.cfg, x_max=a.x_max, t_ramp=a.t_ramp) for mu in a.mu]
    with mp.get_context("spawn").Pool(min(len(jobs), 6)) as pool:
        res = pool.map(run_cycle, jobs)
    flat = {"mu": np.array(a.mu), "meta": json.dumps(dict(params=a.params, cfg=a.cfg, x_max=a.x_max, t_ramp=a.t_ramp))}
    for mu, r in zip(a.mu, res, strict=True):
        for k, v in r.items():
            flat[f"mu{mu:g}/{k}"] = v
        print(f"mu={mu:g}: {len(r['x'])} samples, tendon ERRORs={r['n_errors']}, wall={r['wall_s']:.0f}s")
    np.savez(a.out, **flat)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
