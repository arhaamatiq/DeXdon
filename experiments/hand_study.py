"""Design search and held-out evaluation on the parallel hand harness. Every subcommand is one GPU batch.

    python experiments/hand_study.py explore                 # sanity sweep of closing travel
    python experiments/hand_study.py search --n 384 --n4 256 # random design search (seed 0)
    python experiments/hand_study.py eval                    # held-out objects and friction
    python experiments/hand_study.py report                  # figures + results/hand_metrics.json

Design variables (shared by the three fingers): pinhole depths h (3), spring stiffness k (3),
spring rest angles theta0 (3), pre-tension, and the closing travel of each actuator.
Actuator count k: 1 = one command for all fingers; 2 = fingers | thumb; 3 = one each;
"full" = one torque-limited position servo per joint (9), same links, springs and limits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MU_BLIND, MU_AWARE = 0.0, 0.3
HELDOUT_MU = (0.05, 0.15, 0.25, 0.4, 0.6)
TOP_K = 32


def sample_design(rng, k_act):
    d = dict(
        h=tuple(rng.uniform(0.005, 0.011, 3).round(5)),
        k=tuple(np.exp(rng.uniform(np.log(0.04), np.log(0.2), 3)).round(5)),
        theta0=tuple(rng.uniform(0.0, 0.35, 3).round(4)),
        pretension=round(float(rng.uniform(0.0, 0.002)), 5),
    )
    x = rng.uniform(0.012, 0.034, 3).round(5)
    if k_act == 1:
        d["x_close"] = (x[0], x[0], x[0])
    elif k_act == 2:
        d["x_close"] = (x[0], x[0], x[2])
    elif k_act == 3:
        d["x_close"] = tuple(x)
    else:  # fully actuated: 9 joint targets, no tendon
        d["x_close"] = (0.0, 0.0, 0.0)
        d["joint_cmd"] = tuple(rng.uniform(0.0, 1.3, 9).round(4))
        d["attach_tendon"] = False
    return {k: (tuple(float(v) for v in val) if isinstance(val, tuple) else val) for k, val in d.items()}


def run(trials, out, cfg=None, **harness):
    import warp as wp

    wp.config.quiet = True
    from dexdon.finger import SolverCfg
    from dexdon.hand import HAND_JOINT_RELAXATION, HandBatch

    cfg = cfg or {}
    t0 = time.time()
    hb = HandBatch(trials, SolverCfg(**{"substeps": 64, "iterations": 32, "joint_relaxation": HAND_JOINT_RELAXATION, **cfg}), **harness)
    print(f"built {len(trials)} worlds, {hb.model.body_count} bodies in {time.time() - t0:.0f}s on {hb.model.device}", flush=True)
    t0 = time.time()
    res = hb.run()
    wall = time.time() - t0
    keys = ("grasp", "release", "opened", "peak_force", "work", "z0", "z_after_shake", "z_final", "tendon_error", "theta_closed", "theta_final", "invalid", "peak_link_speed", "first_bad_step")
    np.savez(out, trials=json.dumps([asdict(t) for t in trials]), wall_s=wall, cfg=json.dumps(cfg), harness=json.dumps(harness), n_error_lines=len(hb.errors), **{k: res[k] for k in keys})
    print(f"{out}: {len(trials)} worlds, wall {wall:.0f}s, grasp {res['grasp'].mean():.3f}, release {res['release'].mean():.3f}, "
          f"tendon ERROR worlds {int(res['tendon_error'].sum())}, invalid (non-finite or blow-up) {res['invalid'].mean():.3f}", flush=True)  # fmt: skip
    return res


def load(path):
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z["trials"])), z


def explore(a):
    """Harness factorial: which combination of stops, pinhole setback and actuator speed stays valid?"""
    from dexdon.hand import TRAIN_OBJECTS, Trial

    rng = np.random.default_rng(1)
    designs = [sample_design(rng, 1) for _ in range(48)]
    trials = []
    for soft in (True, False):
        for pin_a, margin in ((0.006, 0.2), (0.010, 0.5)):
            for speed in (0.0, 0.0375):
                for i, d in enumerate(designs):
                    for mu in (0.0, 0.3):
                        for obj in TRAIN_OBJECTS:
                            cell = f"{'soft' if soft else 'hard'}_a{pin_a * 1e3:g}_{'slow' if speed else 'fast'}"
                            trials.append(Trial(**d, mu=mu, obj=obj, soft_stop=soft, pin_a=pin_a, stop_margin=margin, act_speed=speed, meta=dict(cell=cell, design=i)))
    res = run(trials, ROOT / "results" / "hand_factorial.npz", a.cfg)
    cell = np.array([t.meta["cell"] for t in trials])
    mu = np.array([t.mu for t in trials])
    ok = res["grasp"] & res["release"]
    for c in sorted(set(cell)):
        s0, s3 = (cell == c) & (mu == 0.0), (cell == c) & (mu == 0.3)
        print(f"{c:18s} invalid mu0 {res['invalid'][s0].mean():.3f} mu0.3 {res['invalid'][s3].mean():.3f} | cycle success mu0 {ok[s0].mean():.3f} mu0.3 {ok[s3].mean():.3f} | n {int(s0.sum())},{int(s3.sum())}", flush=True)


OLD = {"joint_relaxation": 0.8}  # the setting the first hand runs used
VARIANTS = {  # batch-level changes measured one at a time on the same worlds
    "base": dict(cfg={**OLD}, harness={}),
    "collide1": dict(cfg={**OLD}, harness=dict(collide_every=1)),
    "jointf": dict(cfg={**OLD}, harness=dict(servo_via_joint_f=True)),
    "dt2": dict(cfg={**OLD, "substeps": 128}, harness={}),
    "it2": dict(cfg={**OLD, "iterations": 64}, harness={}),
    "contact04": dict(cfg={**OLD, "rigid_contact_relaxation": 0.4}, harness={}),
    "angdamp20": dict(cfg={**OLD, "angular_damping": 20.0}, harness={}),
    "newton_defaults": dict(cfg={"joint_relaxation": 0.7, "joint_angular_relaxation": 0.4}, harness={}),  # run once per Newton tree
    "relax07": dict(cfg={"joint_relaxation": 0.7}, harness={}),
    "relax06": dict(cfg={"joint_relaxation": 0.6}, harness={}),
    "relax05": dict(cfg={"joint_relaxation": 0.5}, harness={}),
    "relax04": dict(cfg={"joint_relaxation": 0.4}, harness={}),
}


def rate_worlds():
    """Fixed world set for before/after rates: 96 tendon designs and 32 per-joint designs on all 7 objects."""
    from dexdon.hand import HELDOUT_OBJECTS, TRAIN_OBJECTS, Trial

    rng = np.random.default_rng(2)
    trials = []
    for i in range(96):
        d = sample_design(rng, 1)
        for mu in (0.0, 0.3):
            for obj in TRAIN_OBJECTS + HELDOUT_OBJECTS:
                trials.append(Trial(**d, mu=mu, obj=obj, meta=dict(group="tendon", design=i)))
    for i in range(32):
        d = sample_design(rng, 9)
        for obj in TRAIN_OBJECTS + HELDOUT_OBJECTS:
            trials.append(Trial(**d, mu=0.3, obj=obj, meta=dict(group="full", design=i)))
    return trials


def rate(a):
    """Invalid-world rate and first-failure breakdown for one harness variant."""
    from dexdon.hand import CODE_NAMES, NEVER

    v = VARIANTS[a.variant]
    trials = rate_worlds()
    res = run(trials, ROOT / "results" / f"hand_rate_{a.variant}{a.suffix}.npz", {**v["cfg"], **a.cfg}, **v["harness"])
    grp = np.array([t.meta["group"] for t in trials])
    mu = np.array([t.mu for t in trials])
    obj = np.array([t.obj for t in trials])
    inv, first = res["invalid"], res["first_bad_step"]
    nonfinite = ~np.isfinite(res["z_after_shake"]) | ~np.isfinite(res["z_final"])
    sub = v["cfg"].get("substeps", 64)
    print(f"VARIANT {a.variant}{a.suffix}: invalid {inv.mean():.4f} ({int(inv.sum())}/{len(inv)}), non-finite {nonfinite.mean():.4f}, tendon ERROR worlds {int(res['tendon_error'].sum())}")
    print("  tendon mu=0 %.4f | tendon mu=0.3 %.4f | per-joint %.4f" % (inv[(grp == "tendon") & (mu == 0.0)].mean(), inv[(grp == "tendon") & (mu == 0.3)].mean(), inv[grp == "full"].mean()))
    print("  by object:", {o: round(float(inv[obj == o].mean()), 3) for o in sorted(set(obj))})
    hit = first.min(1) < NEVER
    if hit.any():
        t_first = first[hit].min(1) / (60.0 * sub)
        which = [tuple(CODE_NAMES[c] for c in range(first.shape[1]) if first[w, c] == first[w].min()) for w in np.flatnonzero(hit)]
        counts = {}
        for k in which:
            counts[" + ".join(k)] = counts.get(" + ".join(k), 0) + 1
        print("  first value to break (same substep ties joined):", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
        bounds, names = np.cumsum([0.0, *[d for d in __import__("dexdon.hand", fromlist=["PHASES"]).PHASES.values()]]), list(__import__("dexdon.hand", fromlist=["PHASES"]).PHASES)
        ph = np.clip(np.searchsorted(bounds, t_first, side="right") - 1, 0, len(names) - 1)
        print("  phase of first break:", {names[i]: int((ph == i).sum()) for i in range(len(names)) if (ph == i).any()})


def search(a):
    from dexdon.hand import TRAIN_OBJECTS, Trial

    rng = np.random.default_rng(a.seed)
    trials = []
    for i in range(a.n):  # single-actuator population; also the 1-actuator group of the actuator-count comparison
        d = sample_design(rng, 1)
        for mu in (MU_BLIND, MU_AWARE):
            for obj in TRAIN_OBJECTS:
                trials.append(Trial(**d, mu=mu, obj=obj, meta=dict(group="k1", design=i)))
    for g, k_act in (("k2", 2), ("k3", 3), ("full", 9)):
        for i in range(a.n4):
            d = sample_design(rng, k_act)
            for obj in TRAIN_OBJECTS:
                trials.append(Trial(**d, mu=MU_AWARE, obj=obj, meta=dict(group=g, design=i)))
    run(trials, ROOT / "results" / "hand_search.npz", a.cfg)


def rank(trials, z, group, mu, n_limit=None):
    """Designs of a group ranked by training success at friction mu; ties -> lower peak force."""
    score, force, design = {}, {}, {}
    for i, t in enumerate(trials):
        m = t["meta"]
        if m["group"] != group or abs(t["mu"] - mu) > 1e-9 or (n_limit is not None and m["design"] >= n_limit):
            continue
        d = m["design"]
        score.setdefault(d, []).append(float(z["grasp"][i] and z["release"][i]))
        force.setdefault(d, []).append(float(z["peak_force"][i]))
        design[d] = t
    order = sorted(score, key=lambda d: (-np.mean(score[d]), np.mean(force[d])))
    return [(d, float(np.mean(score[d])), float(np.mean(force[d])), design[d]) for d in order]


def evaluate(a):
    from dexdon.hand import HELDOUT_OBJECTS, Trial

    trials_s, z = load(ROOT / "results" / "hand_search.npz")
    n4 = max(t["meta"]["design"] for t in trials_s if t["meta"]["group"] == "k2") + 1
    picks = {
        "blind": rank(trials_s, z, "k1", MU_BLIND)[:TOP_K],
        "aware": rank(trials_s, z, "k1", MU_AWARE)[:TOP_K],
        "k1": rank(trials_s, z, "k1", MU_AWARE, n_limit=n4)[:TOP_K],  # same budget as k2/k3/full
        "k2": rank(trials_s, z, "k2", MU_AWARE)[:TOP_K],
        "k3": rank(trials_s, z, "k3", MU_AWARE)[:TOP_K],
        "full": rank(trials_s, z, "full", MU_AWARE)[:TOP_K],
    }
    fields = ("h", "k", "theta0", "pretension", "x_close", "attach_tendon", "joint_cmd")
    trials = []
    for pop, lst in picks.items():
        mus = HELDOUT_MU if pop in ("blind", "aware") else (0.15, 0.4)
        for rnk, (d, sc, _, t) in enumerate(lst):
            base = {f: (tuple(t[f]) if isinstance(t[f], list) else t[f]) for f in fields}
            for mu in mus:
                for obj in HELDOUT_OBJECTS:
                    for dx in (0.0, 0.008):
                        trials.append(Trial(**base, mu=mu, obj=obj, obj_dx=dx, meta=dict(pop=pop, rank=rnk, design=d, train_score=sc)))
    for obj in HELDOUT_OBJECTS:  # negative control: same hand, tendons not attached
        trials.append(Trial(mu=0.3, obj=obj, attach_tendon=False, meta=dict(pop="detached", rank=0, design=-1, train_score=0.0)))
    print({p: [round(x[1], 2) for x in lst[:TOP_K]].count(1.0) for p, lst in picks.items()}, "perfect training scores among the picks", flush=True)
    run(trials, ROOT / "results" / f"hand_eval{a.suffix}.npz", a.cfg)


def report(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_cycle import INK, MUTED  # noqa: F401  (sets the shared chart style)

    from dexdon.hand import HELDOUT_OBJECTS

    trials, z = load(ROOT / "results" / f"hand_eval{a.suffix}.npz")
    ok = z["grasp"] & z["release"]
    pop = np.array([t["meta"]["pop"] for t in trials])
    mu = np.array([t["mu"] for t in trials])
    obj = np.array([t["obj"] for t in trials])
    metrics = {"n_worlds": len(trials), "tendon_error_worlds": int(z["tendon_error"].sum()), "wall_s": float(z["wall_s"]), "e3": {}, "e4": {}}
    nonfinite = z["invalid"]
    metrics["nonfinite_fraction"] = {p: float(nonfinite[pop == p].mean()) for p in sorted(set(pop))}
    metrics["detached_control"] = dict(n=int((pop == "detached").sum()), grasp=int(z["grasp"][pop == "detached"].sum()))
    for p in ("blind", "aware"):
        fin = (pop == p) & ~nonfinite
        metrics.setdefault("finite_only", {})[p] = dict(n=int(fin.sum()), success=float(ok[fin].mean()), by_mu={f"{m:g}": float(ok[fin & np.isclose(mu, m)].mean()) for m in HELDOUT_MU})
    metrics["by_mu"] = {p: {f"{m:g}": dict(success=float(ok[(pop == p) & np.isclose(mu, m)].mean()), grasp=float(z["grasp"][(pop == p) & np.isclose(mu, m)].mean()),
                                            opened=float(z["opened"][(pop == p) & np.isclose(mu, m)].mean()), n=int(((pop == p) & np.isclose(mu, m)).sum())) for m in HELDOUT_MU} for p in ("blind", "aware")}
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 0.08]))
    for ax, p, title in zip(axs[:2], ("blind", "aware"), ("designs chosen at μ = 0 (friction-blind)", "designs chosen at μ = 0.3 (friction-aware)"), strict=True):
        grid = np.array([[ok[(pop == p) & (obj == o) & np.isclose(mu, m)].mean() for m in HELDOUT_MU] for o in HELDOUT_OBJECTS])
        n_cell = int(((pop == p) & (obj == HELDOUT_OBJECTS[0]) & np.isclose(mu, HELDOUT_MU[0])).sum())
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white" if grid[i, j] > 0.55 else INK, fontsize=9)
        ax.set_xticks(range(len(HELDOUT_MU)), [f"{m:g}" for m in HELDOUT_MU])
        ax.set_yticks(range(len(HELDOUT_OBJECTS)), HELDOUT_OBJECTS)
        ax.set_xlabel("held-out pinhole friction μ")
        ax.set_title(title, loc="left")
        ax.grid(False)
        metrics["e3"][p] = dict(
            grid=grid.tolist(), objects=list(HELDOUT_OBJECTS), mu=list(HELDOUT_MU), trials_per_cell=n_cell,
            overall=float(ok[pop == p].mean()), grasp=float(z["grasp"][pop == p].mean()),
            release_given_grasp=float(z["release"][(pop == p) & z["grasp"]].mean()) if z["grasp"][pop == p].any() else None,
            peak_force_mean_N=float(z["peak_force"][pop == p].mean()), n=int((pop == p).sum()),
        )  # fmt: skip
    fig.colorbar(im, cax=axs[2], label="grasp-and-release success")
    fig.suptitle("Held-out objects × held-out friction: success of the top designs from each search", x=0.01, ha="left", fontsize=11)
    fig.savefig(ROOT / "figures" / f"friction_design{a.suffix}.png", dpi=170)

    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    groups, labels = ("k1", "k2", "k3", "full"), ("1", "2", "3", "9 (one per joint)")
    for m, c in ((0.15, "#86b6ef"), (0.4, "#0d366b")):
        ys = []
        for g in groups:
            sel = (pop == g) & np.isclose(mu, m)
            ys.append(float(ok[sel].mean()))
            metrics["e4"].setdefault(g, {})[f"{m:g}"] = dict(success=ys[-1], grasp=float(z["grasp"][sel].mean()), n=int(sel.sum()), peak_force_mean_N=float(z["peak_force"][sel].mean()))
        ax.plot(range(4), ys, "-o", color=c, ms=7, label=f"μ = {m:g}")
    ax.set_xticks(range(4), labels)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("actuators")
    ax.set_ylabel("grasp-and-release success, held-out objects")
    ax.set_title("Grasp-and-release success vs actuator count", loc="left")
    ax.legend(loc="best")
    fig.savefig(ROOT / "figures" / f"actuator_count{a.suffix}.png", dpi=170)
    (ROOT / "results" / f"hand_metrics{a.suffix}.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["explore", "search", "eval", "report", "rate"])
    ap.add_argument("--variant", default="base", choices=list(VARIANTS))
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--n4", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", type=json.loads, default={})
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    dict(explore=explore, search=search, eval=evaluate, report=report, rate=rate)[a.cmd](a)
