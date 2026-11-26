"""Design-level bootstrap intervals for the hand-study comparisons (resampling the 32 designs of each population).

    python experiments/hand_stats.py results/hand_eval.npz
"""
import json
import sys

import numpy as np

z = np.load(sys.argv[1])
tr = json.loads(str(z["trials"]))
ok = z["grasp"] & z["release"]
pop = np.array([t["meta"]["pop"] for t in tr])
rank = np.array([t["meta"]["rank"] for t in tr])
mu = np.array([t["mu"] for t in tr])
rng = np.random.default_rng(0)


def per_design(p, sel=None):
    m = (pop == p) if sel is None else (pop == p) & sel
    return np.array([ok[m & (rank == r)].mean() for r in sorted(set(rank[pop == p]))])


def ci(a, b=None, n=4000):
    """Mean of a (or mean(a) - mean(b)) with a 95% bootstrap interval over designs."""
    out = []
    for _ in range(n):
        x = a[rng.integers(0, len(a), len(a))].mean()
        if b is not None:
            x -= b[rng.integers(0, len(b), len(b))].mean()
        out.append(x)
    lo, hi = np.percentile(out, [2.5, 97.5])
    point = a.mean() - (b.mean() if b is not None else 0.0)
    return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]"


print("invalid worlds:", int(z["invalid"].sum()), "of", len(tr))
print("friction-aware - friction-blind, all held-out mu:", ci(per_design("aware"), per_design("blind")))
for m in sorted(set(mu[pop == "blind"])):
    print(f"friction-aware - friction-blind at mu = {m:<4g}:", ci(per_design("aware", np.isclose(mu, m)), per_design("blind", np.isclose(mu, m))))
hi_mu, lo_mu = mu >= 0.4, mu <= 0.15
for p in ("blind", "aware"):
    d = per_design(p, hi_mu) - per_design(p, lo_mu)  # paired: same designs at high and low friction
    print(f"{p}: success(mu >= 0.4) - success(mu <= 0.15), paired by design:", ci(d))
for g in ("k1", "k2", "k3", "full"):
    print(f"{g:4s} success (both mu):", ci(per_design(g)))
for g in ("k2", "k3", "full"):
    print(f"{g:4s} - k1:", ci(per_design(g), per_design("k1")))
for g in ("k1", "k2", "k3"):
    print(f"{g:4s} - full:", ci(per_design(g), per_design("full")))
