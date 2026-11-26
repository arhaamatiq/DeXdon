"""Per-pinhole capstan ratios from a cycle file, against exp(-mu * theta/2) at the measured angles.

    python experiments/capstan_check.py results/cycle_base.npz 0.3 0.010
"""
import sys

import numpy as np

z = np.load(sys.argv[1])
mu, x_at = float(sys.argv[2]), float(sys.argv[3])
key = f"mu{mu:g}/"
x, th, T, ld = z[key + "x"], z[key + "theta"], z[key + "tension"], z[key + "loading"]
i = int(np.argmin(np.abs(np.where(ld, x, np.inf) - x_at)))  # loading-branch sample nearest x_at
sim = T[i, 1:] / T[i, :-1]  # ratio across pinholes A1, B1, A2, B2, A3
ref = np.exp(-mu * 0.5 * np.repeat(th[i], 2)[:5])
print(f"mu={mu:g} travel={x[i] * 1e3:.2f} mm theta={th[i].round(4)} span tensions [N]={T[i].round(3)}")
print("pinhole      :", "    A1     B1     A2     B2     A3")
print("sim ratio    :", " ".join(f"{v:.4f}" for v in sim))
print("exp(-mu b)   :", " ".join(f"{v:.4f}" for v in ref))
print("max |diff|   :", f"{np.abs(sim - ref).max():.4f}")
