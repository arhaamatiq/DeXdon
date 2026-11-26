"""Clips rendered from simulation logs (side view; the finger is planar).

    python experiments/clips.py curl                         # finger wrapping a cylinder
    python experiments/clips.py loops results/cycle_base.npz # mu = 0 vs mu = 0.5 close-open, with the loop
    python experiments/clips.py hands results/hand_eval.npz  # grid of hands from the evaluation set

Writes clips/*.gif (and .mp4 when ffmpeg is available through imageio-ffmpeg).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_cycle import INK, MUTED, SURFACE, load

from dexdon.finger import FingerParams

LINK, TENDON, OBST = "#2a78d6", "#eb6834", "#c9c8c2"


def rot(c, v):
    return np.array([v[0] * math.cos(c) + v[1] * math.sin(c), -v[0] * math.sin(c) + v[1] * math.cos(c)])


def finger_geometry(theta, x, p: FingerParams):
    """Link outlines (x, z corner loops) and the tendon polyline for a pose."""
    joints, cum, c = [np.zeros(2)], [], 0.0
    for i, L in enumerate(p.L):
        c += theta[i]
        cum.append(c)
        joints.append(joints[-1] + rot(c, (L, 0.0)))
    links = []
    for i, L in enumerate(p.L):
        corners = [(0.002, p.half_t), (L - 0.002, p.half_t), (L - 0.002, -p.half_t), (0.002, -p.half_t), (0.002, p.half_t)]
        links.append(np.array([joints[i] + rot(cum[i], v) for v in corners]))
    pts = [np.array([-p.a - p.slider_lead - x, -p.h[0]]), np.array([-p.a, -p.h[0]])]
    for i, L in enumerate(p.L):
        pts.append(joints[i] + rot(cum[i], (p.a, -p.h[i])))
        if i < len(p.L) - 1:
            pts.append(joints[i] + rot(cum[i], (L - p.a, -p.h[i + 1])))
    return links, np.array(pts), joints


def draw_finger(ax, p, color=LINK):
    patches = [ax.fill([], [], color=color, alpha=0.9, lw=0)[0] for _ in p.L]
    (tendon,) = ax.plot([], [], color=TENDON, lw=1.6, marker="o", ms=2.5)
    ax.fill([-0.075, -0.004, -0.004, -0.075], [0.012, 0.012, -0.016, -0.016], color="#e9e8e3", lw=0)  # palm block
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 0.115)
    ax.set_ylim(-0.105, 0.03)
    ax.axis("off")

    def update(theta, x):
        links, pts, _ = finger_geometry(theta, x, p)
        for patch, poly in zip(patches, links, strict=True):
            patch.set_xy(poly)
        tendon.set_data(pts[:, 0], pts[:, 1])

    return update


def save(anim, name, fps):
    (ROOT / "clips").mkdir(exist_ok=True)
    anim.save(ROOT / "clips" / f"{name}.gif", writer=animation.PillowWriter(fps=fps))
    try:
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        anim.save(ROOT / "clips" / f"{name}.mp4", writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    except Exception as e:  # mp4 is optional
        print("mp4 skipped:", e)
    print("wrote clips/" + name)


def curl():
    import warp as wp

    wp.config.quiet = True
    import newton
    from dexdon.finger import FingerSim, SolverCfg

    R, cx, cz = 0.022, 0.050, -0.040
    p = FingerParams(mu=0.2)

    def obstacle(b):
        qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
        b.add_shape_capsule(-1, xform=wp.transform(wp.vec3(cx, 0.0, cz), qx), radius=R, half_height=0.03, cfg=newton.ModelBuilder.ShapeConfig(mu=0.6))

    out = {}
    for name, obs in (("free", None), ("cylinder", obstacle)):
        sim = FingerSim(p, SolverCfg(substeps=64, iterations=32), obstacle=obs)
        rows = sim.ramp(0.0, 0.026, 3.0, record_every=64)
        rows += sim.ramp(0.026, 0.026, 0.5, record_every=64)
        out[name] = (np.array([r["theta"] for r in rows]), np.array([r["x"] for r in rows]), np.array([r["F_act"] for r in rows]))
        print(name, "final theta", out[name][0][-1].round(3), "tendon ERRORs", len(sim.errors))
    np.savez(ROOT / "results" / "curl.npz", **{f"{k}_{n}": v for k, vs in out.items() for n, v in zip(("theta", "x", "F"), vs, strict=True)})
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), facecolor=SURFACE)
    ups = []
    for ax, name in zip(axs, ("free", "cylinder"), strict=True):
        ups.append(draw_finger(ax, p))
        ax.set_title("free closing: joints share flexion" if name == "free" else "same tendon, cylinder in the way: the finger wraps", loc="left", fontsize=10, color=INK)
    axs[1].add_patch(plt.Circle((cx, cz), R, color=OBST, lw=0))
    n = len(out["free"][1])

    def frame(i):
        for up, name in zip(ups, ("free", "cylinder"), strict=True):
            up(out[name][0][i], out[name][1][i])

    save(animation.FuncAnimation(fig, frame, frames=n, interval=1000 / 60), "finger_curl", 60)


def loops(path):
    meta, runs = load(path)
    p = FingerParams(**meta["params"])
    mus = [m for m in (0.0, 0.5) if m in runs]
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.9), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    ups, dots = [], []
    colors = {0.0: "#86b6ef", 0.5: "#0d366b"}
    for ax, mu in zip(axs[:2], mus, strict=True):
        ups.append(draw_finger(ax, p, colors[mu]))
        ax.set_title(f"pinhole friction μ = {mu:g}", loc="left", fontsize=10, color=INK)
    ax = axs[2]
    for mu in mus:
        r = runs[mu]
        ax.plot(r["x"] * 1e3, r["theta"][:, 0] - r["theta"][:, 2], color=colors[mu], lw=2, label=f"μ = {mu:g}")
        dots.append(ax.plot([], [], "o", ms=7, mfc=SURFACE, mec=colors[mu], mew=2)[0])
    ax.set_xlabel("actuator travel x [mm]", color=MUTED)
    ax.set_ylabel("θ₁ − θ₃ [rad]", color=MUTED)
    ax.set_title("same travel, different pose", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    n = min(len(runs[m]["x"]) for m in mus)
    idx = np.arange(0, n, 3)

    def frame(k):
        i = idx[k]
        for up, dot, mu in zip(ups, dots, mus, strict=True):
            r = runs[mu]
            up(r["theta"][i], r["x"][i])
            dot.set_data([r["x"][i] * 1e3], [r["theta"][i, 0] - r["theta"][i, 2]])

    fig.tight_layout()
    save(animation.FuncAnimation(fig, frame, frames=len(idx), interval=1000 / 30), "hysteresis_loops", 30)


def hands(path):
    """Grid of hands from the evaluation set (side view), simulated with the stable harness."""
    import json

    import warp as wp

    wp.config.quiet = True
    from dexdon import hand as H

    z = np.load(path)
    trials = json.loads(str(z["trials"]))
    want = [("aware", 0, "cyl30"), ("aware", 0, "cyl50"), ("aware", 0, "sphere60"), ("aware", 1, "cyl30"), ("aware", 1, "cyl50"), ("aware", 1, "box50")]
    idx = [next(i for i, t in enumerate(trials) if t["meta"]["pop"] == p and t["meta"]["rank"] == r and t["obj"] == o and abs(t["mu"] - 0.25) < 1e-9 and t["obj_dx"] == 0.0) for p, r, o in want]
    fields = set(H.Trial.__dataclass_fields__)
    tl = [H.Trial(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in trials[i].items() if k in fields}) for i in idx]
    hb = H.HandBatch(tl)
    res = hb.run(record_frames=True)
    frames = res["frames"].reshape(len(res["frames"]), len(tl), hb.B, 7)
    np.savez(ROOT / "results" / "hands_clip.npz", frames=frames, grasp=res["grasp"], release=res["release"], invalid=res["invalid"], index=np.array(idx))
    print("grasp", res["grasp"], "release", res["release"], "invalid", res["invalid"])

    def rot(q, v):  # rotate vectors v (n, 3) by quaternion q (x, y, z, w)
        u, w = q[:3], q[3]
        return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)

    L, ht = FingerParams().L, FingerParams().half_t
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 7.2), facecolor=SURFACE)
    arts = []
    for ax, t, g, r in zip(axs.ravel(), tl, res["grasp"], res["release"], strict=True):
        ax.set_aspect("equal")
        ax.set_xlim(-0.085, 0.085)
        ax.set_ylim(-0.03, 0.2)
        ax.axis("off")
        ax.set_title(f"{t.obj}, μ = {t.mu:g}: {'held' if g else 'dropped'}{', released' if r else ''}", loc="left", fontsize=10, color=INK)
        ax.fill([-0.03, 0.03, 0.03, -0.03], [H.PALM_PLATE_Z, H.PALM_PLATE_Z, H.PALM_PLATE_Z + 0.016, H.PALM_PLATE_Z + 0.016], color="#e9e8e3", lw=0)
        ax.fill([-0.06, 0.06, 0.06, -0.06], [H.PALM_Z, H.PALM_Z, H.PALM_Z + 0.012, H.PALM_Z + 0.012], color="#e9e8e3", lw=0)
        links = [ax.fill([], [], color=LINK if b < 8 else "#1c5cab", alpha=0.9, lw=0)[0] for b in (1, 2, 3, 9, 10, 11)]
        (table,) = ax.plot([], [], color=MUTED, lw=3)
        kind, size, _ = H.OBJECTS[t.obj]
        obj = ax.fill([], [], color=TENDON, alpha=0.9, lw=0)[0]
        arts.append((links, table, obj, kind, size))
    ang = np.linspace(0, 2 * np.pi, 40)

    def frame(k):
        for w, (links, table, obj, kind, size) in enumerate(arts):
            q = frames[k, w]
            for patch, b in zip(links, (1, 2, 3, 9, 10, 11), strict=True):
                Lb = L[(b - 1) % 4]
                c = np.array([[0.002, 0, ht], [Lb - 0.002, 0, ht], [Lb - 0.002, 0, -ht], [0.002, 0, -ht]])
                p = q[b, :3] + rot(q[b, 3:], c)
                patch.set_xy(p[:, [0, 2]])
            tz = q[12, 2] + 0.01
            table.set_data([-0.08, 0.08], [tz, tz])
            o = q[13]
            if not np.isfinite(o).all():
                continue
            if kind == "box":
                c = np.array([[-size, 0, -size], [size, 0, -size], [size, 0, size], [-size, 0, size]])
                obj.set_xy((o[:3] + rot(o[3:], c))[:, [0, 2]])
            else:
                obj.set_xy(np.c_[o[0] + size * np.cos(ang), o[2] + size * np.sin(ang)])

    fig.suptitle("Six worlds from the held-out evaluation, side view: close, lift (support retracts), shake, open", x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(animation.FuncAnimation(fig, frame, frames=range(0, len(frames), 2), interval=1000 / 30), "hands_grid", 30)


if __name__ == "__main__":
    if sys.argv[1] == "curl":
        curl()
    elif sys.argv[1] == "hands":
        hands(sys.argv[2])
    else:
        loops(sys.argv[2])
