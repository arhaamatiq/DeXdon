"""Replay evaluation worlds one frame at a time and report the first non-finite quantity.

    python experiments/triage_divergence.py results/hand_eval.npz --diverged 3 --finite 1

For each replayed world prints, per phase, the joint-angle range, peak body speed, peak
tendon tension and object height, then the first frame at which anything is non-finite
and which arrays broke (links, sliders, table, object, tendon tension, slider travel).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--diverged", type=int, default=3)
    ap.add_argument("--finite", type=int, default=1)
    ap.add_argument("--obj", default=None)
    ap.add_argument("--substeps", type=int, default=64)
    ap.add_argument("--no-drill", action="store_true")
    ap.add_argument("--collide-every", type=int, default=4)
    ap.add_argument("--cfg", type=json.loads, default={})
    ap.add_argument("--contacts", action="store_true", help="list contact pairs per substep in the drilled frame")
    a = ap.parse_args()
    import warp as wp

    wp.config.quiet = True
    from dexdon import hand as H
    from dexdon.finger import SolverCfg, capture_tendon_errors

    z = np.load(a.npz)
    trials = json.loads(str(z["trials"]))
    nf = ~np.isfinite(z["z_after_shake"]) | ~np.isfinite(z["z_final"])
    ok = [i for i, t in enumerate(trials) if t["attach_tendon"] and (a.obj is None or t["obj"] == a.obj)]
    pick = [i for i in ok if nf[i]][: a.diverged] + [i for i in ok if not nf[i]][: a.finite]
    fields = set(H.Trial.__dataclass_fields__)
    tl = [H.Trial(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in trials[i].items() if k in fields}) for i in pick]
    hb = H.HandBatch(tl, SolverCfg(**{"substeps": a.substeps, "iterations": 32, **a.cfg}), collide_every=a.collide_every)
    hb._warm = False
    n, B, fps = len(tl), hb.B, hb.cfg.fps
    tgt = np.zeros((n, 4), np.float32)
    first = [None] * n
    prev = hb.model.state()
    drilled = a.no_drill

    def drill(w):
        """Replay the failing frame one substep at a time; report the first body to break."""
        hb.s0.assign(prev)
        last = None
        for i in range(hb.cfg.substeps):
            hb.s0.clear_forces()
            hb.rig.apply(hb.s0)
            wp.launch(H.inertial_kernel, dim=hb.model.body_count, inputs=[hb.shake_mass, hb.accel, hb.grav_comp], outputs=[hb.s0.body_f], device=hb.model.device)
            if i % hb.collide_every == 0:
                hb.pipeline.collide(hb.s0, hb.contacts)
            hb.solver.step(hb.s0, hb.s1, hb.control, hb.contacts, hb.cfg.dt)
            hb.s0, hb.s1 = hb.s1, hb.s0
            q = hb.s0.body_q.numpy().reshape(n, B, 7)[w]
            qd = hb.s0.body_qd.numpy().reshape(n, B, 6)[w]
            bad = [b for b in range(B) if not (np.isfinite(q[b]).all() and np.isfinite(qd[b]).all())]
            if bad:
                print(f"    drill world {w}: substep {i} of {hb.cfg.substeps}: first non-finite bodies {[names[b] + str(b) for b in bad]}; rigid contacts in batch {int(hb.contacts.rigid_contact_count.numpy()[0])}")
                if last is not None:
                    print("    last finite substep, per body |v| [m/s]:", np.linalg.norm(last[1][:, :3], axis=1).round(2).tolist())
                    print("    last finite substep, per body |w| [rad/s]:", np.linalg.norm(last[1][:, 3:], axis=1).round(1).tolist())
                    print("    last finite joint angles [rad]:", hb.rig.sp_theta.numpy().reshape(n, 9)[w].round(3).tolist(), "object pos:", last[0][hb.obj_local, :3].round(4).tolist())
                return
            last = (q.copy(), qd.copy())
            if a.contacts:
                cnt = int(hb.contacts.rigid_contact_count.numpy()[0])
                sb = hb.model.shape_body.numpy()
                b0 = sb[hb.contacts.rigid_contact_shape0.numpy()[:cnt]]
                b1 = sb[hb.contacts.rigid_contact_shape1.numpy()[:cnt]]
                mine = (np.maximum(b0, b1) // B) == w
                loc0, loc1 = np.where(b0 >= 0, b0 % B, -1)[mine], np.where(b1 >= 0, b1 % B, -1)[mine]
                pairs = {}
                for x0, x1 in zip(loc0, loc1, strict=True):
                    key = tuple(sorted((int(x0), int(x1))))
                    pairs[key] = pairs.get(key, 0) + 1
                wn = np.linalg.norm(qd[:, 3:], axis=1)
                th = hb.rig.sp_theta.numpy().reshape(n, 9)[w]
                print(f"    sub {i:2d}: contacts(body pair: count) {pairs} | finger max |w| A {wn[1:4].max():.0f} B {wn[5:8].max():.0f} thumb {wn[9:12].max():.0f} rad/s | object |v| {np.linalg.norm(qd[hb.obj_local, :3]):.2f} m/s z {q[hb.obj_local, 2]:.4f} | theta B {th[3:6].round(2).tolist()}")
            if i % 8 == 0:
                print(f"    drill substep {i:2d}: max |v| {np.linalg.norm(qd[:, :3], axis=1).max():.2f} m/s (body {int(np.linalg.norm(qd[:, :3], axis=1).argmax())}), max |w| {np.linalg.norm(qd[:, 3:], axis=1).max():.1f} rad/s (body {int(np.linalg.norm(qd[:, 3:], axis=1).argmax())}), tension max {hb.solver.tendon_segment_material_tension.numpy().reshape(n, -1)[w].max():.1f} N")
    names = ["slider", "link", "link", "link"] * 3 + ["table", "object"]
    errors: list[str] = []
    for w, (i, t) in enumerate(zip(pick, tl, strict=True)):
        print(f"world {w}: eval index {i}, {t.obj} mu={t.mu} dx={t.obj_dx} theta0={t.theta0} h={t.h} recorded_nonfinite={bool(nf[i])}")
    with capture_tendon_errors(errors):
        frame = 0
        for phase, dur in H.PHASES.items():
            nfr = round(dur * fps)
            stats = [dict(thmin=9.0, thmax=-9.0, v=0.0, T=0.0) for _ in range(n)]
            for f in range(nfr):
                s = (f + 1) / nfr
                if phase == "close":
                    tgt[:, :3] = hb.x_close * s
                elif phase == "retract":
                    tgt[:, 3] = H.TABLE_DROP * s
                elif phase == "open":
                    tgt[:, :3] = hb.x_close * (1.0 - s)
                acc = (0.0, 0.0, 0.0)
                if phase == "shake":
                    ph = 2.0 * math.pi * H.SHAKE_HZ * f / fps
                    acc = (H.SHAKE_ACC * math.sin(ph), H.SHAKE_ACC * math.sin(ph + 0.5 * math.pi), 0.0)
                hb.accel.assign(np.array([acc], np.float32))
                hb.rig.ac_target.assign(tgt.ravel())
                prev.assign(hb.s0)
                hb.frame()
                frame += 1
                q = hb.s0.body_q.numpy().reshape(n, B, 7)
                qd = hb.s0.body_qd.numpy().reshape(n, B, 6)
                th = hb.rig.sp_theta.numpy().reshape(n, 9)
                T = hb.solver.tendon_segment_material_tension.numpy().reshape(n, -1)
                x = hb.rig.ac_x.numpy().reshape(n, 4)
                for w in range(n):
                    if first[w] is not None:
                        continue
                    bad_bodies = [f"{names[b]}{b}" for b in range(B) if not (np.isfinite(q[w, b]).all() and np.isfinite(qd[w, b]).all())]
                    bad = bad_bodies + (["tendon_tension"] if not np.isfinite(T[w]).all() else []) + (["slider_travel"] if not np.isfinite(x[w]).all() else [])
                    if bad:
                        first[w] = (frame, phase, f, bad)
                        print(f"  world {w}: FIRST NON-FINITE at frame {frame} ({phase} +{f / fps:.3f}s): {bad}")
                        if not drilled:
                            drilled = True
                            drill(w)
                            raise SystemExit(0)
                        continue
                    st = stats[w]
                    st["thmin"], st["thmax"] = min(st["thmin"], float(th[w].min())), max(st["thmax"], float(th[w].max()))
                    st["v"] = max(st["v"], float(np.abs(qd[w, :, :3]).max()))
                    st["T"] = max(st["T"], float(T[w].max()))
                    st["z"] = float(q[w, hb.obj_local, 2])
            for w in range(n):
                if first[w] is None or first[w][1] == phase:
                    st = stats[w]
                    print(f"  {phase:8s} world {w}: joint angle [{st['thmin']:+.2f}, {st['thmax']:+.2f}] rad, peak speed {st['v']:.2f} m/s, peak tension {st['T']:.1f} N, object z {st.get('z', float('nan')):.3f}")
    print("tendon ERROR lines captured:", len(errors), errors[:2])
    print("summary:", [(w, first[w][:3] if first[w] else None) for w in range(n)])


if __name__ == "__main__":
    main()
