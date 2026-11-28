# Results

Everything the README quotes is here, with the command that produced it. Raw data is in
`results/`.

Versions: Newton `main` at commit 7ee5484 (1.9.0.dev0; the 1.8.0 release lacks
`tendon_settle_atol`), Warp 1.19.0, MuJoCo Warp 4.0.1, Python 3.12.

Hardware: the single-finger runs, replays and clips ran on an Apple M1 Pro (Warp CPU
device), with five of the sensitivity variants on a cloud machine's CPU. The hand batches
and the #5596 check ran on one RTX 4090 (driver 595.112.07) with CUDA graph capture.

Solver settings unless stated otherwise: SolverXPBD at 60 fps × 64 substeps
(dt = 1/3840 s), 32 iterations, joint linear and angular relaxation both 0.8 for the single
finger and 0.6 for the hand, `tendon_max_sweeps` 256, `tendon_settle_tol` 1e-3,
`tendon_settle_atol` 1e-6, tendon span compliance 1e-5 m/N.

Setup:

```bash
git clone https://github.com/newton-physics/newton && git -C newton checkout 7ee5484
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install "./newton[examples]" matplotlib pillow imageio-ffmpeg
```



## Finger statics against the closed form

The finger has links of 45, 30 and 25 mm, pinholes at a = 6 mm and h = 8 mm, springs of
0.1 N·m/rad with zero rest angle, and no gravity. The actuator ramps from 0 to 20 mm in 4 s.
I compare the loading branch (travel above 1 mm) with `dexdon/analytic.py` at the measured
slider travel; the closed form includes tendon stretch.

```bash
python -m dexdon.cycle --mu 0 0.1 0.2 0.3 0.5 --cfg '{"substeps":64,"iterations":32}' --out results/cycle_base.npz
python experiments/analyze_cycle.py results/cycle_base.npz --tag base     # writes results/metrics_base.json, figures/statics.png, figures/hysteresis.png
```

| μ | samples | max joint angle [rad] | angle error max [rad] (deg) | angle error RMS [rad] | base tension error RMS | tendon ERRORs |
|---|---|---|---|---|---|---|
| 0 | 287 | 0.72 | 0.0035 (0.20°) | 0.0017 | 1.3 % | 0 |
| 0.1 | 287 | 0.78 | 0.0034 (0.20°) | 0.0014 | 1.2 % | 0 |
| 0.2 | 287 | 0.84 | 0.0034 (0.20°) | 0.0014 | 1.2 % | 0 |
| 0.3 | 287 | 0.89 | 0.0034 (0.20°) | 0.0014 | 1.2 % | 0 |
| 0.5 | 287 | 1.01 | 0.0034 (0.20°) | 0.0014 | 1.2 % | 0 |

- Base tension is read from the slider servo, not from the solver's tension diagnostic.
  Its largest relative error is 10–11 %, at 1 mm of travel where tension is about 0.4 N
  (start-up transient). Beyond 5 mm the largest error is 0.7 % (μ = 0.3).
- Peak actuator force at 19.5 mm travel: 7.51 / 8.34 / 9.31 / 10.41 / 13.08 N for
  μ = 0 / 0.1 / 0.2 / 0.3 / 0.5.
- Tension ratio across each pinhole on the loading branch, against `e^(−μθ/2)` at the
  measured angles (`python experiments/capstan_check.py results/cycle_base.npz 0.3 0.010`):

  | μ, travel | A1 | B1 | A2 | B2 | A3 | max diff |
  |---|---|---|---|---|---|---|
  | 0.3, 10.0 mm — sim | 0.9371 | 0.9365 | 0.9442 | 0.9444 | 0.9507 | 0.0008 |
  | 0.3, 10.0 mm — closed form | 0.9371 | 0.9371 | 0.9451 | 0.9451 | 0.9511 | |
  | 0.5, 18.0 mm — sim | 0.7934 | 0.7931 | 0.8598 | 0.8598 | 0.8941 | 0.0007 |
  | 0.5, 18.0 mm — closed form | 0.7937 | 0.7937 | 0.8604 | 0.8604 | 0.8941 | |

## Close-open cycle

Same runs as above: 0 → 20 mm → 0, 4 s each way. Loop metrics are taken on a grid over
10–90 % of travel; "work" is ∮F dx over the whole cycle.

| μ | loop work [mJ] | force gap max [N] | joint gap max θ1 / θ2 / θ3 [rad] | fingertip gap max [mm] | total-flexion gap max [rad] |
|---|---|---|---|---|---|
| 0 (control) | −0.65 | 0.07 | 0.005 / 0.001 / 0.004 | 0.32 | 0.0002 |
| 0.1 | 9.55 | 1.19 | 0.088 / 0.029 / 0.087 | 6.22 | 0.0015 |
| 0.2 | 18.55 | 2.12 | 0.131 / 0.074 / 0.131 | 9.41 | 0.0030 |
| 0.3 | 27.04 | 2.92 | 0.173 / 0.104 / 0.153 | 11.05 | 0.0044 |
| 0.5 | 43.88 | 4.63 | 0.223 / 0.144 / 0.162 | 12.00 | 0.0073 |

- Closed-form full-slip prediction of the work between 10 % and 90 % travel:
  8.90 / 17.89 / 27.06 / 46.32 mJ; simulation over the same window:
  8.23 / 16.23 / 23.55 / 37.61 mJ. The simulation is lower because after the
  turnaround the pinholes stick and then reverse one by one; the full-slip unloading
  formula does not describe that transition.
- Unloading branch vs closed form after the transition (travel < 45 %): angle error max
  0.0013 / 0.0012 / 0.0016 / 0.0029 rad for μ = 0.1 / 0.2 / 0.3 / 0.5; tension error max 2.8 %.
- Total flexion is the same on both branches to within 0.0073 rad: tendon length pins
  Σθ. Friction redistributes flexion between joints instead.

## Sensitivity of the finger results

```bash
python -m dexdon.cycle --mu 0 0.3 --cfg '{"substeps":128,"iterations":32}' --out results/cycle_dt2.npz
python -m dexdon.cycle --mu 0 0.3 --cfg '{"substeps":64,"iterations":64}'  --out results/cycle_it2.npz
python -m dexdon.cycle --mu 0 0.3 --cfg '{"substeps":64,"iterations":32,"tendon_max_sweeps":1024}' --out results/cycle_sweeps.npz
python -m dexdon.cycle --mu 0 0.3 --cfg '{"substeps":64,"iterations":32,"tendon_settle_tol":1e-4,"tendon_settle_atol":1e-7}' --out results/cycle_tol.npz
python -m dexdon.cycle --mu 0 0.3 --cfg '{"substeps":64,"iterations":32,"joint_relaxation":0.6}' --out results/cycle_relax06.npz
python -m dexdon.cycle --mu 0 0.3 --cfg '{"substeps":64,"iterations":32,"joint_relaxation":1.0}' --out results/cycle_relax10.npz
python -m dexdon.cycle --mu 0 0.3 --cfg '{"substeps":16,"iterations":8}' --out results/cycle_coarse.npz
python experiments/sensitivity.py results/cycle_base.npz results/cycle_dt2.npz results/cycle_it2.npz results/cycle_sweeps.npz results/cycle_tol.npz results/cycle_relax06.npz results/cycle_relax10.npz results/cycle_coarse.npz
```

All at μ = 0.3. "Δ vs base" is the largest pointwise difference of any joint angle, or of
the actuator force, from the baseline curves over both branches (10–90 % travel). "vs closed
form" is the loading-branch angle error from the statics section.

| variant | loop work [mJ] | force gap [N] | θ1−θ3 gap [rad] | Δ angle vs base [rad] | Δ force vs base [N] | angle error vs closed form max / RMS [rad] |
|---|---|---|---|---|---|---|
| base (64 × 32) | 27.04 | 2.918 | 0.3060 | – | – | 0.0034 / 0.0014 |
| dt / 2 (128 × 32) | 27.46 | 2.934 | 0.3060 | 0.0024 | 0.041 | 0.0035 / 0.0008 |
| 2× iterations (64 × 64) | 26.98 | 2.909 | 0.3057 | 0.0018 | 0.025 | 0.0035 / 0.0008 |
| sweeps 256 → 1024 | 27.05 | 2.918 | 0.3060 | 0.0001 | 0.003 | 0.0034 / 0.0014 |
| settle tol 1e-4, atol 1e-7 | 26.86 | 2.908 | 0.3049 | 0.0014 | 0.016 | 0.0034 / 0.0010 |
| relaxation 0.6 | 27.13 | 2.922 | 0.3057 | 0.0060 | 0.036 | 0.0059 / 0.0029 |
| relaxation 1.0 | 27.89 | 3.091 | 0.3497 | 0.0463 | 0.277 | 0.0483 / 0.0080 |
| Newton example defaults (16 × 8) | 52.42 | 7.879 | 0.1761 | 0.431 | 23.5 | 0.371 / 0.234 |

- Loop work relative to the baseline: +1.6 % (dt/2), −0.2 % (2× iterations), +0.0 % (4×
  sweeps), −0.7 % (tighter tolerances), +0.3 % (relaxation 0.6), +3.1 % (relaxation 1.0).
  All have 0 tendon ERROR diagnostics.
- Relaxation 1.0 is the outlier (+3.1 % work, 0.048 rad off the closed form on the
  loading branch above about 13 mm travel). It is less accurate than 0.6 and 0.8 against
  the closed form, so I take it as a solver-accuracy problem at that setting, not physics.
- The 16 × 8 setting used by Newton's tendon examples is not converged for this finger
  (5–9 g links, 1e-5 m/N spans): its loop work is 1.9× the converged value.

## Negative controls

| control | result | command |
|---|---|---|
| μ = 0 | loop work −0.65 mJ (base), −0.13 mJ (dt/2), −0.64 mJ (2× iterations); force gap 0.07 / 0.045 / 0.072 N against 2.92 N at μ = 0.3 | cycle commands above |
| tendon detached | max joint angle over the whole cycle 0.0 rad while the slider travels 19.98 mm | `python -m dexdon.cycle --mu 0.3 --params '{"attach_tendon": false}' --cfg '{"substeps":64,"iterations":32}' --t-ramp 1.0 --out results/cycle_detached.npz` |
| hand, tendons detached | 0 of 4 held-out objects held, 0 invalid (all three solver settings) | `python experiments/hand_study.py eval` (population `detached`) |

The μ = 0 loop is slightly *negative* (the solver returns 0.65 mJ more than it took),
and shrinks five-fold when dt is halved: it is discretization error and sets the
numerical floor, 2.4 % of the μ = 0.3 loop.

## Recovering friction and stiffness from a trajectory

If μ and k were not known, could they be read back from one close-open cycle? Both
directions are used so that the friction loop separates μ from k.

```bash
python scripts/sysid.py                          # RTX 4090, writes results/sysid.json
python -m pytest tests/test_sysid_gradients.py   # needs pytest
```

Ground truth is the Newton finger at a known (μ, k), with all three joints sharing k, on
the cycle used above: 0 → 20 mm → 0, 4 s each way, 64 substeps × 32 iterations. Once per
frame (480 frames) the run records the actuator's pull force and the three joint angles.
The measurement is those angles plus Gaussian noise of 0.5° standard deviation, from a
seed fixed per pair. The pull force is left noise-free.

Gradient fit. Newton's routed tendons do not give gradients. Finalizing the model with
`requires_grad=True` and stepping `SolverXPBD` under `wp.Tape` raises
`NotImplementedError: Routed tendon simulation is not differentiable; finalize the model
with requires_grad=False`. The sensitivity itself is there: a finite difference through
the simulator gives d(fingertip height)/dμ = −1.151e-3 m over the same 40-substep rollout.
So the fit runs on the closed-form finger from [derivations.md](derivations.md) instead,
written as a float64 Warp kernel in `dexdon/sysid.py`. Each sample solves
`k θ_i = T_i r_i(θ_i)` with `T_i = T0 e^(−μ w_i)` by 10 unrolled Newton iterations. On the
opening branch a joint holds the angle it had at peak pull until the reverse-slip solution
(the sign of μ flipped) falls below it, so `θ_i = min(θ_i at peak, reverse-slip θ_i)`. The
loss is the mean squared angle residual, `wp.Tape` returns its gradient in (μ, k), and Adam
runs 400 steps on (log μ, log k) from μ = 0.2, k = 0.12 N·m/rad.

Brute force. A 64 × 64 grid over μ ∈ [0.02, 0.60] and k ∈ [0.05, 0.20] N·m/rad
(spacing 0.0092 and 0.0024) is simulated as 4,096 Newton worlds in one batch, next to the
ground-truth worlds. Each library world's angles are interpolated to the measured pull
values, branch by branch, and the world with the smallest mean squared residual wins. The
estimate is the minimum of a quadratic fitted to the 3 × 3 block of losses around it.

Five held-out pairs were drawn once from a fixed seed (μ uniform in [0.10, 0.50], k in
[0.06, 0.16]). During development the script ran with `--dev-only`, which simulates three
other pairs and never the held-out ones. Nothing in either method was changed after the
held-out pairs were first simulated.

Held-out pairs, RTX 4090:

| method | μ error, worst / mean | k error, worst / mean | trajectory RMSE | wall time |
|---|---|---|---|---|
| gradient fit of the closed form | 1.51 % / 0.83 % | 0.31 % / 0.23 % | 0.096–0.132° | 0.8 s per fit |
| brute force, 4,096 simulated worlds | 0.41 % / 0.22 % | 0.11 % / 0.06 % | 0.047–0.302° | 34 s to build and 113 s to simulate the grid once, then 0.9 s per match |

Trajectory RMSE is against the noise-free ground truth, over all frames and joints. For
brute force it is the best grid world's trajectory, before the quadratic refinement.

Every pair:

| pair | μ | k [N·m/rad] | fit: μ error | fit: k error | fit: RMSE | brute force: μ error | brute force: k error | brute force: RMSE |
|---|---|---|---|---|---|---|---|---|
| development | 0.2000 | 0.1000 | +0.42 % | +0.30 % | 0.109° | −0.73 % | −0.01 % | 0.115° |
| development | 0.3500 | 0.0800 | +0.47 % | +0.16 % | 0.127° | −0.17 % | −0.04 % | 0.302° |
| development | 0.1200 | 0.1400 | +2.43 % | +0.40 % | 0.087° | +0.59 % | +0.02 % | 0.096° |
| held-out | 0.2323 | 0.0722 | +1.03 % | +0.29 % | 0.131° | −0.41 % | +0.07 % | 0.302° |
| held-out | 0.4615 | 0.1190 | +0.39 % | +0.19 % | 0.107° | +0.10 % | −0.04 % | 0.047° |
| held-out | 0.4272 | 0.1374 | +0.35 % | +0.20 % | 0.098° | −0.01 % | −0.11 % | 0.143° |
| held-out | 0.3868 | 0.1032 | +0.85 % | +0.16 % | 0.132° | +0.40 % | −0.08 % | 0.215° |
| held-out | 0.1694 | 0.1099 | +1.51 % | +0.31 % | 0.096° | +0.19 % | +0.01 % | 0.110° |

Both methods are inside 5 % on every pair. No world produced a tendon ERROR line or a
non-finite state.

What to keep in mind when reading these:

- The fit's errors are all positive, in μ and in k, and largest at low μ. That is model
  bias and not noise: the closed form is a simplification of the simulator, mostly in how
  the pinholes stick and release after the turnaround.
- Brute force is scored against the simulator that produced the ground truth, so it has
  no model error by construction. Its numbers show what the noise and the grid spacing
  cost, and nothing about how it would do on a finger the simulator does not match.
- The gradient check failed once at the usual step. The script compares the tape gradient
  with central finite differences at (1.1 μ, 0.9 k) of each fitted pair. With a relative
  step of 1e-5 the two agree to 1.1e-8 or better on seven pairs and differ by 4.0 % on
  held-out pair (0.4272, 0.1374). There, one sample (joint 3, one opening frame) sits
  within the step of the switch from sticking to reverse slip, where the model has a
  kink, and the finite difference straddles it. With a step of 1e-6 the agreement is
  2.6e-10 on that pair and 4.0e-9 or better on all eight. The pytest test checks five
  points on a synthetic cycle at the 1e-5 step, and the worst relative error there is
  2.8e-9.
- The script was run three times on the held-out pairs with `dexdon/sysid.py` unchanged,
  the third time only to add the second finite-difference step to the check above. The
  fit's errors were the same to the two decimals shown each time. The brute-force errors
  moved by up to 0.02 percentage points on held-out pairs and 0.06 on a development pair,
  because the GPU batch is not bitwise repeatable between runs. The table is the third
  run.
- The actuator target has to advance every substep. An earlier version of the batch moved
  it once per frame, and the servo force then read low because the slider was
  decelerating at the end of each frame. That was found and fixed on the development
  pairs.
- One noise draw per pair, one finger geometry, and the geometry (a, h, link lengths) is
  taken as known. Only μ and k are fitted.

## Solver behaviour I ran into

```bash
python upstream/repro_xpbd_joint_accuracy.py     # results/repro_xpbd_joint_accuracy.log
python upstream/repro_5596.py --sweep            # results/repro_5596.log (RTX 4090)
```

### Joint drive stiffness depends on the iteration count

This is newton#4315, reported by pulipakaa24 with a draft fix in #4318. I hit it
independently. A hinged rod with no gravity, `target_ke` = 2 N·m/rad and a constant
0.2 N·m torque should deflect 0.1000 rad. It deflects 0.2434 / 0.1152 / 0.0629 / 0.0314 /
0.0200 rad at 1 / 2 / 4 / 8 / 16 iterations (dt = 1/3840 s), an effective stiffness of
0.41× / 0.87× / 1.59× / 3.18× / 5.00× the requested value. That is why the return springs
and the actuator servo here are explicit forces.

### Unequal joint relaxation factors

This is newton#4314, also reported by pulipakaa24; #4317 fixes it by using the linear
factor for both parts of a positional row. My reproduction: a 0.3 m rod released from
0.1 rad should swing with a period of 0.8982 s. With the defaults (linear 0.7, angular 0.4)
the period is 0.9790 s, a ratio of 1.0900, at 2 and at 16 iterations. With equal factors
(0.7/0.7 or 1.0/1.0) it is 0.8982 s (1.0000). With the factors swapped (0.4/0.7) it is
0.8485 s (0.9447).

Two rougher checks from debugging, not scripted: in the 3-link finger without a tendon, a
1 N force at the distal link's COM produced 0.499–0.504× the expected joint torques at the
defaults and 0.987–1.014× with equal factors (16 × 16). In the tendon finger at μ = 0,
tendon torque over spring torque at the distal joint was 1.147 at the defaults and 1.002
with equal factors (64 × 64).

### newton#5596 did not reproduce

In a pure-Newton scene (4 worlds × 8 boxes, world 1 reset at step 100) with `njmax` ∈
{256, 300, 320, 384, 600} there were no overflow bits, `nacon` peaked at 256 of 384 and
`nefc` at 128 per world. The largest position difference of a non-reset world from the
reference was 1.2e-6 m, against 4.2e-3 m reported in the issue at `njmax` 300. Two no-reset
references were not bitwise identical in this harness either (they differ at the 1e-7 m
level from steps 43–60 on), so the bitwise claim could not be tested.

## Hand study

The first hand runs used joint relaxation 0.8 and lost 28–56 % of their worlds to
divergence (next section). They are not quoted. Everything below was rerun once at 0.6 with
the same search budgets and seed.

The hand is 2 fingers and a thumb on a fixed palm (90 mm between finger bases, so every
object fits), flexion only, pinhole setback 6 mm, flexion stops 0.2 rad before the pinhole-crossing
angle. Actuators: explicit PD position servo per finger, force limit 15 N, no-load speed
limit 37.5 mm/s, slider gravity-compensated. Contact μ 0.5; objects 35–90 g. Trial: close
1.0 s, hold, retract the support, shake at 15 m/s² peak, 3 Hz for 1.0 s (grasp check), open
1.0 s, wait 0.6 s (release check: object dropped and every joint within 0.15 rad of rest).
SolverXPBD 64 substeps × 32 iterations, joint relaxation 0.6 (linear = angular),
collision every 4th substep. A world is invalid if anything is non-finite, any hand body
exceeds 10 m/s or 500 rad/s, tension exceeds 150 N, or a tendon ERROR is printed; invalid
worlds count as failures and stay in the denominator.

```bash
python experiments/hand_study.py search --n 384 --n4 256        # results/hand_search.npz, seed 0
python experiments/hand_study.py eval                            # results/hand_eval.npz
python experiments/hand_study.py eval --cfg '{"iterations":64}' --suffix _it2
python experiments/hand_study.py eval --cfg '{"substeps":128}' --suffix _dt2
PYTHONPATH=<newton tree with consistent rows> python experiments/hand_study.py eval --cfg '{"joint_relaxation":0.7,"joint_angular_relaxation":0.4}' --suffix _newtonA
python experiments/hand_study.py report [--suffix _it2|_dt2|_newtonA]   # figures and results/hand_metrics*.json
python experiments/hand_stats.py results/hand_eval.npz           # bootstrap intervals, results/hand_stats*.txt
```

Throughput on the RTX 4090 with graph capture and the per-substep recorder on: the search
(4,608 worlds, 64,512 bodies) takes 320 s. An evaluation (4,612 worlds) takes 307 s, 590 s
at 64 iterations, 600 s at dt/2 and 382 s with consistent rows. Invalid worlds: 0 of 4,608
in the search and 0 of 4,612 in each of the four evaluations. Tendon ERROR worlds: 0.

### Design search

Training objects are cyl40, sphere50 and box40; designs are random with seed 0.

| group | friction | worlds | cycle success (grasp and release) | designs with 3/3 |
|---|---|---|---|---|
| 1 actuator | μ = 0 | 1,152 | 0.745 | 197 of 384 |
| 1 actuator | μ = 0.3 | 1,152 | 0.825 | 248 of 384 |
| 2 actuators | μ = 0.3 | 768 | 0.812 | 156 of 256 |
| 3 actuators | μ = 0.3 | 768 | 0.699 | 89 of 256 |
| 9 (per joint) | μ = 0.3 | 768 | 0.326 | 18 of 256 |

Top 32 designs per population by training success (ties: lower peak actuator force, then
sample order). The μ = 0 and μ = 0.3 top-32 sets share 20 designs.

### Friction-aware against friction-blind designs

Held-out objects × held-out μ with 2 object offsets and 32 designs per population: 1,280
worlds each, 64 per heatmap cell. Intervals are 95 % bootstrap over designs.

| solver setting | blind | aware | aware − blind | blind: high μ − low μ | aware: high μ − low μ | invalid worlds |
|---|---|---|---|---|---|---|
| baseline: relaxation 0.6 / 0.6, 64 × 32 | 0.485 | 0.505 | +0.020 [-0.055, +0.098] | +0.061 [+0.023, +0.098] | +0.096 [+0.062, +0.129] | 0 |
| 2× iterations (64 × 64) | 0.521 | 0.523 | +0.002 [-0.095, +0.102] | +0.043 [-0.006, +0.092] | +0.025 [-0.012, +0.062] | 0 |
| dt / 2 (128 × 32) | 0.559 | 0.573 | +0.014 [-0.080, +0.111] | +0.045 [-0.010, +0.102] | +0.043 [-0.008, +0.094] | 0 |
| consistent rows (as in newton#4317), factors 0.7 / 0.4, 64 × 32 | 0.487 | 0.497 | +0.010 [-0.066, +0.090] | +0.047 [+0.006, +0.086] | +0.045 [+0.012, +0.078] | 0 |

"high μ − low μ" is success at μ ≥ 0.4 minus success at μ ≤ 0.15, paired by design.

Baseline success by μ = 0.05 / 0.15 / 0.25 / 0.4 / 0.6: blind 0.438 / 0.461 / 0.508 / 0.512 / 0.508;
aware 0.441 / 0.473 / 0.508 / 0.547 / 0.559. Grasp 0.506 (blind) and 0.530 (aware); release given grasp
0.958 and 0.953.

Baseline heatmap, friction-blind (columns μ = 0.05, 0.15, 0.25, 0.4, 0.6):

| object | 0.05 | 0.15 | 0.25 | 0.4 | 0.6 |
|---|---|---|---|---|---|
| cyl30 | 0.80 | 0.84 | 0.84 | 0.81 | 0.81 |
| sphere60 | 0.17 | 0.12 | 0.23 | 0.28 | 0.25 |
| box50 | 0.02 | 0.06 | 0.08 | 0.09 | 0.11 |
| cyl50 | 0.77 | 0.81 | 0.88 | 0.86 | 0.86 |

friction-aware:

| object | 0.05 | 0.15 | 0.25 | 0.4 | 0.6 |
|---|---|---|---|---|---|
| cyl30 | 0.89 | 0.92 | 0.94 | 0.94 | 0.94 |
| sphere60 | 0.16 | 0.19 | 0.22 | 0.25 | 0.34 |
| box50 | 0.06 | 0.05 | 0.08 | 0.09 | 0.17 |
| cyl50 | 0.66 | 0.73 | 0.80 | 0.91 | 0.78 |

- The hypothesis that friction-aware designs are more robust is not supported. Aware −
  blind is +0.020, +0.002, +0.014 and +0.010 in the four solver settings and every interval
  spans zero. The two top-32 sets also overlap in 20 designs.
- Success rises with pinhole friction in every setting (+0.03 to +0.10). The interval
  excludes zero at the baseline setting and with consistent rows, and includes zero at
  2× iterations and at dt/2. I treat "friction helps the grasp hold" as suggestive, not
  established.
- Both populations hold the cylinders (0.66–0.94) and mostly fail the 60 mm sphere and the
  50 mm box.

### Actuator count

Success on held-out objects with 32 designs per actuator count, 256 worlds per point.

Success at μ = 0.15 / 0.4:

| actuators | baseline | 2× iterations | dt / 2 | consistent rows |
|---|---|---|---|---|
| 1 | 0.457 / 0.531 | 0.512 / 0.520 | 0.547 / 0.570 | 0.492 / 0.500 |
| 2 | 0.449 / 0.500 | 0.508 / 0.574 | 0.539 / 0.605 | 0.449 / 0.488 |
| 3 | 0.480 / 0.578 | 0.559 / 0.605 | 0.574 / 0.621 | 0.496 / 0.566 |
| 9 (per joint) | 0.418 / 0.414 | 0.418 / 0.410 | 0.402 / 0.398 | 0.402 / 0.414 |

Success over both μ, and differences with 95 % bootstrap intervals over designs:

| solver setting | 1 | 2 | 3 | 9 (per joint) | 3 − 1 | 1 − per-joint | 3 − per-joint |
|---|---|---|---|---|---|---|---|
| baseline: relaxation 0.6 / 0.6, 64 × 32 | 0.494 | 0.475 | 0.529 | 0.416 | +0.035 [-0.053, +0.121] | +0.078 [+0.010, +0.146] | +0.113 [+0.027, +0.197] |
| 2× iterations (64 × 64) | 0.516 | 0.541 | 0.582 | 0.414 | +0.066 [-0.029, +0.162] | +0.102 [+0.018, +0.180] | +0.168 [+0.086, +0.252] |
| dt / 2 (128 × 32) | 0.559 | 0.572 | 0.598 | 0.400 | +0.039 [-0.053, +0.129] | +0.158 [+0.080, +0.234] | +0.197 [+0.125, +0.275] |
| consistent rows (as in newton#4317), factors 0.7 / 0.4, 64 × 32 | 0.496 | 0.469 | 0.531 | 0.408 | +0.035 [-0.045, +0.117] | +0.088 [+0.016, +0.156] | +0.123 [+0.041, +0.205] |

- 1, 2 and 3 actuators are not distinguishable: 3 − 1 is positive in all four settings and
  every interval spans zero.
- The tendon hands score above the per-joint baseline in all four settings, with intervals
  that exclude zero for both 1 − per-joint and 3 − per-joint. That baseline is nine
  torque-limited joint servos with open-loop targets found by the same random-search budget
  (256 designs in a 9-D command space; only 18 passed all training objects), so this says
  the underactuated hand is easier to command open-loop, not that nine actuators are worse
  in general. It has no tendon, so μ does not act on it.
- Absolute success of the tendon hands shifts by up to 0.074 between solver settings
  (friction-blind population: 0.485, 0.521, 0.559, 0.487). The ordering is stable; the levels
  are not converged to better than that.

### Do the comparisons keep their sign?

The fourth setting reruns the same selected designs on Newton with consistent positional
rows (the row change proposed in newton#4317: both parts scaled by the linear factor) at
Newton's default factors of 0.7 linear and 0.4 angular, instead of stock Newton with both
factors at 0.6. `*` marks intervals that exclude zero.

| comparison | baseline | 2× iterations | dt / 2 | consistent rows |
|---|---|---|---|---|
| friction-aware − friction-blind | +0.020 [-0.055, +0.098] | +0.002 [-0.095, +0.102] | +0.014 [-0.080, +0.111] | +0.010 [-0.066, +0.090] |
| 3 actuators − 1 actuator | +0.035 [-0.053, +0.121] | +0.066 [-0.029, +0.162] | +0.039 [-0.053, +0.129] | +0.035 [-0.045, +0.117] |
| 1 actuator − per-joint | +0.078 [+0.010, +0.146] * | +0.102 [+0.018, +0.180] * | +0.158 [+0.080, +0.234] * | +0.088 [+0.016, +0.156] * |
| 3 actuators − per-joint | +0.113 [+0.027, +0.197] * | +0.168 [+0.086, +0.252] * | +0.197 [+0.125, +0.275] * | +0.123 [+0.041, +0.205] * |

All four comparisons keep their sign in all four settings. Friction-aware vs blind and
3 vs 1 actuator never separate from zero; tendon vs per-joint always does.

The detached-tendon control ran in the same batches: 0 of 4 held-out objects held and 0
invalid worlds in each of the four settings.

## Why the first hand runs diverged

The hand batches first ran with joint linear and angular relaxation both at 0.8. XPBD
applies all rows of one kind from the same pose and sums their corrections, each scaled by
the relaxation factor. A phalanx in a grasp has several rows acting on the same rotation
(two joints, two tendon spans, a contact). At 0.8 the summed correction overshoots, the
error grows every iteration, and one link spins up until the world is non-finite. At 0.6,
still with equal factors, it does not happen. The single finger has no contacts and fewer
overlapping rows; it is stable at 0.8 and was left alone.

Setting both factors to 0.8 was my choice: equal factors are needed for correct dynamics
(newton#4314), and I had used the tendon examples' linear value. newton#4317 reports the
same effect from the other side: with consistent rows, a linear factor of 0.7 injects
energy into a floating humanoid, which is why that PR lowers the default to 0.5.

### Rate before and after each change

Same 1,568 worlds every time (96 tendon designs × 7 objects × μ ∈ {0, 0.3}, plus 32
per-joint designs × 7 objects), on the RTX 4090. A world is invalid if anything is
non-finite, a hand body exceeds 10 m/s or 500 rad/s, tension exceeds 150 N, or a tendon
ERROR is printed.

```bash
python experiments/hand_study.py rate --variant base      # likewise jointf, collide1, dt2, relax07, relax06, relax05, relax04
```

| change (one at a time) | invalid worlds | non-finite | invalid: tendon μ = 0 / tendon μ = 0.3 / per-joint | tendon ERROR worlds |
|---|---|---|---|---|
| as first run: joint relaxation 0.8 (linear = angular), body-force PD servo, collision every 4th substep | 60.78 % (953 / 1568) | 42.73 % | 64.4 % / 54.8 % / 67.9 % | 0 |
| servo delivered through `Control.joint_f` instead of body forces | 60.40 % (947 / 1568) | 43.75 % | 63.7 % / 55.1 % / 66.5 % | 0 |
| collision every substep | 61.35 % (962 / 1568) | 45.98 % | 65.2 % / 53.6 % / 73.2 % | 0 |
| dt / 2 (128 substeps) | 61.80 % (969 / 1568) | 45.54 % | 66.7 % / 54.5 % / 69.2 % | 0 |
| joint relaxation 0.7 | 0.13 % (2 / 1568) | 0.00 % | 0.3 % / 0.0 % / 0.0 % | 0 |
| joint relaxation 0.6 (the setting now used) | 0.00 % (0 / 1568) | 0.00 % | 0.0 % / 0.0 % / 0.0 % | 0 |
| joint relaxation 0.5 | 0.00 % (0 / 1568) | 0.00 % | 0.0 % / 0.0 % / 0.0 % | 0 |
| joint relaxation 0.4 | 0.00 % (0 / 1568) | 0.00 % | 0.0 % / 0.0 % / 0.0 % | 0 |

### What breaks first

A kernel after every substep records, per world, the first substep at which something goes
bad and what it was. It runs inside the captured graph with no host syncs. In the 953
invalid baseline worlds it was a finger link in 933 and a link together with tendon tension
in 20; never the slider, the contact data or the object. By phase: close 374, hold 31,
retract 118, second hold 11, shake 404, open 15.

### The actuator drive is not the cause

The sliders never used `target_ke`: since the first hand run they were driven by an
explicit PD force clamped to 15 N. Delivering the same force through `Control.joint_f`
gives identical slider travel, force and joint angles in a side-by-side test, and the same
divergence.

### Replays of diverged worlds

`experiments/triage_divergence.py` replays individual worlds on the CPU, 7 per object; logs
are in `results/triage_*.log`.

| change | sphere60 diverged | cyl30 | box50 |
|---|---|---|---|
| none (relaxation 0.8, collide every 4) | 6 of 7 | 2 of 7 | – |
| collision every substep | 7 of 7 | 2 of 7 | – |
| dt / 2 | 4 of 4, same frame | – | – |
| contact relaxation 0.8 → 0.4 | 7 of 7 | – | – |
| angular damping 20 s⁻¹ | 4 of 7 | – | – |
| joint relaxation 0.6 | 0 of 7 | 0 of 7 | 0 of 7 |
| joint relaxation 0.5 / 0.4 / 0.3 | 0 of 7 each | – | – |

Stepping one failing world at 0.8 substep by substep: the finger that breaks is in
mid-range (joint angles 0.91 / 0.39 / 0.61 rad, no stop engaged), tendon tension is 3 N and
the object moves at 0.1 m/s. Its links go from 12 to 1,756 rad/s in eight substeps and the
world is non-finite seven substeps later.

### What the lower factor costs

Single finger against the closed form at 64 × 32, loading branch
(`python -m dexdon.cycle --mu 0 0.3 --cfg '{"substeps":64,"iterations":32,"joint_relaxation":0.5}' --out results/cycle_relax05.npz`):

| joint relaxation | angle error max, μ = 0 [rad] | angle error max, μ = 0.3 [rad] | loop work μ = 0.3 [mJ] |
|---|---|---|---|
| 0.8 (single finger) | 0.0035 | 0.0034 | 27.04 |
| 0.6 (hand) | 0.0082 | 0.0059 | 27.13 |
| 0.5 | 0.0154 | 0.0090 | 27.25 |
| 0.4 | 0.0298 | 0.0161 | 27.47 |

So the hand uses 0.6, the highest value measured stable.

### Tendon ERROR logging

While checking that every tendon ERROR is counted I found my capture was broken:
kernel `printf` goes through C's buffered stdout and I restored the file descriptor
without `fflush`, so the capture always returned zero lines. Fixed in
`capture_tendon_errors`. `python experiments/check_error_capture.py` now passes on CPU
and on CUDA with graph replay (2 lines captured, mapped to worlds 2 and 1333). The earlier
"0 tendon ERRORs" entries rest on a different check: all 38 raw logs of those runs were
searched for `ERROR: Tendon` and contain none (uncaptured diagnostics land in the log).
The log of the dt/2 and 2×-iteration single-finger runs had been overwritten, so both
were rerun with the fixed capture (`results/cycle_dt2_recheck.npz`, `cycle_it2_recheck.npz`):
0 tendon ERRORs in each.

### Things I tried first that did not address the cause

All at relaxation 0.8: a flexion stop before the pinhole-crossing angle, a wider palm,
compliant stops, an actuator speed limit, longer tendon spans, and an eight-cell harness
factorial (30–73 % invalid in every cell, `results/hand_factorial.npz`). The wider palm,
the stop and the speed limit stayed because they are physically sensible; the compliant
stops and longer spans did not.

## Consistent positional rows: which factor?

newton#4317 makes both parts of a positional joint row use `joint_linear_relaxation`. To
see what the choice of factor does, I ran Newton three ways: unmodified `main` at 7ee5484;
A, with both parts on the linear factor (the row change of #4317, with the default factors
left at 0.7 / 0.4); and B, with both parts on the angular factor
(`upstream/angular_factor_variant.patch`, a diff against A). The trees were selected with
`PYTHONPATH`; raw outputs are in `results/xpbd_variants/`.

```bash
cd <tree> && python -m newton.tests -k xpbd -k tendon -k joint -k pendulum            # tests and examples
PYTHONPATH=<tree> python upstream/measure_xpbd_variants.py <label>                     # pendulum, joint convergence
PYTHONPATH=<tree>:. python -m dexdon.cycle --mu 0 0.3 --cfg '{"substeps":64,"iterations":32,"joint_relaxation":0.7,"joint_angular_relaxation":0.4}' --out results/cycle_newtondefaults_<label>.npz
PYTHONPATH=<tree> python experiments/hand_study.py rate --variant newton_defaults --suffix _<label>   # RTX 4090
```

| | `main` | A: both parts × `joint_linear_relaxation` (as in #4317) | B: both parts × `joint_angular_relaxation` |
|---|---|---|---|
| Newton tests, `-k xpbd -k tendon -k joint -k pendulum` (1,228 tests incl. 19 example tests of which 7 are skipped, macOS CPU) | not rerun | **OK**, 79 skipped | **2 failures**, 79 skipped |
| pendulum period / theory, default factors, dt = 1/600 s | 1.0901 | 1.0001 | 1.0002 |
| `basic_pendulum --solver xpbd`: worst anchor error [mm] (2 iterations) | 0.047 | 0.096 | 0.200 |
| `basic_joints --solver xpbd`: worst anchor error [mm] | 0.017 | 0.019 | 0.043 |
| `basic_urdf --solver xpbd` (48 joints): worst / final RMS anchor error [mm] | 6.03 / 0.250 | 5.87 / 0.250 | 9.04 / 0.462 |
| tendon finger vs closed-form statics, μ = 0.3: joint angle error max [rad] | 0.0734 | 0.0037 | 0.0051 |
| same finger: base tension RMS error | 16.0 % | 1.1 % | 1.1 % |
| 3-finger hand scene, 1,568 worlds, default factors: diverged worlds | 0 | 0 | 1 |
| same worlds: objects still held after the shake | 0.906 | 0.630 | 0.635 |


- Tests: A ran 1,228 tests, OK, 79 skipped (845 s). On a later `main` (a1903cd) 1,237 ran,
  OK, 84 skipped, and the pendulum script prints 1.0901 on that `main` and 1.0001 with A.
  B ran 1,228 with 2 failures, 79 skipped (816 s):
  `test_articulation_contact_drift_cpu` (root drifted 0.0499 m, limit 0.01) and
  `test_ball_joint_recovers_from_large_anchor_separation_cpu` (gap 0.5513 m, limit 0.5).
- Joint convergence is at each example's own defaults (2 iterations, 10 substeps, 300 frames).
- Hand scene: Newton default factors 0.7 / 0.4, otherwise the hand harness defaults; wall
  324 s (A), 286 s (B), 221 s (main). B's one invalid world broke at a finger link in the shake.
- A passes everything and B does not, which supports the factor choice in #4317.
- On unmodified `main` the same hand worlds hold 0.906 of objects against 0.630 with
  consistent rows, so the relaxation split changes task-level outcomes, not only a period.
