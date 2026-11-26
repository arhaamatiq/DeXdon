# Derivations and worked examples

Notes on the mechanics and the solver behind the results, each with a small worked
example. The numbers in the examples come from:

```bash
python -c "from dexdon import analytic as A; print(A.solve_angles(4.0,(0.1,)*3,(0,)*3,0.006,(0.008,)*3,mu=0.3,s=1.0))"
```

---

## Finger statics: where does a tendon-driven finger stop?

One tendon runs along a 3-joint finger. Each joint has a return spring.
I pull the tendon with tension `T`. What angle does each joint settle at?

At rest, torques about each joint cancel. The tendon tries to
close the joint with torque `T × (moment arm)`. The spring tries to open it with
torque `k × (angle − rest angle)`. The joint stops where the two are equal.

The moment arm is the perpendicular distance from the joint axis to the straight
piece of tendon that crosses the joint. With pinhole routing that distance changes
as the joint bends, so it is a function `r(θ)`, not a constant.

The geometry: each joint has two pinholes, both a depth `h` below the joint line:
one `a` behind the axis (on the proximal link), one `a` in front (on the distal
link). Both sit at distance `d = √(a² + h²)` from the axis. Let `φ = atan2(h, a)`.

- Angle between the two pinholes, seen from the axis: `π − 2φ − θ`.
- Length of the crossing span (a chord): `l(θ) = 2 d cos(φ + θ/2)`.
- Moment arm (distance from the axis to that chord): `r(θ) = d sin(φ + θ/2)`.

Check: at `θ = 0`, `r = d sin φ = h`. And `r = −dl/dθ`, which is the principle of
virtual work: torque = tension × (how fast the tendon path shortens with angle).

In pseudocode:

```
for each joint i:
    find θ such that   k_i (θ − θ0_i)  =  T_i · d_i · sin(φ_i + θ/2)
```

The code (`dexdon/analytic.py`):

```python
def moment_arm(theta, a, h):
    return np.hypot(a, h) * np.sin(np.arctan2(h, a) + 0.5 * theta)   # r(θ) = d sin(φ + θ/2)

def resid(x):                                                        # zero at equilibrium
    return k[i] * (x - theta0[i]) - Ti * moment_arm(x, a, h[i])      # spring torque − tendon torque

th[i] = _bisect(resid, theta0[i], theta_max)                         # root by bisection
```

Line by line: `np.hypot(a, h)` is `d`. `np.arctan2(h, a)` is `φ`. `resid` is
negative while the tendon wins and positive once the spring wins, so its root is
the resting angle. Bisection halves the bracket 80 times; it cannot diverge.

A worked example with `a = 6 mm`, `h = 8 mm`, `k = 0.1 N·m/rad`, `θ0 = 0`, `T = 4 N`.

- `d = √(36 + 64) = 10 mm`, `φ = atan2(8, 6) = 0.9273 rad`.
- Constant-arm estimate: `θ = h T / k = 0.008 × 4 / 0.1 = 0.320 rad`.
- True arm, fixed-point iteration `θ ← T d sin(φ + θ/2) / k`:
  - `θ = 0.320 → sin(1.0873) = 0.8855 → 0.3542`
  - `θ = 0.3542 → sin(1.1044) = 0.8933 → 0.3573`
  - `θ = 0.3573 → sin(1.1060) = 0.8940 → 0.3576` (converged)
- Answer: **0.358 rad** per joint. The constant-arm formula is 11 % low here,
  because the arm has grown from 8.0 mm to 8.94 mm.

Assumptions: a massless inextensible-ish tendon (compliance is added
separately), no gravity in E1/E2, frictionless revolute joints, springs linear in
angle, pinholes are points.

---

## Capstan friction: why tension drops along the route

The tendon bends as it passes through each pinhole. The pinhole rubs
on it. How much tension is lost, and in which direction?

A rope wrapped around a post by angle `β` can hold a load ratio
of at most `e^(μβ)` before it slips (the capstan equation). A pinhole is a tiny
post. While the tendon is sliding through it, the ratio is exactly `e^(μβ)`, and
friction always opposes the sliding. So:

- **Closing (loading):** the actuator pulls, tendon slides toward the actuator,
  tension *falls* going outward: `T_out = T_in e^(−μβ)`.
- **Opening (unloading):** the springs pull the tendon back, sliding reverses,
  tension *rises* going outward: `T_out = T_in e^(+μβ)`.
- **In between:** the ratio is inside the band `[e^(−μβ), e^(+μβ)]` and the tendon
  sticks. That band is the source of hysteresis.

Bend angles in this finger: the span along a link is parallel to the link. By
symmetry the crossing span makes `θ_i/2` with each link. So each of joint `i`'s two
pinholes bends the tendon by `β = θ_i/2`; one joint wraps the tendon by `θ_i` total.

Tension in the span crossing joint `i` (`s = +1` loading, `−1` unloading):

```
T_1 = T_0 e^(−s μ θ1/2)
T_2 = T_0 e^(−s μ (θ1 + θ2/2))
T_3 = T_0 e^(−s μ (θ1 + θ2 + θ3/2))
```

Only the crossing span's tension matters for joint `i`. Cut the tendon in
the span that crosses joint `i` and take everything distal as one free body. The
only tendon force entering it is `T_i` along that span. All pinhole reactions
farther out are internal to that body. So torque about joint `i` is `T_i r_i(θ_i)`.

In pseudocode:

```
wrap = 0
for joint i from base to tip:
    solve  k_i (θ − θ0_i) = T0 · exp(−s μ (wrap + θ/2)) · r_i(θ)   for θ
    wrap += θ
```

The code (`solve_angles`):

```python
wrap = 0.0                                        # bend already paid for, proximal to joint i
for i in range(n):
    def resid(x, i=i, wrap=wrap):
        Ti = T0 * np.exp(-s * mu * (wrap + 0.5 * abs(x)))   # capstan up to this crossing span
        return k[i] * (x - theta0[i]) - Ti * moment_arm(x, a, h[i])
    th[i] = _bisect(resid, theta0[i], theta_max)
    wrap += abs(th[i])                            # both pinholes of joint i: θ/2 + θ/2
```

The joints decouple base-to-tip: joint `i`'s tension depends only on angles already
solved plus its own. That is why a 1-D root-find per joint is enough.

A worked example with the same finger, `μ = 0.3`, loading, `T0 = 4 N`.

| joint | wrap before | equation solved | θ [rad] | T crossing [N] |
|---|---|---|---|---|
| 1 | 0 | `0.1 θ = 4 e^(−0.15 θ) · 0.01 sin(0.9273 + θ/2)` | 0.338 | 3.80 |
| 2 | 0.338 | `0.1 θ = 4 e^(−0.3(0.338 + θ/2)) · …` | 0.304 | 3.45 |
| 3 | 0.642 | `0.1 θ = 4 e^(−0.3(0.642 + θ/2)) · …` | 0.277 | 3.16 |

Frictionless gives 0.358 rad at every joint. With friction the tip gets 21 % less
tension than the base, so distal joints lag while closing. On the way back the
exponent flips sign and the distal joints *lead*. Same actuator position, two poses.

Why total flexion barely shows hysteresis: the tendon's
length is fixed, so actuator travel must equal the shortening of the crossing spans:

```
x = Σ_i [ l_i(θ0_i) − l_i(θ_i) ]  +  compliance · Σ_spans T
```

With equal moment arms this pins `Σθ_i` to `x` no matter what friction does.
Friction can only *redistribute* flexion between joints, and change the force.
That is exactly what E2 measured.

---

## XPBD basics: how the simulator enforces "this length stays fixed"

A constraint `C(x) = 0` (a joint stays together, a tendon span keeps
its length) must hold after every time step, and stiff springs must not blow up.

XPBD moves bodies freely for one step, then nudges them back
onto the constraints. Each constraint keeps a running total `λ` of how hard it has
pushed this step. A *compliance* `α` (inverse stiffness) lets the constraint give
a little: the nudge stops when `C = −α̃ λ`, i.e. when stretch matches force.

In pseudocode, one substep:

```
v ← v + dt · f_ext / m ;  x ← x + dt · v            # predict
λ ← 0 for every constraint
repeat `iterations` times:
    for every constraint:
        Δλ = −(C(x) + α̃ λ) / (w + α̃)                # α̃ = α / dt², w = Σ 1/m along the constraint
        λ ← λ + Δλ ;  x ← x + (1/m) ∇C · Δλ
v ← (x − x_prev) / dt
```

The code (Newton, `solvers/xpbd/tendon_kernels.py`, tendon stretch; `λ` is stored as an impulse):

```python
alpha = compliance * compliance_lambda_scale          # compliance / dt
d_lambda = -(err + alpha * lambda_prev + gamma * derr) # stretch + compliance·λ (+ damping)
linear_denom = (dt + gamma) * denom + compliance / dt  # dt·w + compliance/dt
d_lambda = d_lambda / linear_denom * relaxation
lambda_new = wp.min(lambda_prev + d_lambda, 0.0)       # a cable can pull, never push
```

`err` is `length − rest_length`. `denom` is the generalized inverse mass `w`.
`wp.min(…, 0)` makes the constraint one-sided: tension only.

A worked example with one span, `compliance = 1e-5 m/N`, `dt = 1/3840 s = 2.6e-4 s`,
body with `w = 800 kg⁻¹` (a 5 g phalanx with a lever arm), stretched by `err = 4e-5 m`.

- `alpha = 1e-5 / 2.6e-4 = 0.0384`; `linear_denom = 2.6e-4 × 800 + 0.0384 = 0.2468`.
- First iteration, `λ = 0`: `Δλ = −4e-5 / 0.2468 = −1.62e-4 N·s` → force `λ/dt = 0.62 N`.
- The physical answer is `err / compliance = 4 N`. One iteration delivers 16 % of it,
  because `dt·w` (0.208) is larger than `compliance/dt` (0.038): the bodies are light
  and the constraint is stiff. More iterations (or smaller `dt`) close the gap.
- This is why the validated setting is 64 substeps × 32 iterations, and why
  `tension = stretch / compliance` only equals the applied force once converged.

Four things about the solver that I measured (commands in [results.md](results.md)):

1. *Joint drives are not accumulated across iterations* (newton#4315). `SolverXPBD`
   solves `target_ke` drives with `λ = 0` every iteration, so the drive's effective
   stiffness grows with the iteration count (0.41×, 0.87×, 1.59×, 3.18×, 5.0× the requested
   value at 1, 2, 4, 8, 16 iterations). The return springs and the actuator servo here are
   explicit forces instead.
2. *Unequal joint relaxation factors bias dynamics* (newton#4314). The defaults scale the
   linear part of a positional correction by 0.7 and its angular part by 0.4. A hinged rod
   then swings with a 9 % longer period than theory, at any iteration count. With equal
   factors the period matches to 4 digits, so both are set to the same value here.
3. *Explicit damping has a stability limit.* A damper torque `c ω` applied
   explicitly needs `c · dt / I < 1`. With `I = 2.4e-7 kg·m²` and `dt = 1e-3 s`,
   `c = 1e-3` gives 4.2 and diverges; `c = 1e-4` at `dt = 2.6e-4 s` gives 0.11.

4. *Relaxation is a stability knob, not only a speed knob.* XPBD here updates all rows of
   one kind from the same pose and adds their corrections (Jacobi style), each scaled by
   the relaxation factor. If `n` rows act on the same motion of a body, the combined step
   is about `n × relaxation` times the needed correction. Above 2 the error grows every
   iteration instead of shrinking.

   For example, a middle phalanx in a grasp carries two joints, two tendon spans and a
   contact, so up to `n = 5` rows can push on its rotation. At relaxation 0.8 the worst
   case is `5 × 0.8 = 4.0` (diverges). At 0.6 it is 3.0 on paper, but the rows only
   partly overlap, and measured on 1,568 hand worlds the invalid rate was 60.8 % at 0.8
   and 0 % at 0.6, 0.5 and 0.4. The single finger without contacts has fewer overlapping
   rows and is stable at 0.8.

   The price of a lower factor is slower convergence: single-finger closed-form error at
   64 × 32 is 0.0035 rad at 0.8, 0.0082 at 0.6, 0.0154 at 0.5 and 0.0298 at 0.4. So the
   hand runs at 0.6: the highest stable value I measured.

---

## Grasp basics

*Friction cone.* A contact can push along its normal with force `f_n` and resist
sliding with at most `μ f_n` sideways. The set of forces it can apply is a cone of
half-angle `atan(μ)` around the normal. With contact `μ = 0.5` that is 26.6°.

*Force closure.* A grasp is force-closed if the contact cones together can
produce any net force and torque on the object, so any disturbance can be resisted
by squeezing harder. Two opposing fingers plus a thumb can achieve this on a
cylinder when their contact normals are spread around it.

*Form closure, or caging.* The fingers geometrically block every escape path, with
no reliance on friction. A power grasp that wraps the object approaches this.

*Why underactuation helps.* One tendon drives three joints. When a proximal link
hits the object it stops, and the remaining tendon travel flows to the distal
joints, which keep closing until they touch too. The finger adapts its shape to the
object with no sensing. The price: the pose is not controlled joint by joint, and
tendon friction decides how force is shared along the finger.

*How the harness tests a grasp.* Close, remove the support (lift), shake with
1.5 g at 3 Hz, check the object is still held; then open and check the object is
released and every joint is back within 0.15 rad of its rest angle.

---

## Why the runs are set up the way they are

| Choice | Reason |
|---|---|
| XPBD only | The handoff notes known CUDA tendon failures in VBD; one solver keeps the study consistent. |
| Slider + position target, no winch | A zero-friction roller slips freely, and driven rollers need pre-tension. A slider anchors the tendon end, so actuation is identical at every `μ`. |
| Pinholes, not rollers, at joints | Fixed roller wraps must stay in `[0, π]`. A pinhole has no wrap limit and its bend angle is exactly `θ/2`, which makes the closed form possible. |
| Friction only at passive pinholes | Isolates the variable under study. The actuator coupling never changes. |
| Explicit pre-tension via `rest_length` | Auto rest lengths (`−1`) measure the initial geometry and add zero pre-tension. |
| Read tension after `step`, outside capture | The arrays are solver diagnostics; `.numpy()` forces a device sync and cannot live inside a captured graph. |
| Any tendon `ERROR` invalidates the trial | After the rest-length-floor error the reported rest lengths and tensions are invalid by the docs. Kernel `printf` goes through C's buffered stdout, so the capture must `fflush` before restoring the descriptor; without it the count silently reads zero. |
| In-graph first-failure recorder | A kernel after every substep stores, per world, the first substep at which a slider, tendon tension, contact, link or object goes non-finite or impossibly fast. No host sync, so it works inside a captured CUDA graph and says what broke first. |
| Invalid worlds stay in the denominator | A diverged world is counted as a failure and its rate is reported. Dropping it would bias every success rate. |
| Halve `dt`, double iterations, more sweeps, tighter tolerances, other relaxations | If the loop moved with these knobs it would be a numerical artifact. Physics must be invariant to the discretization. |
| `μ = 0` control | Without friction there is no dissipative mechanism, so the loop must close. Its residual width is the numerical floor. |
| Detached-tendon control | Proves the motion comes from the tendon and not from the springs, the servo or solver drift. |
| Slow ramps, never steps | A step overshoots; the tendon slides back and forth and ends at an arbitrary point inside the stick band. Only a monotone quasi-static ramp stays on the loading branch. |
| Fixed seeds, same budgets | Every design population sees the same samples, objects and limits, so differences come from the scoring friction alone. |
