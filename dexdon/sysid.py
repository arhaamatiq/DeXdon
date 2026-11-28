"""System identification of the finger: recover pinhole friction mu and joint stiffness k
from joint angles measured against tendon pull over one close-open cycle.

The model is the closed-form finger of docs/derivations.md written as a Warp kernel, so
``wp.Tape`` gives d(loss)/d(mu, k). Newton's routed tendons are not differentiable (the
tendon docs say so, and ``check_xpbd_tendon_gradient`` shows the gradient comes back zero),
so the gradient fit runs on this model, not on the simulator.

Per joint i the crossing-span tension T_i satisfies k * theta_i = T_i * r_i(theta_i).

  closing:  T_i = T0 * exp(-mu * w_i)      w_i = bend accumulated up to joint i's span
  opening:  the pinholes stick first, then reverse one by one from the actuator outward,
            so  theta_i = min(theta_i at the turnaround, reverse-slip solution at T0)
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .finger import FingerParams

NEWTON_ITERS = 10
THETA_CAP = 2.5  # [rad] keeps Newton bounded where the reverse-slip equation has no root


@wp.func
def joint_angle(
    tension: wp.float64, k: wp.float64, mu_s: wp.float64, wrap: wp.float64, d: wp.float64, phi: wp.float64
) -> wp.float64:
    """Solve k * th = tension * exp(-mu_s * (wrap + th/2)) * d * sin(phi + th/2) for th by Newton."""
    half = wp.float64(0.5)
    cap = wp.float64(THETA_CAP)
    th = wp.min(tension * d * wp.sin(phi) / k, cap)
    for _ in range(NEWTON_ITERS):
        e = wp.exp(-mu_s * (wrap + half * th))
        f = k * th - tension * e * d * wp.sin(phi + half * th)
        df = k - tension * e * d * half * (wp.cos(phi + half * th) - mu_s * wp.sin(phi + half * th))
        step = wp.float64(0.0)
        if df > wp.float64(1.0e-9):
            step = f / df
        else:
            step = th - cap  # no root below the cap: the joint is held by stiction
        th = wp.clamp(th - step, wp.float64(0.0), cap)
    return th


@wp.kernel
def finger_model_kernel(
    pull: wp.array[wp.float64],
    opening: wp.array[int],
    pull_peak: wp.float64,
    params: wp.array[wp.float64],
    d: wp.array[wp.float64],
    phi: wp.array[wp.float64],
    theta_meas: wp.array2d[wp.float64],
    weight: wp.float64,
    theta_out: wp.array2d[wp.float64],
    loss: wp.array[wp.float64],
):
    """Model joint angles for every sample and accumulate the mean squared angle error."""
    j = wp.tid()
    mu = params[0]
    k = params[1]
    zero = wp.float64(0.0)

    # pose at the turnaround (end of closing)
    p1 = joint_angle(pull_peak, k, mu, zero, d[0], phi[0])
    p2 = joint_angle(pull_peak, k, mu, p1, d[1], phi[1])
    p3 = joint_angle(pull_peak, k, mu, p1 + p2, d[2], phi[2])

    t = pull[j]
    th1 = zero
    th2 = zero
    th3 = zero
    if opening[j] == 0:
        th1 = joint_angle(t, k, mu, zero, d[0], phi[0])
        th2 = joint_angle(t, k, mu, th1, d[1], phi[1])
        th3 = joint_angle(t, k, mu, th1 + th2, d[2], phi[2])
    else:
        th1 = wp.min(p1, joint_angle(t, k, -mu, zero, d[0], phi[0]))
        th2 = wp.min(p2, joint_angle(t, k, -mu, th1, d[1], phi[1]))
        th3 = wp.min(p3, joint_angle(t, k, -mu, th1 + th2, d[2], phi[2]))

    theta_out[j, 0] = th1
    theta_out[j, 1] = th2
    theta_out[j, 2] = th3
    r1 = th1 - theta_meas[j, 0]
    r2 = th2 - theta_meas[j, 1]
    r3 = th3 - theta_meas[j, 2]
    wp.atomic_add(loss, 0, weight * (r1 * r1 + r2 * r2 + r3 * r3))


class FingerModel:
    """Closed-form finger evaluated on one measured cycle (float64, any Warp device)."""

    def __init__(self, pull, opening, theta_meas, p: FingerParams | None = None, device=None):
        p = p or FingerParams()
        f64 = wp.float64
        self.device = wp.get_device(device)
        self.n = len(pull)
        self.pull = wp.array(np.asarray(pull, np.float64), dtype=f64, device=self.device)
        self.opening = wp.array(np.asarray(opening, np.int32), dtype=int, device=self.device)
        self.pull_peak = float(np.max(pull))
        h = np.asarray(p.h, np.float64)
        self.d = wp.array(np.hypot(p.a, h), dtype=f64, device=self.device)
        self.phi = wp.array(np.arctan2(h, p.a), dtype=f64, device=self.device)
        self.theta_meas = wp.array(np.asarray(theta_meas, np.float64), dtype=f64, device=self.device)
        self.theta_out = wp.zeros((self.n, 3), dtype=f64, device=self.device)
        self.weight = 1.0 / (3.0 * self.n)

    def loss_and_grad(self, mu, k, grad=True):
        """Mean squared angle error [rad^2] and its gradient with respect to (mu, k)."""
        params = wp.array(np.array([mu, k], np.float64), dtype=wp.float64, device=self.device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float64, device=self.device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(
                finger_model_kernel, dim=self.n, device=self.device,
                inputs=[self.pull, self.opening, wp.float64(self.pull_peak), params, self.d, self.phi, self.theta_meas, wp.float64(self.weight)],
                outputs=[self.theta_out, loss],
            )  # fmt: skip
        value = float(loss.numpy()[0])
        if not grad:
            return value, None
        tape.backward(loss=loss)
        g = params.grad.numpy().copy()
        tape.zero()
        return value, g

    def predict(self, mu, k):
        self.loss_and_grad(mu, k, grad=False)
        return self.theta_out.numpy().copy()

    def fit(self, mu0=0.2, k0=0.12, iters=400, lr=0.03):
        """Adam on (log mu, log k) with tape gradients. Returns (mu, k, loss, loss history)."""
        x = np.log(np.array([mu0, k0]))
        m, v = np.zeros(2), np.zeros(2)
        hist = []
        best = (np.inf, x.copy())
        for it in range(1, iters + 1):
            p = np.exp(x)
            value, g = self.loss_and_grad(p[0], p[1])
            hist.append(value)
            if value < best[0]:
                best = (value, x.copy())
            gx = g * p  # chain rule to log parameters
            m = 0.9 * m + 0.1 * gx
            v = 0.999 * v + 0.001 * gx * gx
            step = lr * (m / (1 - 0.9**it)) / (np.sqrt(v / (1 - 0.999**it)) + 1e-12)
            if it > iters * 0.6:
                step *= 0.3  # smaller steps to settle
            x = x - step
        mu, k = np.exp(best[1])
        return float(mu), float(k), float(best[0]), hist


def finite_difference_grad(model: FingerModel, mu, k, rel=1.0e-5):
    """Central finite differences of the loss with respect to (mu, k)."""
    g = np.zeros(2)
    for i, (val, name) in enumerate(((mu, "mu"), (k, "k"))):
        hstep = rel * val
        hi = [mu, k]
        lo = [mu, k]
        hi[i] += hstep
        lo[i] -= hstep
        g[i] = (model.loss_and_grad(*hi, grad=False)[0] - model.loss_and_grad(*lo, grad=False)[0]) / (2.0 * hstep)
        _ = name
    return g


@wp.kernel
def _tip_height_kernel(body_q: wp.array[wp.transform], body: int, loss: wp.array[float]):
    loss[0] = wp.transform_get_translation(body_q[body])[2]


def check_xpbd_tendon_gradient(steps=40, device=None):
    """Ask ``wp.Tape`` for d(fingertip height)/d(pinhole mu) through SolverXPBD with routed tendons.

    Returns a dict describing what came back. A zero or missing gradient next to a non-zero
    finite-difference sensitivity means the simulator path gives no usable gradient.
    """
    import newton

    from .finger import DriveRig, SolverCfg, add_finger

    def rollout(mu, with_tape):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
        rig = DriveRig()
        hnd = add_finger(builder, rig, FingerParams(mu=mu))
        model = builder.finalize(device=device, requires_grad=with_tape)
        rig.finalize(model.device)
        rig.ac_target.fill_(0.008)
        cfg = SolverCfg(substeps=64, iterations=32)
        solver = cfg.make_solver(model)
        states = [model.state() for _ in range(steps + 1)]
        control = model.control()
        loss = wp.zeros(1, dtype=float, device=model.device, requires_grad=with_tape)
        tape = wp.Tape() if with_tape else None

        def run():
            for i in range(steps):
                states[i].clear_forces()
                rig.apply(states[i])
                solver.step(states[i], states[i + 1], control, None, cfg.dt)
            wp.launch(_tip_height_kernel, dim=1, inputs=[states[-1].body_q, hnd.links[-1]], outputs=[loss], device=model.device)

        if with_tape:
            with tape:
                run()
        else:
            run()
        return float(loss.numpy()[0]), model, tape, loss

    out = {}
    try:
        value, model, tape, loss = rollout(0.3, True)
        tape.backward(loss=loss)
        grad = model.tendon_guide_mu.grad
        out["tape_gradient"] = None if grad is None else [float(g) for g in grad.numpy()]
        out["loss"] = value
    except Exception as e:  # report whatever the solver says
        out["tape_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    hi = rollout(0.33, False)[0]
    lo = rollout(0.27, False)[0]
    out["finite_difference"] = (hi - lo) / 0.06
    return out


@wp.kernel
def _advance_target_kernel(rate: wp.array[float], target: wp.array[float]):
    i = wp.tid()
    target[i] = target[i] + rate[0]


class FingerBatch:
    """Many single fingers, one per world, each with its own (mu, k). Gravity off, no contacts."""

    def __init__(self, mus, ks, cfg=None, device=None):
        import newton

        from .finger import DriveRig, SolverCfg, add_finger

        self.cfg = cfg or SolverCfg(substeps=64, iterations=32)
        self.n = len(mus)
        main = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
        rigs = []
        for mu, k in zip(mus, ks, strict=True):
            wb = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
            rig = DriveRig()
            add_finger(wb, rig, FingerParams(mu=float(mu), k=(float(k),) * 3))
            main.add_world(wb)
            rigs.append(rig)
        bodies = wb.body_count
        self.model = main.finalize(device=device)
        self.rig = DriveRig()
        for w, r in enumerate(rigs):
            for store, src in ((self.rig.sp, r.sp), (self.rig.ac, r.ac)):
                for key, vals in src.items():
                    if key in ("child", "parent"):
                        vals = [v + w * bodies if v >= 0 else -1 for v in vals]
                    store[key] += vals
        self.rig.finalize(self.model.device)
        self.solver = self.cfg.make_solver(self.model)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.rate = wp.zeros(1, dtype=float, device=self.model.device)
        self.errors: list[str] = []
        self.graph = None

    def _frame(self):
        for _ in range(self.cfg.substeps):
            wp.launch(_advance_target_kernel, dim=self.n, inputs=[self.rate], outputs=[self.rig.ac_target], device=self.model.device)
            self.s0.clear_forces()
            self.rig.apply(self.s0)
            self.solver.step(self.s0, self.s1, self.control, None, self.cfg.dt)
            self.s0, self.s1 = self.s1, self.s0

    def cycle(self, x_max=0.020, t_ramp=4.0):
        """Ramp every actuator 0 -> x_max -> 0 at constant speed, the target moving every substep.

        Returns per-frame pull (servo force) [N], joint angles [rad], slider travel [m] and the opening flag.
        """
        from .finger import capture_tendon_errors

        frames = round(t_ramp * self.cfg.fps)
        per_substep = x_max / (frames * self.cfg.substeps)
        pull = np.zeros((2 * frames, self.n))
        theta = np.zeros((2 * frames, self.n, 3))
        travel = np.zeros((2 * frames, self.n))
        opening = np.arange(2 * frames) >= frames
        self.rig.ac_target.zero_()
        with capture_tendon_errors(self.errors):
            for f in range(2 * frames):
                self.rate.fill_(per_substep if f < frames else -per_substep)
                if self.model.device.is_cuda and self.graph is None and f == 1:
                    with wp.ScopedCapture(device=self.model.device) as cap:
                        self._frame()
                    self.graph = cap.graph
                elif self.graph is not None:
                    wp.capture_launch(self.graph)
                else:
                    self._frame()
                pull[f] = self.rig.ac_force.numpy()
                travel[f] = self.rig.ac_x.numpy()
                theta[f] = self.rig.sp_theta.numpy().reshape(self.n, 3)
        return dict(pull=pull, theta=theta, travel=travel, opening=opening)


def _interp_branch(pull_w, theta_w, pull_q):
    order = np.argsort(pull_w)
    return np.stack([np.interp(pull_q, pull_w[order], theta_w[order, i]) for i in range(3)], 1)


def trajectory_at(pull_w, theta_w, opening, pull_q):
    """A world's joint angles read off at the measured pull values, branch by branch."""
    out = np.zeros((len(pull_q), 3))
    for branch in (False, True):
        m = opening == branch
        out[m] = _interp_branch(pull_w[m], theta_w[m], pull_q[m])
    return out


def brute_force(lib_pull, lib_theta, opening, mus, ks, shape, pull_meas, theta_meas):
    """Score every library world against the measurement and return the best (mu, k).

    ``shape`` is the (n_mu, n_k) layout of the grid. The estimate is the grid minimum refined
    by the minimum of a quadratic fitted to the 3 x 3 block of losses around it.
    """
    n = len(mus)
    loss = np.empty(n)
    for w in range(n):
        r = trajectory_at(lib_pull[:, w], lib_theta[:, w], opening, pull_meas) - theta_meas
        loss[w] = float((r * r).mean())
    best = int(np.argmin(loss))
    grid = loss.reshape(shape)
    i, j = np.unravel_index(best, shape)
    mu_g, k_g = np.asarray(mus).reshape(shape), np.asarray(ks).reshape(shape)

    # Least-squares quadratic over the 3 x 3 block around the grid minimum, in grid-index units.
    mu_hat, k_hat = float(mu_g[i, j]), float(k_g[i, j])
    if 0 < i < shape[0] - 1 and 0 < j < shape[1] - 1:
        u, v = [a.ravel().astype(float) for a in np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij")]
        basis = np.stack([np.ones(9), u, v, u * u, u * v, v * v], 1)
        c = np.linalg.lstsq(basis, grid[i - 1 : i + 2, j - 1 : j + 2].ravel(), rcond=None)[0]
        hess = np.array([[2.0 * c[3], c[4]], [c[4], 2.0 * c[5]]])
        if np.all(np.linalg.eigvalsh(hess) > 0.0):
            du, dv = np.clip(np.linalg.solve(hess, -c[1:3]), -1.0, 1.0)
            mu_hat += du * float(mu_g[i + 1, j] - mu_g[i, j])
            k_hat += dv * float(k_g[i, j + 1] - k_g[i, j])
    return dict(mu=mu_hat, k=k_hat, mu_grid=float(mu_g[i, j]), k_grid=float(k_g[i, j]), world=best, loss=float(loss[best]))
