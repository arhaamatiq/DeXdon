"""Closed-form quasi-static model of a pinhole-routed, spring-return tendon finger.

Geometry (per joint i): one pinhole on the proximal body at (-a, -h_i) and one on
the distal body at (+a, -h_i), both measured from the joint axis in the straight
pose. With d_i = hypot(a, h_i) and phi_i = atan2(h_i, a):

    crossing-span length   l_i(theta) = 2 d_i cos(phi_i + theta/2)
    moment arm             r_i(theta) = -dl_i/dtheta = d_i sin(phi_i + theta/2)
    bend at each pinhole   beta_i     = theta_i / 2

Statics (joint i):   k_i (theta_i - theta0_i) = T_i r_i(theta_i)
Capstan (full slip): T_i = T_0 exp(-s mu w_i),  s = +1 loading, -1 unloading
    w_1 = th1/2,  w_2 = th1 + th2/2,  w_3 = th1 + th2 + th3/2
Kinematics:          x = sum_i [l_i(theta_i at rest) - l_i(theta_i)]  (inextensible)
"""

from __future__ import annotations

import numpy as np


def moment_arm(theta, a, h):
    return np.hypot(a, h) * np.sin(np.arctan2(h, a) + 0.5 * theta)


def span_length(theta, a, h):
    return 2.0 * np.hypot(a, h) * np.cos(np.arctan2(h, a) + 0.5 * theta)


def _bisect(f, lo, hi, n=80):
    flo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0.0) == (flo > 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_angles(T0, k, theta0, a, h, mu=0.0, s=1.0, theta_max=2.2):
    """Joint angles for base tension T0 on the full-slip branch s (+1 load, -1 unload).

    Joints are solved proximal to distal: joint i's tension depends only on the
    wrap accumulated before it plus half of its own bend.
    """
    n = len(k)
    th = np.zeros(n)
    T = np.zeros(n)
    wrap = 0.0  # bend accumulated at pinholes proximal to joint i's crossing span
    for i in range(n):

        def resid(x, i=i, wrap=wrap):
            Ti = T0 * np.exp(-s * mu * (wrap + 0.5 * abs(x)))
            return k[i] * (x - theta0[i]) - Ti * moment_arm(x, a, h[i])

        th[i] = _bisect(resid, theta0[i], theta_max)
        T[i] = T0 * np.exp(-s * mu * (wrap + 0.5 * abs(th[i])))
        wrap += abs(th[i])
    return th, T


def span_tensions(T0, th, mu=0.0, s=1.0):
    """Tension in every span, base to tip: lead, then (crossing, along-link) per joint.

    The last joint has no along-link span because its distal guide is the anchor.
    """
    out, T = [T0], T0
    for i, t in enumerate(th):
        T = T * np.exp(-s * mu * 0.5 * abs(t))  # proximal pinhole of joint i
        out.append(T)
        if i < len(th) - 1:
            T = T * np.exp(-s * mu * 0.5 * abs(t))  # distal pinhole of joint i
            out.append(T)
    return np.array(out)


def displacement(th, theta0, a, h, T0=0.0, mu=0.0, s=1.0, compliance=0.0):
    """Actuator travel = geometric shortening of the crossing spans + elastic stretch."""
    geo = float(sum(span_length(theta0[i], a, h[i]) - span_length(th[i], a, h[i]) for i in range(len(th))))
    return geo + compliance * float(np.sum(span_tensions(T0, th, mu, s)))


def solve_from_displacement(x, k, theta0, a, h, mu=0.0, s=1.0, compliance=0.0, T_hi=500.0):
    """Angles and tensions at actuator displacement x on full-slip branch s."""

    def resid(T0):
        th, _ = solve_angles(T0, k, theta0, a, h, mu, s)
        return displacement(th, theta0, a, h, T0, mu, s, compliance) - x

    T0 = _bisect(resid, 0.0, T_hi)
    th, T = solve_angles(T0, k, theta0, a, h, mu, s)
    return th, T, T0


def fingertip(th, L):
    """Planar fingertip position (x, z); flexion curls toward -z."""
    c = np.cumsum(th)
    return float(np.sum(np.asarray(L) * np.cos(c))), float(-np.sum(np.asarray(L) * np.sin(c)))
