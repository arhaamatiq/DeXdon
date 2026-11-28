"""The tape gradient of the closed-form finger loss must match central finite differences."""

import numpy as np
import pytest
import warp as wp

from dexdon.sysid import FingerModel, finite_difference_grad

wp.config.quiet = True


def synthetic_cycle(mu, k, seed=0, n=120):
    """Angles from the model itself at (mu, k), plus 0.5 degrees of noise."""
    pull = np.concatenate([np.linspace(0.3, 9.0, n), np.linspace(9.0, 0.3, n)])
    opening = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    clean = FingerModel(pull, opening, np.zeros((2 * n, 3)), device="cpu").predict(mu, k)
    noisy = clean + np.random.default_rng(seed).normal(0.0, np.radians(0.5), clean.shape)
    return FingerModel(pull, opening, noisy, device="cpu")


@pytest.mark.parametrize("mu,k", [(0.05, 0.08), (0.2, 0.12), (0.35, 0.1), (0.5, 0.06), (0.3, 0.18)])
def test_tape_gradient_matches_finite_differences(mu, k):
    model = synthetic_cycle(0.25, 0.1)
    _, tape_grad = model.loss_and_grad(mu, k)
    fd_grad = finite_difference_grad(model, mu, k)
    rel = np.abs(tape_grad - fd_grad) / np.abs(fd_grad)
    assert np.all(rel < 0.01), f"tape {tape_grad} vs finite differences {fd_grad}, relative error {rel}"


def test_fit_recovers_parameters_from_model_data():
    model = synthetic_cycle(0.25, 0.1, seed=1)
    mu, k, _, _ = model.fit()
    assert abs(mu / 0.25 - 1.0) < 0.05
    assert abs(k / 0.1 - 1.0) < 0.05
