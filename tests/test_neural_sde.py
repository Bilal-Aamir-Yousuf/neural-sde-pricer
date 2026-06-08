"""Tests for the Neural SDE architecture (Phase 2 correctness)."""
import torch

from config import RISK_FREE_RATE
from models.neural_sde import GeometricBrownianSDE, NeuralSDE


def test_shapes_and_diffusion_positivity():
    sde = NeuralSDE()
    y = torch.randn(32, 1) + 5.0
    t = torch.tensor(0.1)
    assert sde.f(t, y).shape == (32, 1)
    assert sde.g(t, y).shape == (32, 1)
    assert bool((sde.g(t, y) > 0).all()), "diffusion must be strictly positive"


def test_diffusion_positive_extreme_inputs():
    sde = NeuralSDE()
    y = torch.tensor([[-50.0], [0.0], [50.0]])     # extreme log-prices
    assert bool((sde.diffusion(torch.zeros(()), y) > 0).all())


def test_risk_neutral_drift_formula():
    """Under measure Q the log-drift must equal r - 1/2 sigma^2."""
    sde = NeuralSDE().configure(measure="Q", rate=0.05)
    y = torch.randn(16, 1) + 5.0
    t = torch.zeros(())
    sigma = sde.g(t, y)
    expected = 0.05 - 0.5 * sigma ** 2
    assert torch.allclose(sde.f(t, y), expected, atol=1e-10)


def test_physical_measure_uses_learned_drift():
    sde = NeuralSDE().configure(measure="P")
    y = torch.randn(8, 1) + 5.0
    t = torch.zeros(())
    assert torch.allclose(sde.f(t, y), sde.drift_physical(t, y))


def test_gbm_matches_black_scholes_drift():
    sde = GeometricBrownianSDE(sigma=0.2).configure(measure="Q", rate=0.03)
    y = torch.zeros(4, 1)
    assert torch.allclose(sde.g(y.new_zeros(()), y), torch.full((4, 1), 0.2))
    assert torch.allclose(sde.f(y.new_zeros(()), y),
                          torch.full((4, 1), 0.03 - 0.5 * 0.2 ** 2))


def test_vol_shift_scales_diffusion():
    sde = NeuralSDE()
    y = torch.randn(8, 1) + 5.0
    t = torch.zeros(())
    base = sde.g(t, y)
    sde.vol_shift = 0.5
    assert torch.allclose(sde.g(t, y), base * 1.5)
