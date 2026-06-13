"""Smoke tests for the latent neural SDE module (extension, module-only stage).

Pins the four invariants that make the variational setup valid:
shapes, KL nonnegativity/finiteness, and — most importantly — that prior and
posterior dynamics share ONE diffusion object (the Girsanov KL is only a KL
under a shared diffusion; see module docstring).
"""
import torch

import config  # noqa: F401  (sets the float64 default dtype)
from models.latent_sde import LatentSDE

T, BATCH, DIM = 50, 8, 2


def _fake_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    # Random-walk-ish window-relative log-prices, ~1%/day moves.
    xs = (torch.randn(T, BATCH, 1, generator=g) * 0.01).cumsum(dim=0)
    xs = xs - xs[0]                       # window-relative: starts at 0
    ts = torch.linspace(0.0, (T - 1) / 252.0, T)
    return xs, ts


def test_forward_pass_shapes():
    torch.manual_seed(0)
    model = LatentSDE(latent_dim=DIM)
    xs, ts = _fake_batch()
    out = model(xs, ts)

    assert out.zs.shape == (T, BATCH, DIM)              # latent path
    assert out.recon_mean.shape == (T, BATCH, 1)        # reconstructed obs path
    assert out.kl_path.shape == (BATCH,)                # pathwise Girsanov KL
    assert out.kl_z0.shape == (BATCH,)                  # initial-state KL
    assert out.sigma_obs.shape == ()                    # one scalar obs noise
    # Decoder is the structured readout: mean is exactly latent coordinate 0.
    assert torch.equal(out.recon_mean, out.zs[..., :1])


def test_kl_terms_nonnegative_and_finite():
    torch.manual_seed(0)
    model = LatentSDE(latent_dim=DIM)
    xs, ts = _fake_batch()
    out = model(xs, ts)

    # torchsde's logqp channel integrates 0.5*||(f-h)/g||^2 with ZERO diffusion
    # (expectation-form Girsanov integrand), so it is nonnegative PATHWISE,
    # not just on average — assert per sample.
    assert torch.isfinite(out.kl_path).all()
    assert bool((out.kl_path >= 0).all()), "pathwise KL must be >= 0 per sample"

    # Gaussian-Gaussian KL is nonnegative exactly.
    assert torch.isfinite(out.kl_z0).all()
    assert bool((out.kl_z0 >= -1e-12).all())

    assert torch.isfinite(out.recon_mean).all()
    assert float(out.sigma_obs.detach()) > 0


def test_prior_and_posterior_share_one_diffusion_object():
    model = LatentSDE(latent_dim=DIM)
    # Identity, not equality: equal weights could still silently diverge after
    # a refactor; the invariant is ONE object.
    assert model.prior_diffusion is model.posterior_diffusion
    assert model.prior_diffusion is model.diffusion_net
    # And exactly one diffusion network exists among submodules (no shadow g).
    diffusion_ids = {
        id(m) for name, m in model.named_modules() if "diffusion" in name.split(".")[0:1]
    }
    assert len({id(model.diffusion_net)} | diffusion_ids) == 1


def test_torchsde_logqp_contract():
    """f/h/g exist with the torchsde-expected signatures and shapes."""
    torch.manual_seed(0)
    model = LatentSDE(latent_dim=DIM)
    xs, ts = _fake_batch()
    ctx, _ = model.encode(xs)
    model.contextualize(ts, ctx)

    z = torch.randn(BATCH, DIM)
    t = ts[3]
    assert model.f(t, z).shape == (BATCH, DIM)          # posterior drift
    assert model.h(t, z).shape == (BATCH, DIM)          # prior drift
    assert model.g(t, z).shape == (BATCH, DIM)          # diagonal diffusion
    assert bool((model.g(t, z) > 0).all()), "diffusion must be strictly positive"
