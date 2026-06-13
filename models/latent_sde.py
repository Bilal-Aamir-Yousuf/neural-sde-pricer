"""Extension — Latent Neural SDE (variational, torchsde logqp convention).

The v1 model (``neural_sde.py``) uses the log-price as the *entire* state, so
volatility can only ever be a function of the current price level.  This module
lifts the dynamics to a hidden state Z (default 2-D: coordinate 0 plays
log-price, coordinate 1 is free to become a volatility state), observed through
a noisy readout.

Pieces (and the torchsde contract)
----------------------------------
torchsde's ``sdeint(..., logqp=True)`` expects, per its source
(``_core/base_sde.py::RenameMethodsSDE`` defaults / ``SDELogqp``):

    f(t, y) -> posterior drift      (this is what is *integrated*)
    h(t, y) -> prior drift          (used only inside the KL channel)
    g(t, y) -> diffusion            (SHARED by construction — see below)

``SDELogqp`` augments the state with one channel whose drift is
``0.5 * || (f - h) / g ||^2`` and whose diffusion is **zero** — i.e. the
*expectation-form* Girsanov integrand.  Consequences we rely on:

  * the returned log-ratio increments are pathwise **nonnegative**;
  * the KL is correct **only if prior and posterior share one diffusion**.
    Two SDEs with different diffusions have mutually singular path laws
    (quadratic variation identifies g a.s.) — the true KL would be +inf and
    the integral above would silently stop being a KL.  Structurally there is
    exactly ONE diffusion network on this module; ``prior_diffusion`` and
    ``posterior_diffusion`` are aliases of the same object so tests can
    identity-check the invariant.

Observation model (explicit)
----------------------------
For a window of (window-relative) log-prices x_{0:T}:

    x_t | z_t  ~  Normal( z_t[0] ,  sigma_obs^2 )

i.e. the decoder reads coordinate 0 of the latent state as the log-price mean,
with a single learned observation std ``sigma_obs = softplus(raw) + floor``
(floor ~1bp in log-price units, so the likelihood cannot diverge).  This
structured decoder is deliberate: it keeps risk-neutral pricing and
"start paths at the observed spot" well-defined (no decoder inversion needed).

Posterior context
-----------------
A GRU consumes the observed window in REVERSE time, so its output at index i
is a function of x_{i:T} — the future from t_i's perspective (the smoothing
posterior, not the lagging filter).  ``f`` looks the context up with
``searchsorted(ts, t, right=False)``: at a grid time t_i the context includes
x_{t_i}; strictly between grid points only genuinely-future observations.
q(z0) is a diagonal Gaussian read from the reverse-GRU state at t=0 (which
summarizes the whole window); the prior p(z0) is a learned global diagonal
Gaussian — the unconditional initial law used by generative sampling/pricing.

Pricing-measure logic (Q drift, vol shift) intentionally lives with the
pricing module, not here — this file is the probabilistic model only.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.neural_sde import MLP  # noqa: E402  (reused small MLP)

LATENT_DIM = 2


@dataclass
class LatentSDEOutput:
    """One variational forward pass over a batch of observed windows."""
    zs: torch.Tensor          # (T, batch, d)   posterior latent path
    recon_mean: torch.Tensor  # (T, batch, 1)   decoder mean for x_t (= zs[..., :1])
    sigma_obs: torch.Tensor   # ()              observation std
    kl_z0: torch.Tensor       # (batch,)        KL(q(z0) || p(z0)), closed form
    kl_path: torch.Tensor     # (batch,)        pathwise Girsanov KL (logqp sum)
    qz0_mean: torch.Tensor    # (batch, d)      posterior z0 mean (diagnostics)
    qz0_logstd: torch.Tensor  # (batch, d)


class LatentSDE(nn.Module):
    """Latent neural SDE with shared diffusion, exposing torchsde's f/h/g."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = 64,
                 n_layers: int = 2, ctx_dim: int = 32, enc_hidden: int = 64,
                 sigma_floor: float = 1e-3, obs_floor: float = 1e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.ctx_dim = ctx_dim
        self._sigma_floor = sigma_floor

        # --- dynamics: two drifts, ONE diffusion ---------------------------
        self.h_net = MLP(latent_dim + 1, latent_dim, hidden, n_layers)            # prior drift
        self.f_net = MLP(latent_dim + 1 + ctx_dim, latent_dim, hidden, n_layers)  # posterior drift
        self.diffusion_net = MLP(latent_dim + 1, latent_dim, hidden, n_layers)    # the shared g

        # --- posterior machinery -------------------------------------------
        # Encoder input per step: [x_t, forward increment (0-padded at the end)].
        self.encoder_gru = nn.GRU(input_size=2, hidden_size=enc_hidden)
        self.ctx_proj = nn.Linear(enc_hidden, ctx_dim)
        self.qz0_net = nn.Linear(enc_hidden, 2 * latent_dim)

        # --- learned prior over the initial state p(z0) ---------------------
        self.pz0_mean = nn.Parameter(torch.zeros(latent_dim))
        self.pz0_logstd = nn.Parameter(torch.zeros(latent_dim))

        # --- observation model:  x_t | z_t ~ N(z_t[0], sigma_obs^2) ---------
        self._obs_floor = obs_floor
        # Init sigma_obs ~ 1% (NOT at the floor): starting deep in softplus's
        # saturated tail makes d(sigma)/d(raw) ~ sigma, and the resulting tiny
        # update direction is then flattened further by global grad clipping —
        # sigma_obs would barely move.  Start loose, let training tighten it.
        self.raw_obs_noise = nn.Parameter(torch.tensor(-4.61))  # sigma_obs ~ 0.01

        self._ctx: tuple[torch.Tensor, torch.Tensor] | None = None
        self._init_dynamics_scales()

    def _init_dynamics_scales(self) -> None:
        """Start the dynamics at market scale instead of softplus-of-random.

        Untouched, softplus(MLP(z)) initializes the diffusion near ~0.7/sqrt(yr)
        — several times real equity log-vol — so prior paths wander O(1) in
        log-price and the reconstruction term starts ~1e7, drowning every other
        gradient.  We bias the diffusion head so g ~ 0.15/sqrt(yr) at init, and
        start both drifts near zero (standard neural-SDE practice: learn
        departures from 'no drift', don't start from random drift).
        """
        with torch.no_grad():
            for net in (self.h_net, self.f_net):
                net.net[-1].weight.mul_(0.1)
                net.net[-1].bias.zero_()
            self.diffusion_net.net[-1].weight.mul_(0.1)
            # softplus(b) + floor ~= 0.15  =>  b = ln(e^{0.15 - floor} - 1)
            target = 0.15 - self._sigma_floor
            self.diffusion_net.net[-1].bias.fill_(
                float(torch.log(torch.expm1(torch.tensor(target))))
            )
            # q(z0): start tight and centred at 0.  Windows are window-relative
            # (x0 = 0 exactly), so a random O(+-1) initial mean parks the whole
            # near-zero-drift path an O(1) log-price away from the data — that
            # offset, not the dynamics, then dominates the reconstruction term.
            # Output layout is [mean | logstd]: mean -> 0, logstd -> -3 (~0.05).
            self.qz0_net.weight.mul_(0.1)
            self.qz0_net.bias.zero_()
            self.qz0_net.bias[self.latent_dim:].fill_(-3.0)

    # ------------------------------------------------------------------ guards
    @property
    def prior_diffusion(self) -> nn.Module:
        """The diffusion used under the prior — same object as the posterior's."""
        return self.diffusion_net

    @property
    def posterior_diffusion(self) -> nn.Module:
        """The diffusion used under the posterior — same object as the prior's."""
        return self.diffusion_net

    @property
    def sigma_obs(self) -> torch.Tensor:
        return F.softplus(self.raw_obs_noise) + self._obs_floor

    # ---------------------------------------------------------------- features
    @staticmethod
    def _with_time(t, z):
        t = torch.as_tensor(t, dtype=z.dtype, device=z.device)
        return torch.cat([z, t.expand(z.shape[0], 1)], dim=1)

    # ------------------------------------------------- torchsde f / h / g API
    def f(self, t, z):
        """POSTERIOR drift: conditions on z and the encoder context at time t."""
        if self._ctx is None:
            raise RuntimeError(
                "Posterior drift needs encoder context — call forward() / "
                "contextualize() first (or use h() for the prior)."
            )
        ts, ctx = self._ctx
        t_val = torch.as_tensor(t, dtype=ts.dtype, device=ts.device)
        # right=False: at a grid time t_i use ctx[i] (includes x_{t_i});
        # strictly between grid points use the next index (future obs only).
        i = int(torch.searchsorted(ts, t_val.reshape(1), right=False).item())
        i = min(i, ctx.shape[0] - 1)
        return self.f_net(torch.cat([self._with_time(t, z), ctx[i]], dim=1))

    def h(self, t, z):
        """PRIOR drift (torchsde's logqp KL compares f against this)."""
        return self.h_net(self._with_time(t, z))

    def g(self, t, z):
        """The single shared diffusion, strictly positive (softplus + floor)."""
        return F.softplus(self.diffusion_net(self._with_time(t, z))) + self._sigma_floor

    # ----------------------------------------------------------------- encoder
    def encode(self, xs: torch.Tensor):
        """Backwards GRU over the observed window.

        xs: (T, batch, 1) window-relative log-prices.
        Returns (ctx_path, summary):
          ctx_path (T, batch, ctx_dim) — ctx_path[i] is a function of x_{i:T}
          summary  (batch, enc_hidden) — function of the WHOLE window (for q(z0))
        """
        dx = xs[1:] - xs[:-1]
        dx = torch.cat([dx, torch.zeros_like(dx[:1])], dim=0)   # pad last step
        feats = torch.cat([xs, dx], dim=-1)                     # (T, batch, 2)
        out_rev, _ = self.encoder_gru(feats.flip(0))            # consume x_T .. x_0
        out = out_rev.flip(0)                                   # out[i] <- x_{i:T}
        return torch.tanh(self.ctx_proj(out)), out[0]

    def contextualize(self, ts: torch.Tensor, ctx: torch.Tensor) -> None:
        """Stash the (ts, context-path) pair that f() interpolates over."""
        self._ctx = (ts, ctx)

    # ----------------------------------------------------------------- decoder
    def decode_mean(self, zs: torch.Tensor) -> torch.Tensor:
        """E[x_t | z_t] = z_t[0]  (structured readout: coord 0 IS log-price)."""
        return zs[..., :1]

    def obs_log_prob(self, zs: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """log N(x_t | z_t[0], sigma_obs^2), summed over time: (batch,)."""
        dist = torch.distributions.Normal(self.decode_mean(zs), self.sigma_obs)
        return dist.log_prob(xs).sum(dim=0).squeeze(-1)

    # ------------------------------------------------------------------ z0 law
    def _qz0(self, summary: torch.Tensor):
        mean, logstd = self.qz0_net(summary).chunk(2, dim=-1)
        return mean, logstd

    @staticmethod
    def _kl_diag_gaussians(mq, log_sq, mp, log_sp):
        """KL( N(mq, e^{2 log_sq}) || N(mp, e^{2 log_sp}) ), summed over dims."""
        var_q, var_p = torch.exp(2 * log_sq), torch.exp(2 * log_sp)
        kl = log_sp - log_sq + (var_q + (mq - mp) ** 2) / (2 * var_p) - 0.5
        return kl.sum(dim=-1)

    # ----------------------------------------------------------------- forward
    def forward(self, xs: torch.Tensor, ts: torch.Tensor,
                dt: float | None = None) -> LatentSDEOutput:
        """Variational pass: encode -> sample z0 -> sdeint(logqp) -> decode.

        xs: (T, batch, 1) observed window-relative log-prices, ts: (T,) times.
        """
        ctx, summary = self.encode(xs)
        qz0_mean, qz0_logstd = self._qz0(summary)
        z0 = qz0_mean + torch.randn_like(qz0_mean) * torch.exp(qz0_logstd)

        self.contextualize(ts, ctx)
        if dt is None:
            dt = float(ts[1] - ts[0])
        zs, log_ratio = torchsde.sdeint(self, z0, ts, method="euler", dt=dt,
                                        logqp=True)
        # zs: (T, batch, d); log_ratio increments: (T-1, batch), each >= 0.

        kl_z0 = self._kl_diag_gaussians(qz0_mean, qz0_logstd,
                                        self.pz0_mean, self.pz0_logstd)
        return LatentSDEOutput(
            zs=zs,
            recon_mean=self.decode_mean(zs),
            sigma_obs=self.sigma_obs,
            kl_z0=kl_z0,
            kl_path=log_ratio.sum(dim=0),
            qz0_mean=qz0_mean,
            qz0_logstd=qz0_logstd,
        )


if __name__ == "__main__":
    import config  # noqa: F401  (sets float64 default)

    torch.manual_seed(0)
    model = LatentSDE()
    T, B = 50, 8
    xs = torch.randn(T, B, 1).cumsum(0) * 0.01
    ts = torch.linspace(0.0, (T - 1) / 252.0, T)
    out = model(xs, ts)
    print("zs", tuple(out.zs.shape), "recon", tuple(out.recon_mean.shape))
    print("kl_z0", out.kl_z0.detach().numpy().round(3))
    print("kl_path", out.kl_path.detach().numpy().round(5))
    print("sigma_obs", float(out.sigma_obs))
