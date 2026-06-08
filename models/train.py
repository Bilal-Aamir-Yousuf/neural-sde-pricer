"""Phase 3 — Training the Neural SDE by Euler-Maruyama maximum likelihood.

Approach (option 1 in the spec — simplest to debug, implemented first)
----------------------------------------------------------------------
Discretise the learned SDE and treat each one-day step as approximately
Gaussian.  In log-space, for a step of size ``dt``:

    Delta Y = Y_{t+dt} - Y_t  ~  N( mu_theta(Y_t, t) * dt ,  sigma_phi(Y_t, t)^2 * dt )

We fit the drift and diffusion networks by **maximising the likelihood** of the
observed daily log-returns under this Gaussian transition density, i.e. by
minimising the negative log-likelihood

    NLL = 1/2 * [ (dY - mu*dt)^2 / (sigma^2 * dt) + log(2*pi * sigma^2 * dt) ]

averaged over every (state, increment) pair pooled across the basket.

Notes
-----
* Training is on the **physical measure P** (real historical returns).  The
  *diffusion* learned here is what matters for pricing (it is measure-invariant);
  the physical drift is replaced by the risk-neutral drift when pricing — see
  ``neural_sde.py``.
* The model is fit time-homogeneously (drift/diffusion depend on the log-price
  *level*), which is the regime the economic sanity-check in the spec refers to:
  in *price* space the diffusion sigma*S should increase with the price level.
* A temporal (never random) train/validation split lets us watch for overfitting.

Run:  python -m models.train
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CHECKPOINT_DIR, DT, DTYPE, OUTPUT_DIR, PRIMARY_TICKER, set_seed,
)
from data.fetch_data import build_regime_splits  # noqa: E402
from models.neural_sde import NeuralSDE  # noqa: E402

CHECKPOINT_PATH = CHECKPOINT_DIR / "neural_sde.pt"


# ---------------------------------------------------------------------------
# Build the (state, increment) training set
# ---------------------------------------------------------------------------
def build_increment_dataset(train_ds):
    """Pool daily (log-price Y_t, log-return dY) pairs across all basket names."""
    log_price = np.log(train_ds.prices)
    ys, dys = [], []
    for col in log_price.columns:
        series = log_price[col].dropna().to_numpy()
        if len(series) < 2:
            continue
        ys.append(series[:-1])
        dys.append(np.diff(series))
    Y = np.concatenate(ys)
    dY = np.concatenate(dys)
    # Drop non-finite (e.g. ticker IPO gaps).
    mask = np.isfinite(Y) & np.isfinite(dY)
    return Y[mask], dY[mask]


def gaussian_nll(sde: NeuralSDE, Y, dY, dt):
    """Mean negative log-likelihood of increments under the Euler step density."""
    t0 = torch.zeros((), dtype=Y.dtype, device=Y.device)
    mu = sde.drift_physical(t0, Y).squeeze(-1)        # physical drift
    sigma = sde.diffusion(t0, Y).squeeze(-1)          # diffusion > 0
    var = sigma ** 2 * dt
    nll = 0.5 * ((dY - mu * dt) ** 2 / var + torch.log(2 * np.pi * var))
    return nll.mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(epochs=400, lr=1e-3, hidden=64, n_layers=3, activation="tanh",
          val_frac=0.15, weight_decay=1e-5, seed=42, verbose=True):
    set_seed(seed)
    train_ds, _ = build_regime_splits()
    Y_np, dY_np = build_increment_dataset(train_ds)

    # Temporal split: last `val_frac` of the pooled series is validation.
    n = len(Y_np)
    cut = int(n * (1 - val_frac))
    Yt = torch.tensor(Y_np[:cut], dtype=DTYPE)
    dYt = torch.tensor(dY_np[:cut], dtype=DTYPE)
    Yv = torch.tensor(Y_np[cut:], dtype=DTYPE)
    dYv = torch.tensor(dY_np[cut:], dtype=DTYPE)

    sde = NeuralSDE(hidden=hidden, n_layers=n_layers, activation=activation)
    sde.set_normalization(Y_np.mean(), Y_np.std(), t_scale=1.0)

    opt = torch.optim.Adam(sde.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        sde.train()
        opt.zero_grad()
        loss = gaussian_nll(sde, Yt, dYt, DT)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sde.parameters(), 5.0)
        opt.step()
        sched.step()

        sde.eval()
        with torch.no_grad():
            vloss = gaussian_nll(sde, Yv, dYv, DT)
        history["train"].append(loss.item())
        history["val"].append(vloss.item())
        if vloss.item() < best_val:
            best_val = vloss.item()
            best_state = {k: v.clone() for k, v in sde.state_dict().items()}
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}  train NLL {loss.item():+.4f}  "
                  f"val NLL {vloss.item():+.4f}")

    sde.load_state_dict(best_state)  # restore best-val weights

    # --- estimate sensible pricing defaults from the data ---
    with torch.no_grad():
        # ATM-ish diffusion at the mean log-price = implied "v0".
        y_ref = torch.tensor([[Y_np.mean()]], dtype=DTYPE)
        sigma_ref = float(sde.diffusion(torch.zeros(()), y_ref).item())
    meta = {
        "hidden": hidden, "n_layers": n_layers, "activation": activation,
        "y_mean": float(Y_np.mean()), "y_std": float(Y_np.std()),
        "sigma_ref": sigma_ref, "best_val_nll": best_val,
    }
    torch.save({"state_dict": sde.state_dict(), "meta": meta}, CHECKPOINT_PATH)
    print(f"[train] saved checkpoint -> {CHECKPOINT_PATH.name} "
          f"(ref diffusion sigma~{sigma_ref:.3f}, best val NLL {best_val:.4f})")

    _plot_loss(history)
    _plot_drift_diffusion(sde, Y_np)
    return sde, history, meta


def load_trained(path=CHECKPOINT_PATH) -> NeuralSDE:
    """Reconstruct a trained NeuralSDE from a checkpoint."""
    ckpt = torch.load(path, weights_only=False)
    m = ckpt["meta"]
    sde = NeuralSDE(hidden=m["hidden"], n_layers=m["n_layers"], activation=m["activation"])
    sde.load_state_dict(ckpt["state_dict"])
    sde.eval()
    return sde


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------
def _plot_loss(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train"], label="train NLL")
    plt.plot(history["val"], label="val NLL")
    plt.xlabel("epoch"); plt.ylabel("negative log-likelihood")
    plt.title("Neural SDE training (Euler-Maruyama MLE)")
    plt.legend(); plt.tight_layout()
    out = OUTPUT_DIR / "training_loss.png"
    plt.savefig(out, dpi=130); plt.close()
    print(f"[train] wrote {out.name}")


def _plot_drift_diffusion(sde: NeuralSDE, Y_np):
    """Sanity-check the learned functions across the observed price range."""
    lo, hi = np.percentile(Y_np, 1), np.percentile(Y_np, 99)
    ys = torch.linspace(lo, hi, 200, dtype=DTYPE).unsqueeze(-1)
    with torch.no_grad():
        t0 = torch.zeros(())
        drift = sde.drift_physical(t0, ys).squeeze(-1).numpy()
        sigma = sde.diffusion(t0, ys).squeeze(-1).numpy()
    prices = np.exp(ys.squeeze(-1).numpy())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(prices, drift); ax[0].set_title("learned physical drift  mu(Y)")
    ax[0].set_xlabel("price S"); ax[0].set_ylabel("log-drift / yr")
    ax[1].plot(prices, sigma, color="C1")
    ax[1].set_title("learned log-vol  sigma(Y)")
    ax[1].set_xlabel("price S"); ax[1].set_ylabel("log diffusion / sqrt(yr)")
    # In PRICE space the diffusion is sigma*S — should increase with price level.
    ax[2].plot(prices, sigma * prices, color="C2")
    ax[2].set_title("price-space diffusion  sigma(Y)*S\n(should rise with S)")
    ax[2].set_xlabel("price S"); ax[2].set_ylabel("diffusion of S / sqrt(yr)")
    fig.tight_layout()
    out = OUTPUT_DIR / "learned_dynamics.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[train] wrote {out.name}")


if __name__ == "__main__":
    train()
