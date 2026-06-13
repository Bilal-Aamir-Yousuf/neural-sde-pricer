"""Extension — variational training of the latent neural SDE (ELBO + logqp).

Objective (per window of observed log-prices x_{0:T})
-----------------------------------------------------
    loss = -E_q[ log p(x | z) ]  +  beta * KL
    KL   = KL( q(z0) || p(z0) )  +  E_q[ pathwise Girsanov KL ]   (torchsde logqp)

The pathwise term comes natively from ``torchsde.sdeint(..., logqp=True)``
(see ``latent_sde.py`` for the contract).  ``beta`` ramps linearly from 0 to 1
over the first ``KL_ANNEAL_EPOCHS`` epochs and then stays at 1 — annealing
prevents the classic collapse where the posterior is pinned to the prior
before the decoder has learned anything.

Collapse monitoring (this file watches for it; it does not pretend it away)
---------------------------------------------------------------------------
* Reconstruction and BOTH KL terms are logged separately every epoch.
* After annealing completes, a pathwise KL below ``KL_COLLAPSE_THRESHOLD``
  triggers a LOUD warning each epoch it persists (posterior == prior means the
  encoder is being ignored).
* The per-epoch line also reports corr(z1, realized vol) on the validation
  set: the Pearson correlation between the posterior's second latent
  coordinate and the trailing realized volatility of the observed series —
  the model is never shown realized vol, so a sustained correlation is direct
  evidence the free coordinate became a volatility state.  At the end of
  training the same correlation is also computed for the model's *instantaneous
  price-vol* g(z)[0] along the posterior path, which is invariant to the sign/
  scale of z1 (z1 itself could track vol with either sign).

Data
----
Reuses the existing Phase-1 pipeline (``build_regime_splits``): same basket,
same strictly-pre-2017 train window, so the latent model sees exactly the data
the v1 model saw.  Each ticker's log-price series is cut into overlapping
windows (length/stride in config) and made WINDOW-RELATIVE (x - x[0]) so the
dynamics are level-free.  Train/validation is a temporal split per ticker.

Run:  python -m models.train_latent            # full training
      python -m models.train_latent --smoke    # tiny subset, few epochs
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
    CHECKPOINT_DIR, DT, DTYPE, KL_ANNEAL_EPOCHS, KL_COLLAPSE_THRESHOLD,
    LATENT_WINDOW_LEN, LATENT_WINDOW_STRIDE, OUTPUT_DIR, REALIZED_VOL_WINDOW,
    TRADING_DAYS, set_seed,
)
from data.fetch_data import build_regime_splits  # noqa: E402
from models.latent_sde import LatentSDE  # noqa: E402

CHECKPOINT_PATH = CHECKPOINT_DIR / "latent_sde.pt"
RV_WINDOW = 10   # trailing window (days) for the realized-vol diagnostic


# ---------------------------------------------------------------------------
# Windowed sequence dataset (the latent model trains on PATHS, not increments)
# ---------------------------------------------------------------------------
def build_window_dataset(train_ds, window_len=LATENT_WINDOW_LEN,
                         stride=LATENT_WINDOW_STRIDE, val_frac=0.15):
    """Cut each ticker's log-prices into window-relative overlapping windows.

    Returns (train, val) arrays of shape (T, n_windows, 1).  The split is
    temporal PER TICKER: windows whose start falls in the last ``val_frac`` of
    that ticker's series are validation — no shuffled leakage.
    """
    log_price = np.log(train_ds.prices)
    train_wins, val_wins = [], []
    for col in log_price.columns:
        s = log_price[col].dropna().to_numpy()
        s = s[np.isfinite(s)]
        starts = range(0, len(s) - window_len + 1, stride)
        cut = int(len(s) * (1.0 - val_frac))
        for start in starts:
            w = s[start:start + window_len]
            if not np.isfinite(w).all():
                continue
            (val_wins if start >= cut else train_wins).append(w - w[0])
    if not train_wins or not val_wins:
        raise RuntimeError("not enough data to build train/val windows")
    to_arr = lambda ws: np.stack(ws, axis=1)[..., None]   # noqa: E731
    return to_arr(train_wins), to_arr(val_wins)


def realized_vol_matrix(X_np: np.ndarray, rv_window: int = RV_WINDOW) -> np.ndarray:
    """Trailing annualised realized vol per (day, window): (T, N), NaN warm-up.

    rv[t] uses the returns of days t-rv_window+1 .. t, so it is aligned with
    (and only with) information available AT day t — comparable to z1[t].
    """
    x = X_np[..., 0]                      # (T, N)
    dx = np.diff(x, axis=0)               # (T-1, N); dx[i] = return over (i, i+1)
    T, N = x.shape
    rv = np.full((T, N), np.nan)
    for t in range(rv_window, T):
        rv[t] = dx[t - rv_window:t].std(axis=0) * np.sqrt(TRADING_DAYS)
    return rv


# ---------------------------------------------------------------------------
# ELBO pieces + diagnostics
# ---------------------------------------------------------------------------
def beta_schedule(epoch: int, anneal_epochs: int) -> float:
    """Linear 0 -> 1 over the first ``anneal_epochs`` epochs, then 1."""
    if anneal_epochs <= 0:
        return 1.0
    return min(1.0, epoch / anneal_epochs)


def _pooled_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over all finite (a, b) pairs, pooled across windows."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def model_vol_along_path(model: LatentSDE, zs: torch.Tensor,
                         ts: torch.Tensor) -> np.ndarray:
    """Instantaneous price-coordinate vol g(t, z_t)[0] along a path: (T, N)."""
    with torch.no_grad():
        cols = [model.g(ts[i], zs[i])[:, 0] for i in range(zs.shape[0])]
    return torch.stack(cols, dim=0).numpy()


def val_pass(model: LatentSDE, Xva: torch.Tensor, ts: torch.Tensor,
             rv: np.ndarray):
    """Validation forward: ELBO terms at beta=1 plus the z1-usage diagnostic."""
    model.eval()
    with torch.no_grad():
        out = model(Xva, ts)
        recon = model.obs_log_prob(out.zs, Xva).mean()
        klp, klz = out.kl_path.mean(), out.kl_z0.mean()
    corr_z1 = _pooled_corr(out.zs[:, :, 1].numpy(), rv)
    return float(recon), float(klp), float(klz), corr_z1, out.zs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_latent(epochs=120, lr=1e-3, batch_size=32, latent_dim=2, hidden=64,
                 n_layers=2, ctx_dim=32, enc_hidden=64,
                 anneal_epochs=KL_ANNEAL_EPOCHS, window_len=LATENT_WINDOW_LEN,
                 stride=LATENT_WINDOW_STRIDE, val_frac=0.15, weight_decay=1e-5,
                 kl_collapse_threshold=KL_COLLAPSE_THRESHOLD,
                 max_windows=None, max_val_windows=256, seed=42, verbose=True):
    set_seed(seed)
    train_ds, _ = build_regime_splits()
    Xtr_np, Xva_np = build_window_dataset(train_ds, window_len, stride, val_frac)

    if max_windows is not None and Xtr_np.shape[1] > max_windows:
        keep = np.linspace(0, Xtr_np.shape[1] - 1, max_windows).astype(int)
        Xtr_np = Xtr_np[:, keep]
    if Xva_np.shape[1] > max_val_windows:
        keep = np.linspace(0, Xva_np.shape[1] - 1, max_val_windows).astype(int)
        Xva_np = Xva_np[:, keep]

    Xtr = torch.tensor(Xtr_np, dtype=DTYPE)
    Xva = torch.tensor(Xva_np, dtype=DTYPE)
    rv_va = realized_vol_matrix(Xva_np)
    T = Xtr.shape[0]
    ts = torch.linspace(0.0, (T - 1) * DT, T, dtype=DTYPE)
    print(f"[latent] windows: train {Xtr.shape[1]}, val {Xva.shape[1]} "
          f"(len {T} days, stride {stride})")

    model = LatentSDE(latent_dim=latent_dim, hidden=hidden, n_layers=n_layers,
                      ctx_dim=ctx_dim, enc_hidden=enc_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {"recon": [], "kl_path": [], "kl_z0": [], "beta": [],
               "train": [], "val": [], "corr_z1": []}
    best_val, best_state = float("inf"), None

    for ep in range(epochs):
        beta = beta_schedule(ep, anneal_epochs)
        model.train()
        perm = torch.randperm(Xtr.shape[1])
        ep_recon = ep_klp = ep_klz = ep_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            xs = Xtr[:, perm[i:i + batch_size], :]
            out = model(xs, ts)
            recon = model.obs_log_prob(out.zs, xs).mean()
            kl_path, kl_z0 = out.kl_path.mean(), out.kl_z0.mean()
            loss = -recon + beta * (kl_path + kl_z0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_recon += recon.item(); ep_klp += kl_path.item()
            ep_klz += kl_z0.item(); ep_loss += loss.item()
            n_batches += 1
        sched.step()
        ep_recon /= n_batches; ep_klp /= n_batches
        ep_klz /= n_batches; ep_loss /= n_batches

        # Validation at beta=1 always: the true (negative) ELBO, so checkpoint
        # selection is not distorted by the annealing schedule.
        v_recon, v_klp, v_klz, corr_z1, _ = val_pass(model, Xva, ts, rv_va)
        val_loss = -v_recon + (v_klp + v_klz)

        history["recon"].append(ep_recon); history["kl_path"].append(ep_klp)
        history["kl_z0"].append(ep_klz); history["beta"].append(beta)
        history["train"].append(ep_loss); history["val"].append(val_loss)
        history["corr_z1"].append(corr_z1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  ep {ep:3d}  beta {beta:4.2f} | recon {ep_recon:9.2f} | "
                  f"KL_path {ep_klp:8.3f} | KL_z0 {ep_klz:7.3f} | "
                  f"val(-ELBO) {val_loss:9.2f} | corr(z1,rv) {corr_z1:+.3f} | "
                  f"sig_obs {float(model.sigma_obs.detach()):.5f}")

        # ---- posterior collapse alarm (only meaningful once beta == 1) -----
        if ep >= anneal_epochs and ep_klp < kl_collapse_threshold:
            print("  " + "!" * 66)
            print(f"  !!! POSSIBLE POSTERIOR COLLAPSE: pathwise KL "
                  f"{ep_klp:.4f} < {kl_collapse_threshold} after annealing.")
            print("  !!! The posterior is ~identical to the prior — the encoder "
                  "is being ignored.")
            print("  " + "!" * 66)

    model.load_state_dict(best_state)

    # ---- final diagnostics on the best checkpoint ---------------------------
    f_recon, f_klp, f_klz, f_corr_z1, zs_va = val_pass(model, Xva, ts, rv_va)
    g0 = model_vol_along_path(model, zs_va, ts)
    f_corr_g0 = _pooled_corr(g0, rv_va)
    print("\n[latent] FINAL (best checkpoint, validation):")
    print(f"  reconstruction NLL / window : {-f_recon:10.2f}")
    print(f"  KL pathwise (logqp)         : {f_klp:10.3f}")
    print(f"  KL z0                       : {f_klz:10.3f}")
    print(f"  corr(z1, realized vol)      : {f_corr_z1:+.3f}")
    print(f"  corr(g(z)[0], realized vol) : {f_corr_g0:+.3f}   (sign/scale-invariant)")

    meta = {"latent_dim": latent_dim, "hidden": hidden, "n_layers": n_layers,
            "ctx_dim": ctx_dim, "enc_hidden": enc_hidden,
            "window_len": window_len, "best_val_neg_elbo": best_val,
            "sigma_obs": float(model.sigma_obs.detach()),
            "final_recon_nll": -f_recon, "final_kl_path": f_klp,
            "final_kl_z0": f_klz, "corr_z1_rv": f_corr_z1,
            "corr_g0_rv": f_corr_g0}
    torch.save({"state_dict": model.state_dict(), "meta": meta}, CHECKPOINT_PATH)
    print(f"[latent] saved checkpoint -> {CHECKPOINT_PATH.name} "
          f"(best val -ELBO {best_val:.2f})")

    _plot_training(history)
    _plot_latent_vol(model, Xva, ts)
    return model, history, meta


def load_trained_latent(path=CHECKPOINT_PATH) -> LatentSDE:
    """Reconstruct a trained LatentSDE from a checkpoint."""
    ckpt = torch.load(path, weights_only=False)
    m = ckpt["meta"]
    model = LatentSDE(latent_dim=m["latent_dim"], hidden=m["hidden"],
                      n_layers=m["n_layers"], ctx_dim=m["ctx_dim"],
                      enc_hidden=m["enc_hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def _plot_training(history):
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    ax[0].plot(history["train"], label="train loss")
    ax[0].plot(history["val"], label="val -ELBO (beta=1)")
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title("loss")
    ax[1].plot(history["recon"], color="C2")
    ax[1].set_xlabel("epoch"); ax[1].set_title("reconstruction log-lik / window")
    ax[2].plot(history["kl_path"], label="KL path (logqp)")
    ax[2].plot(history["kl_z0"], label="KL z0")
    ax2b = ax[2].twinx()
    ax2b.plot(history["beta"], color="gray", ls="--", alpha=0.6)
    ax2b.set_ylabel("beta")
    ax[2].set_xlabel("epoch"); ax[2].legend(loc="upper left")
    ax[2].set_title("KL terms (watch for collapse -> 0)")
    ax[3].plot(history["corr_z1"], color="C3")
    ax[3].axhline(0, color="gray", lw=0.5)
    ax[3].set_xlabel("epoch"); ax[3].set_ylim(-1, 1)
    ax[3].set_title("corr(z1, realized vol) on validation")
    fig.tight_layout()
    out = OUTPUT_DIR / "latent_training.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[latent] wrote {out.name}")


def _plot_latent_vol(model, Xva, ts, n_show=3, rv_window=RV_WINDOW):
    """The thesis plot: does posterior z1 track realized vol it was never shown?"""
    model.eval()
    with torch.no_grad():
        out = model(Xva[:, :n_show, :], ts)
    rv = realized_vol_matrix(Xva[:, :n_show, :].numpy(), rv_window)
    days = np.arange(Xva.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4), squeeze=False)
    for j in range(n_show):
        a = axes[0, j]
        a.plot(days, rv[:, j], color="C0", label=f"realized vol ({rv_window}d)")
        a2 = a.twinx()
        a2.plot(days, out.zs[:, j, 1].numpy(), color="C1", label="posterior z1")
        a.set_xlabel("day in window"); a.set_title(f"val window {j}")
        a.legend(loc="upper left"); a2.legend(loc="upper right")
    fig.suptitle("latent coordinate 1 vs realized vol (never shown to the model)")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "latent_vol_tracking.png"
    fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[latent] wrote {out_path.name}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset, few epochs — checks losses are finite/decreasing")
    args = ap.parse_args()
    if args.smoke:
        train_latent(epochs=6, batch_size=16, max_windows=48, max_val_windows=16,
                     anneal_epochs=3)
    else:
        train_latent()
