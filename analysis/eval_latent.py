"""Extension — head-to-head evaluation: latent SDE vs the v1 MLE neural SDE.

Three comparisons, each with its metric definition stated up front:

1.  OUT-OF-SAMPLE NLL — on the same held-out segment (the last 15% of each
    ticker's 2010-2016 series, the temporal split the models were validated
    on).  Definitions:
      * v1:     EXACT conditional Gaussian transition NLL per daily increment
                (its training objective, computed on absolute log-prices).
      * latent: ELBO and IWAE-K LOWER BOUNDS on the JOINT path likelihood,
                divided by the number of increments.  The pathwise importance
                weight uses the full Girsanov Radon-Nikodym derivative
                (including the martingale term  int u . dW  — torchsde's
                logqp only provides the expectation-form integrand, which is
                NOT the pathwise log-density ratio).
    These are NOT the same quantity: the latent number is (a) a lower bound,
    and (b) a joint density that also pays for observation noise and the
    initial state.  Both (a) and (b) push the latent number DOWN, so:
    latent > v1 would be decisive; v1 > latent is suggestive but partly
    attributable to the definitional handicap.  Both readings are printed.

2.  PATH REALISM — simulate from both models' PHYSICAL/prior dynamics and
    compare to real out-of-sample SPY (2017-2022) on volatility clustering
    (ACF of |r| and r^2 at several lags) and fat tails (excess kurtosis,
    P(|r| > 3 sigma), QQ plot).  Both models use a frozen time feature
    (v1: t=0, its training value; latent: mid-window T_REF) — "days since
    window start" has no economic meaning, so the dynamics are treated as
    time-homogeneous rather than extrapolating the t-input.

3.  REGIME RMSE — the existing Phase-7 benchmark regenerated with a "Latent
    SDE" column.  The latent model gets exactly the same single degree of
    freedom as v1 (one ATM vol-level shift); Heston keeps its 5-parameter
    least-squares fit to the scoring target, so Heston's number remains
    structurally flattered (see regime_comparison.py).

Run:  python -m analysis.eval_latent
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.stats import kurtosis

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    DT, DTYPE, OUTPUT_DIR, PRIMARY_TICKER, TRADING_DAYS, set_seed,
)
from data.fetch_data import build_regime_splits, download_prices  # noqa: E402
from models.train import gaussian_nll, load_trained  # noqa: E402
from models.train_latent import build_window_dataset, load_trained_latent  # noqa: E402
from pricing.latent_pricing import (  # noqa: E402
    T_REF, calibrate_latent_vol_shift, price_european_latent,
)

RESULTS_MD = OUTPUT_DIR / "eval_latent_results.md"


# ===========================================================================
# 1) Out-of-sample NLL
# ===========================================================================
def v1_oos_nll(sde, train_ds, val_frac=0.15) -> float:
    """Exact transition NLL per increment on each ticker's last 15% segment."""
    log_price = np.log(train_ds.prices)
    Ys, dYs = [], []
    for col in log_price.columns:
        s = log_price[col].dropna().to_numpy()
        s = s[np.isfinite(s)]
        seg = s[int(len(s) * (1 - val_frac)):]
        Ys.append(seg[:-1]); dYs.append(np.diff(seg))
    Y = torch.tensor(np.concatenate(Ys), dtype=DTYPE)
    dY = torch.tensor(np.concatenate(dYs), dtype=DTYPE)
    with torch.no_grad():
        return float(gaussian_nll(sde, Y, dY, DT))


def latent_oos_bounds(model, Xva_np, K=64, chunk=32, seed=0):
    """ELBO and IWAE-K bounds on the per-window joint log-likelihood.

    Importance weight per posterior sample:
        log w = log p(x|z) + log p(z0) - log q(z0) - log dQpost/dPprior(path)
    with the PATHWISE Girsanov term
        log dQ/dP = sum u . dW  +  1/2 sum ||u||^2 dt ,   u = (f - h) / g,
    accumulated along the same Euler discretisation used to simulate z.
    """
    torch.manual_seed(seed)
    Xva = torch.tensor(Xva_np, dtype=DTYPE)
    T, N, _ = Xva.shape
    ts = torch.linspace(0.0, (T - 1) * DT, T, dtype=DTYPE)
    sq = math.sqrt(DT)
    Normal = torch.distributions.Normal
    elbos, iwaes = [], []
    with torch.no_grad():
        for s0 in range(0, N, chunk):
            xs = Xva[:, s0:s0 + chunk]
            n = xs.shape[1]
            ctx, summary = model.encode(xs)
            qm, qls = model._qz0(summary)
            ctxK = ctx.repeat_interleave(K, dim=1)
            xsK = xs.repeat_interleave(K, dim=1)
            qmK = qm.repeat_interleave(K, dim=0)
            qlsK = qls.repeat_interleave(K, dim=0)
            B = n * K
            model.contextualize(ts, ctxK)

            z = qmK + torch.exp(qlsK) * torch.randn(B, model.latent_dim, dtype=DTYPE)
            logq0 = Normal(qmK, qlsK.exp()).log_prob(z).sum(-1)
            logp0 = Normal(model.pz0_mean, model.pz0_logstd.exp()).log_prob(z).sum(-1)
            sig = model.sigma_obs
            recon = Normal(z[:, 0], sig).log_prob(xsK[0, :, 0])
            logRN = torch.zeros(B, dtype=DTYPE)
            for i in range(T - 1):
                t = ts[i]
                f, h, g = model.f(t, z), model.h(t, z), model.g(t, z)
                u = (f - h) / g
                eps = torch.randn(B, model.latent_dim, dtype=DTYPE)
                logRN += (u * eps).sum(-1) * sq + 0.5 * (u * u).sum(-1) * DT
                z = z + f * DT + g * sq * eps
                recon = recon + Normal(z[:, 0], sig).log_prob(xsK[i + 1, :, 0])

            logw = (recon + logp0 - logq0 - logRN).reshape(n, K)
            elbos.append(logw.mean(dim=1))
            iwaes.append(torch.logsumexp(logw, dim=1) - math.log(K))
    return (float(torch.cat(elbos).mean()), float(torch.cat(iwaes).mean()),
            T - 1)


def part1_oos_nll(v1, latent, train_ds, K=64):
    print("\n" + "=" * 72)
    print("[1] OUT-OF-SAMPLE NLL  (held-out = last 15% of each ticker, 2010-16)")
    print("=" * 72)
    nll_v1 = v1_oos_nll(v1, train_ds)
    _, Xva_np = build_window_dataset(train_ds)
    elbo, iwae, n_inc = latent_oos_bounds(latent, Xva_np, K=K)
    nll_elbo, nll_iwae = -elbo / n_inc, -iwae / n_inc
    print(f"  v1 MLE   exact conditional NLL / increment : {nll_v1:+.4f}")
    print(f"  latent   ELBO-bound NLL / increment        : {nll_elbo:+.4f}")
    print(f"  latent   IWAE-{K}-bound NLL / increment      : {nll_iwae:+.4f}")
    print("  Definitions differ (exact-conditional vs lower-bound-joint with")
    print("  observation noise): see module docstring. Lower is better for all.")
    winner = "v1 MLE" if nll_v1 < nll_iwae else "latent"
    print(f"  --> On the numbers as computed: {winner} is ahead.")
    return {"v1_nll": nll_v1, "latent_elbo_nll": nll_elbo,
            "latent_iwae_nll": nll_iwae, "winner": winner}


# ===========================================================================
# 2) Path realism vs real out-of-sample data
# ===========================================================================
ACF_LAGS = [1, 2, 3, 5, 10, 21]


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    v = (x * x).mean()
    if v <= 0 or len(x) <= lag:
        return float("nan")
    return float((x[:-lag] * x[lag:]).mean() / v)


def stats_block(returns: np.ndarray) -> dict:
    """returns: (n_series, n_days). ACFs averaged across series; moments pooled."""
    out = {}
    for lag in ACF_LAGS:
        out[f"acf_abs_{lag}"] = float(np.nanmean([_acf(np.abs(r), lag) for r in returns]))
    for lag in (1, 5, 21):
        out[f"acf_sq_{lag}"] = float(np.nanmean([_acf(r ** 2, lag) for r in returns]))
    pooled = returns.ravel()
    out["kurtosis"] = float(kurtosis(pooled, fisher=True))
    out["tail_3sig"] = float((np.abs(pooled) > 3 * pooled.std()).mean())
    return out


def simulate_v1_physical(sde, y0, n_days, n_paths, seed=0):
    g = torch.Generator().manual_seed(seed)
    sde.configure(measure="P")
    Y = torch.full((n_paths,), float(y0), dtype=DTYPE)
    t0 = torch.zeros(())
    sq = math.sqrt(DT)
    path = [Y.clone()]
    with torch.no_grad():
        for _ in range(n_days):
            yv = Y.unsqueeze(-1)
            mu = sde.drift_physical(t0, yv).squeeze(-1)
            sig = sde.diffusion(t0, yv).squeeze(-1)
            Y = Y + mu * DT + sig * sq * torch.randn(n_paths, dtype=DTYPE, generator=g)
            path.append(Y.clone())
    return np.diff(torch.stack(path, dim=1).numpy(), axis=1)   # (n_paths, n_days)


def simulate_latent_physical(model, n_days, n_paths, seed=0):
    g = torch.Generator().manual_seed(seed)
    tr = torch.tensor(T_REF, dtype=DTYPE)
    sq = math.sqrt(DT)
    # p(z0) is built from nn.Parameters, so the whole simulation (init included)
    # must sit under no_grad before we ever touch .numpy().
    with torch.no_grad():
        z = model.pz0_mean + model.pz0_logstd.exp() * torch.randn(
            n_paths, model.latent_dim, dtype=DTYPE, generator=g)
        path = [z[:, 0].clone()]
        for _ in range(n_days):
            h, gg = model.h(tr, z), model.g(tr, z)
            z = z + h * DT + gg * sq * torch.randn(
                n_paths, model.latent_dim, dtype=DTYPE, generator=g)
            path.append(z[:, 0].clone())
    return np.diff(torch.stack(path, dim=1).numpy(), axis=1)


def part2_realism(v1, latent, n_paths=100, seed=0):
    print("\n" + "=" * 72)
    print(f"[2] PATH REALISM vs real {PRIMARY_TICKER} 2017-2022 (out-of-sample)")
    print("=" * 72)
    prices = download_prices()
    real_px = prices[PRIMARY_TICKER].loc["2017-01-01":"2022-12-31"].dropna()
    real = np.diff(np.log(real_px.to_numpy()))[None, :]        # (1, n_days)
    n_days = real.shape[1]

    y0 = float(v1.y_mean.item())
    sim_v1 = simulate_v1_physical(v1, y0, n_days, n_paths, seed)
    sim_lat = simulate_latent_physical(latent, n_days, n_paths, seed)

    blocks = {"real": stats_block(real), "v1 MLE": stats_block(sim_v1),
              "latent": stats_block(sim_lat)}
    keys = ([f"acf_abs_{L}" for L in ACF_LAGS]
            + [f"acf_sq_{L}" for L in (1, 5, 21)] + ["kurtosis", "tail_3sig"])
    hdr = f"  {'metric':14s} {'real':>9s} {'v1 MLE':>9s} {'latent':>9s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    lines = ["| metric | real | v1 MLE | latent |", "|---|---|---|---|"]
    for k in keys:
        r, a, b = blocks["real"][k], blocks["v1 MLE"][k], blocks["latent"][k]
        print(f"  {k:14s} {r:9.4f} {a:9.4f} {b:9.4f}")
        lines.append(f"| {k} | {r:.4f} | {a:.4f} | {b:.4f} |")

    # --- plots: ACF(|r|) + QQ of standardized returns ----------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, blk, style in [("real", blocks["real"], "ko-"),
                              ("v1 MLE", blocks["v1 MLE"], "C0s--"),
                              ("latent", blocks["latent"], "C1^--")]:
        ax[0].plot(ACF_LAGS, [blk[f"acf_abs_{L}"] for L in ACF_LAGS], style,
                   label=label)
    ax[0].axhline(0, color="gray", lw=0.5)
    ax[0].set_xlabel("lag (days)"); ax[0].set_ylabel("ACF of |returns|")
    ax[0].set_title("volatility clustering"); ax[0].legend()

    probs = np.linspace(0.005, 0.995, 99)
    qr = np.quantile(real.ravel() / real.std(), probs)
    for label, sim, color in [("v1 MLE", sim_v1, "C0"), ("latent", sim_lat, "C1")]:
        qs = np.quantile(sim.ravel() / sim.std(), probs)
        ax[1].plot(qr, qs, color=color, label=label)
    lims = [qr.min(), qr.max()]
    ax[1].plot(lims, lims, "k--", lw=0.8, label="perfect match")
    ax[1].set_xlabel("real quantiles (standardized)")
    ax[1].set_ylabel("model quantiles")
    ax[1].set_title("QQ: tails vs real"); ax[1].legend()
    fig.tight_layout()
    out = OUTPUT_DIR / "eval_latent_realism.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out.name}")
    return blocks, lines


# ===========================================================================
# 3) Regime RMSE table with the latent model added
# ===========================================================================
def latent_grid_prices(model, vol_shift, S0, r):
    import analysis.regime_comparison as rc
    out = {}
    for T in rc.MATURITIES:
        for m in rc.MONEYNESS:
            K, opt = S0 * m, ("call" if m >= 1.0 else "put")
            res = price_european_latent(model, S0, K, float(T), r, opt,
                                        n_paths=30000, n_steps=80,
                                        vol_shift=vol_shift)
            out[(round(float(T), 3), round(float(m), 3))] = res.price
    return out


def part3_regime_table(v1, latent):
    import analysis.regime_comparison as rc
    print("\n" + "=" * 72)
    print("[3] REGIME RMSE vs empirical option prices (Phase-7 protocol + latent)")
    print("=" * 72)
    set_seed(0)
    rc.S0 = round(float(np.exp(v1.y_mean.item())), 1)
    print(f"  spot S0={rc.S0} (v1 training-domain centre; latent is level-free)")
    _, regimes = build_regime_splits()
    r = 0.03
    rows = []
    for name, ds in regimes.items():
        rets = ds.primary_log_returns()
        sigma_ann = float(np.std(rets) * np.sqrt(TRADING_DAYS))
        print(f"  regime '{name}' (realised vol {sigma_ann:.1%}) ...")
        target = rc.empirical_prices(rets, sigma_ann)
        bs = rc.bs_prices(sigma_ann)
        hes = rc.heston_prices(rc.calibrate_heston(target, sigma_ann))
        nsde = rc.neural_sde_prices(v1, rc.calibrate_vol_shift(v1, sigma_ann))
        lsh = calibrate_latent_vol_shift(latent, sigma_ann, rc.S0, r)
        lat = latent_grid_prices(latent, lsh, rc.S0, r)
        rows.append({
            "regime": name, "realized_vol": sigma_ann,
            "Black-Scholes": rc._rmse(bs, target),
            "Heston (COS)": rc._rmse(hes, target),
            "Neural SDE v1": rc._rmse(nsde, target),
            "Latent SDE": rc._rmse(lat, target),
        })

    models = ["Black-Scholes", "Heston (COS)", "Neural SDE v1", "Latent SDE"]
    hdr = (f"  {'regime':8s} {'real vol':>9s}"
           + "".join(f" {m:>14s}" for m in models))
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))
    lines = ["| Regime | Realised vol | " + " | ".join(models) + " |",
             "|---|---|" + "---|" * len(models)]
    for row in rows:
        print(f"  {row['regime']:8s} {row['realized_vol']:>8.1%}"
              + "".join(f" {row[m]:>14.4f}" for m in models))
        lines.append(f"| {row['regime']} | {row['realized_vol']:.1%} | "
                     + " | ".join(f"{row[m]:.4f}" for m in models) + " |")

    x = np.arange(len(rows)); w = 0.2
    plt.figure(figsize=(9, 5))
    for i, m in enumerate(models):
        plt.bar(x + (i - 1.5) * w, [row[m] for row in rows], w, label=m)
    plt.xticks(x, [row["regime"] for row in rows])
    plt.ylabel("RMSE vs empirical prices")
    plt.title("Pricing error by model and regime (incl. latent SDE)")
    plt.legend(); plt.tight_layout()
    out = OUTPUT_DIR / "eval_latent_regime.png"
    plt.savefig(out, dpi=130); plt.close()
    print(f"  wrote {out.name}")
    return rows, lines


# ===========================================================================
def main():
    set_seed(0)
    v1 = load_trained()
    latent = load_trained_latent()
    train_ds, _ = build_regime_splits()

    res1 = part1_oos_nll(v1, latent, train_ds)
    _, md2 = part2_realism(v1, latent)
    rows3, md3 = part3_regime_table(v1, latent)

    md = ["# Latent SDE vs v1 MLE — evaluation results", "",
          "## 1. Out-of-sample NLL (per daily increment; lower is better)", "",
          f"- v1 MLE exact conditional NLL: **{res1['v1_nll']:+.4f}**",
          f"- latent ELBO-bound NLL: **{res1['latent_elbo_nll']:+.4f}**",
          f"- latent IWAE-64-bound NLL: **{res1['latent_iwae_nll']:+.4f}**",
          "- Caveat: exact-conditional vs lower-bound-joint-with-obs-noise — "
          "not the same definition; see analysis/eval_latent.py docstring.",
          "", "## 2. Path realism vs real SPY 2017-2022", ""] + md2 + [
          "", "## 3. Regime RMSE (empirical option prices)", ""] + md3
    RESULTS_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[eval] wrote {RESULTS_MD.name}")


if __name__ == "__main__":
    main()
