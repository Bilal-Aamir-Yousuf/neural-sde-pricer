"""Run the entire Neural SDE Options Pricer pipeline end-to-end.

    python run_all.py            # full pipeline (trains, prices, plots, benchmarks)
    python run_all.py --quick    # smaller path counts / epochs for a fast smoke run
"""
from __future__ import annotations

import argparse
import time


def banner(msg):
    print("\n" + "=" * 70 + f"\n  {msg}\n" + "=" * 70)


def main(quick=False):
    t0 = time.time()

    banner("Phase 1 — Data pipeline")
    from data.fetch_data import build_regime_splits, fetch_options_chain
    train_ds, regimes = build_regime_splits()
    chain = fetch_options_chain()
    print(f"train days={len(train_ds.prices)}, regimes={list(regimes)}, "
          f"chain rows={len(chain)}")

    banner("Phase 2-3 — Train the Neural SDE (Euler-Maruyama MLE)")
    from models.train import train, load_trained
    train(epochs=120 if quick else 400)
    sde = load_trained()

    banner("Phase 4 — Monte Carlo pricing + convergence")
    from pricing.monte_carlo import price_option, convergence_study
    S0 = float(__import__("torch").exp(sde.y_mean).item())
    npaths = 10000 if quick else 40000
    for kind, kw in [("european", {}), ("asian", {}),
                     ("lookback", {"strike": "floating"}),
                     ("barrier", {"barrier": S0 * 1.3, "barrier_type": "up-and-out"})]:
        res = price_option(sde, S0, S0, 1.0, kind=kind, n_paths=npaths, **kw)
        print(f"  {kind:10s} {res}")
    convergence_study(sde, S0, S0, 1.0,
                      path_counts=(1000, 2000, 4000, 8000) if quick
                      else (500, 1000, 2000, 4000, 8000, 16000, 32000, 64000))

    banner("Phase 5 — Implied-volatility surface")
    from analysis.vol_surface import main as vol_main
    vol_main()

    banner("Phase 6 — Greeks via adjoint (validation table)")
    from pricing.greeks import validate_greeks
    validate_greeks(sde, S0=round(S0, 1), K=round(S0, 1), T=1.0,
                    n_paths=10000 if quick else 20000)

    banner("Phase 7 — Benchmark across regimes")
    from analysis.regime_comparison import run_comparison
    run_comparison()

    banner(f"DONE in {time.time() - t0:.0f}s — see outputs/ for plots & tables")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
