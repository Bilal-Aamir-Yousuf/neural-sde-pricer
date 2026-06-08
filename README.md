# Neural SDE Options Pricer

This project teaches a small neural network to learn how stock prices actually
move from real market data, and then uses what it learned to price options
(including tricky exotic ones).

## The idea in one paragraph

The classic Black-Scholes model assumes a stock has one fixed volatility
forever. Real markets don't work that way: volatility changes with the price
level and jumps around in a crisis. Instead of assuming a formula, this project
lets two small neural networks learn the "drift" (where the price tends to go)
and the "volatility" (how much it wiggles) directly from years of real price
data. Once the model knows how prices behave, you can simulate thousands of
possible futures and average the option payoffs to get a price.

## What it does

* Pulls real daily prices for SPY and several large stocks (via `yfinance`).
* Trains the neural network on that data.
* Prices options by simulating lots of price paths (Monte Carlo):
  * European, barrier, Asian, and lookback options.
* Computes the option "Greeks" (Delta, Gamma, Vega, Theta, Rho), which tell you
  how sensitive the price is to its inputs.
* Builds a volatility surface and compares it to the real options market.
* Benchmarks itself against Black-Scholes and the Heston model across calm,
  crisis, and bear-market periods.

## Main results

**It beats Black-Scholes in every market regime.** Each model was tested against
prices implied by what actually happened in the market. Lower error is better.

| Market period | Black-Scholes | Heston | Neural SDE |
|---|---|---|---|
| Calm (2017-2019) | 0.0475 | 0.0065 | **0.0328** |
| Crisis (2020 COVID) | 0.5900 | 0.0835 | **0.5466** |
| Bear (2022) | 0.0749 | 0.0086 | **0.0507** |

(Heston wins because it was fit directly to the test prices. The point is that
the learned model beats plain Black-Scholes everywhere.)

A few other checks that show it's working correctly:

* The model learns that volatility is higher when prices are low, which is what
  really happens in markets (the "leverage effect").
* It reproduces the volatility "skew" that real options show but Black-Scholes
  cannot.
* The Monte Carlo error shrinks at the expected `1/sqrt(N)` rate.
* The Greeks computed three different ways all agree.

Plots for all of this are saved in the `outputs/` folder.

## How to run it

You need 64-bit Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python run_all.py     # runs the whole thing
pytest -q             # runs the tests (25 of them)
```

## Project layout

```
data/      pulling and preparing market data
models/    the neural network and its training
pricing/   Monte Carlo engine, Greeks, and the baseline models
analysis/  volatility surface and the regime comparison
tests/     the test suite
outputs/   saved plots, tables, and the trained model
```

## One honest note

The model learns the volatility from one historical period. To price in a very
different period (like the 2020 crash) it gets re-scaled to that period's
volatility level. It captures a lot of what real markets do, but not everything,
which is why a model fit directly to live prices (Heston) still scores best.
