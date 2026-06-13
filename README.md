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
pytest -q             # runs the tests (33 of them)
```

## Extension: latent neural SDE (stochastic volatility)

The model above treats the (log-)price as the entire state, so its volatility
can only ever be a function of the current price level. It learns the leverage
effect but **structurally cannot represent stochastic volatility** — volatility
that has its own random dynamics — which is the main reason a calibrated Heston
model still beats it.

This extension lifts the dynamics into a **2-dimensional hidden state** `Z`:
one coordinate plays log-price, the second is free to become a volatility
state. The price you observe is a noisy readout of the first coordinate,
`x_t ~ N(z_t[0], sigma_obs^2)`. Volatility is now a genuine latent factor with
its own SDE, not a fixed function of price.

Does the second coordinate actually become volatility? It is never shown
realized volatility during training, yet after training the model's
instantaneous diffusion along the inferred path correlates **+0.37** with the
trailing realized volatility of the held-out series. The free coordinate
learned to track vol on its own.

### Training: variational ELBO with a shared diffusion

We can't observe `Z`, so the model is trained variationally. There are two
drifts and one diffusion:

* a **prior** SDE (generative): drift `f_prior(z, t)`;
* a **posterior** SDE (inference): drift `f_post(z, t, context)`, where the
  context is produced by a **backwards GRU** over the observed window — running
  the encoder in reverse time lets the latent state at time `t` condition on
  the *future* of the path, which is what actually identifies volatility (this
  is smoothing, not lagging filtering);
* **one diffusion network `g(z, t)`, shared by both.**

The objective is the ELBO:

```
ELBO  =  E_q[ log p(x | z) ]                                   (reconstruction)
         -  KL[ q(z0) || p(z0) ]                               (initial state)
         -  E_q[ integral_0^T  0.5 * ||(f_post - f_prior)/g||^2  dt ]   (paths)
```

The last term is the Kullback-Leibler divergence between the posterior and
prior path measures. By **Girsanov's theorem** that KL is exactly the
integrated, diffusion-scaled **drift mismatch** shown above — which is why the
prior and posterior *must* share one diffusion: if their diffusions differed,
the two path measures would be mutually singular and the KL would be infinite.
`torchsde.sdeint(..., logqp=True)` computes this term natively. A linear ramp
("KL annealing") raises the weight on the KL from 0 to 1 over the first epochs
so the decoder learns something before the KL pulls the posterior toward the
prior.

### Guarding against posterior collapse

The classic failure of these models is *posterior collapse*: the latent is
ignored and the posterior just equals the prior. The training loop watches for
it directly:

* the pathwise KL and the initial-state KL are logged **separately** every
  epoch (not just the total loss);
* a **loud alarm** fires if the pathwise KL falls below a configurable
  threshold after annealing finishes (encoder being ignored);
* a **usage diagnostic** — the correlation between the latent vol coordinate
  and trailing realized volatility on validation — is printed every epoch;
* the observation noise `sigma_obs` has a hard floor, so the decoder can't make
  reconstruction trivially cheap and starve the latent.

On the full run the pathwise KL settled around 19-23 (no collapse) and the
usage correlation stayed clearly non-zero.

### Risk-neutral pricing with a hidden vol state

For the original model, switching to the risk-neutral measure was a clean drift
swap, because volatility was a function of the observed price. With a *hidden*
vol coordinate the change of measure is no longer unique — this is the **market
price of volatility risk**. We make the standard, explicit assumption:

> **`lambda_vol = 0`** — zero market price of volatility risk. The price
> coordinate gets the martingale drift `r - 0.5 * g0^2`; the volatility
> coordinate keeps its physical (prior) dynamics unchanged.

This is the same assumption textbook Heston pricing folds invisibly into its
calibrated parameters. It is also the assumption that *matches this project's
benchmark*, whose target prices are built from realized returns re-centred to
the risk-neutral drift (so they carry no volatility risk premium by
construction). A martingale test confirms the discounted simulated price equals
the spot to within Monte Carlo error. Pricing can start the vol state either
unconditionally (from the prior) or **conditioned on a recent window** via the
encoder/filter — the latter is a capability the original model simply doesn't
have.

### Honest evaluation vs the original MLE model

Measured head-to-head. **The original MLE model wins the headline metrics — the
latent model does not beat it on pricing.** Reported plainly:

**1. Out-of-sample likelihood** (per daily increment, lower is better). Note
these are not the same quantity: the MLE number is an exact conditional density;
the latent numbers are *lower bounds* on a joint likelihood that also pays for
observation noise — a definitional handicap to the latent model. Even so, taken
at face value:

| metric | value |
|---|---|
| MLE, exact conditional NLL | **-2.859** |
| Latent, ELBO bound | -2.730 |
| Latent, IWAE-64 bound | -2.809 |

The MLE model is ahead.

**2. Regime pricing RMSE** vs empirical option prices, each model given the same
single vol-level knob (Heston excepted — it is fit to the targets, as before):

| Regime | Black-Scholes | Heston | Neural SDE (MLE) | **Latent SDE** |
|---|---|---|---|---|
| Calm (2017-19) | 0.0475 | 0.0065 | 0.0322 | **0.1174** |
| Crisis (2020) | 0.5900 | 0.0835 | 0.5573 | **1.0808** |
| Bear (2022) | 0.0749 | 0.0086 | 0.0707 | **0.2969** |

**The latent model is the worst of the four in every regime — worse than plain
Black-Scholes.** With only a single multiplicative vol-level knob and an
unconditional (prior-drawn) initial vol state, its implied smile is mis-shaped
relative to the benchmark; the extra flexibility hurt rather than helped under
this thin calibration. This is the clearest place the original model wins.

**3. Path realism** — simulate from each model's prior and compare to real SPY
(2017-2022) on volatility clustering and fat tails:

| metric | real | MLE | latent |
|---|---|---|---|
| ACF of \|returns\|, lag 1 | 0.409 | 0.005 | 0.014 |
| ACF of \|returns\|, lag 5 | 0.374 | 0.011 | 0.017 |
| excess kurtosis | 13.40 | 0.30 | 0.05 |
| P(\|return\| > 3 sigma) | 0.0159 | 0.0039 | 0.0031 |

**Both models are far from real markets here.** Real returns cluster strongly
(ACF ~0.4) and are extremely fat-tailed (excess kurtosis ~13); both models are
nearly Gaussian and nearly memoryless by comparison. The latent model shows
*slightly* stronger volatility clustering than the MLE model (consistent with
having a vol factor at all) but slightly *thinner* tails — neither is a win
worth claiming.

**Summary, no spin:** the latent extension is a more expressive, genuinely
stochastic-volatility generative model whose hidden coordinate demonstrably
learned to represent volatility (+0.37 correlation, arbitrage-free pricing
verified). But on the metrics that matter for an options pricer — out-of-sample
likelihood and regime pricing accuracy — **the simpler MLE model is better**
under the current training budget and thin (one-knob) pricing calibration.
Closing that gap (richer Q-calibration to option data, conditional initial vol,
correlated noise) is the natural next step; see Limitations.

### Limitations

* **Latent dimension fixed at 2.** Chosen for interpretability (price + one vol
  factor). Real volatility has multiple time scales; `d > 2` is an obvious
  ablation that this work does not run.
* **Market price of volatility risk assumed zero (`lambda_vol = 0`).** Defensible
  and explicit, but the real variance risk premium is non-zero; with `lambda`
  unidentified from returns alone, pricing inherits this assumption.
* **No option-market data used for Q-calibration.** The model is fit to
  historical returns only and pricing is level-matched with a single knob. A
  real desk would calibrate the risk-neutral dynamics to an option chain; doing
  so would likely fix much of the regime-RMSE gap above.
* **Diagonal noise.** The two Brownian shocks are independent, so the model
  cannot put correlation *in the shocks* (Heston's `rho`); any leverage effect
  must come through state-dependent drift/diffusion. This caps how much skew it
  can generate.
* **Unconditional initial vol for the regime test.** The benchmark uses a
  prior-drawn `v0`; conditioning `v0` on each regime's recent window (supported
  in the code) is untested in the table and would be a fairer, likely better,
  comparison.
* **Both models miss real fat tails and volatility clustering** when simulated
  from their priors (see table above). Capturing those is unfinished business
  for both.
* **Data fallback.** When `yfinance` is unavailable the pipeline uses a
  synthetic (Heston-like) dataset; results above are on whatever the cache
  holds, and the synthetic generator's structure can flatter or penalize models
  differently than live data would.

### Files in this extension

| File | What it is |
|---|---|
| `models/latent_sde.py` | the latent SDE: prior/posterior drifts, shared diffusion, backwards-GRU encoder, decoder, learned `q(z0)` |
| `models/train_latent.py` | variational ELBO training, KL annealing, collapse alarm + vol-usage diagnostic |
| `pricing/latent_pricing.py` | risk-neutral MC pricing (`lambda_vol = 0`), encoder-filtered initial vol state |
| `analysis/eval_latent.py` | the three-part evaluation above (OOS NLL, path realism, regime RMSE) |
| `tests/test_latent_sde.py` | module invariants, incl. the shared-diffusion identity check |
| `tests/test_latent_pricing.py` | martingale / arbitrage-free pricing checks |
| `config.py` | added latent window + KL-annealing + collapse-threshold settings |
| `docs/latent_sde_extension_session.md` | a log of the build session for this extension |

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
