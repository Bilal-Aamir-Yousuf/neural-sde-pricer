# Session log — Latent Neural SDE extension

**Scope:** this document covers ONE working session (2026-06-12 → 2026-06-13,
Claude Code session `c54ef4db`): the design and build of the *latent* neural
SDE extension. The original pricer (the MLE-trained `NeuralSDE`, `train.py`,
the Phase 1–7 pipeline) was built in earlier, separate sessions and is
referenced here only where this session touched it — as the codebase being
extended and as the comparison baseline. Its construction is deliberately not
recounted.

It is a chronological log of what actually happened, including the dead ends.

---

## Stage 0 — Setup and ground rules

The session opened with a standing instruction: all extension work must be
exportable on its own (this document), separate from the original project's
history. The repo was surveyed (packages: `data/`, `models/`, `pricing/`,
`analysis/`, `tests/` — 25 tests passing at session start) and the obligation
recorded before any extension work began.

## Stage 1 — The extension spec, and a read-first planning pass

**Goal given:** extend the pricer to a latent neural SDE — a 2-D hidden state
Z (one coordinate playing log-price, one free to become volatility), observed
price as a noisy readout; variational training with a prior SDE and a
posterior SDE (conditioned on the observed path via a backwards GRU encoder)
**sharing one diffusion network**; ELBO via `torchsde.sdeint(..., logqp=True)`.
Explicit instruction: read the existing code first, plan only, no code.

**Why extend at all:** the existing model's state *is* the log-price, so its
volatility can only ever be a function of the current price level. It learned
the leverage effect but structurally cannot represent stochastic volatility —
the stated reason calibrated Heston beat it in the regime comparison.

**Mismatches flagged between the spec and the actual code** (planning
deliverable, before any building):

1. The v1 networks take `(Y, t)`, not just Y — but training evaluates them at
   a frozen `t=0` while pricing feeds them `t ∈ [0, T]`: a latent
   extrapolation wrinkle the extension should not replicate.
2. No "observed path" dataset existed: v1 trains on pooled one-day increments;
   the posterior encoder needs a *windowed sequence* dataset (new).
3. The ELBO as specified omitted the initial-state term
   `KL(q(z0|path) ‖ p(z0))` — `logqp` provides only the pathwise Girsanov
   integral.
4. Q-measure pricing is underdetermined for a latent model (market price of
   volatility risk) — the v1 "keep g, swap drift" trick doesn't pin down the
   vol coordinate. Deferred to a dedicated design discussion (Stage 7).
5. The existing regime benchmark flatters Heston: its 5 parameters are
   least-squares fit to the very empirical prices the RMSE is scored against,
   while the neural model gets a single vol-level shift. The latent model
   must be scored under the same one-knob protocol.
6. `noise_type="diagonal"` (the practical constraint for native logqp) cannot
   express Brownian price–vol correlation (Heston's ρ); leverage must enter
   through state dependence. Accepted as a v1 ceiling.
7. Decision recorded: the decoder should be *structured* (coordinate 0 IS
   log-price) — a free-form decoder makes Q-pricing and spot-conditioning
   ill-defined.

## Stage 2 — Design decisions, interrogated before building

Five design questions were asked and answered with justifications:

- **Latent dim = 2, hardcoded, dim as a later ablation.** d=2 caps the model
  at one-factor stochastic vol; the measurable criterion for revisiting is
  **per-coordinate KL usage** (a dimension carrying ≈0 KL is provably unused)
  plus validation ELBO and regime RMSE.
- **One shared diffusion, enforced structurally.** Two SDEs with different
  diffusions have mutually singular path laws (quadratic variation identifies
  g almost surely) — the true KL would be +∞ and the logqp integral would
  silently stop being a KL. Enforcement: a single `nn.Module` owns ONE
  `diffusion_net`; `prior_diffusion` and `posterior_diffusion` are property
  aliases of the same object so a test can identity-check (`is`, not weight
  equality).
- **Backwards GRU encoder = smoothing, not filtering.** The optimal posterior
  at time t conditions on the *future* of the observed path (where the path
  went reveals what vol was). The GRU consumes the window in reverse, so
  ctx[i] is a function of x_{i:T}; the past enters through z_t itself. Context
  lookup uses `searchsorted(..., right=False)` so at a grid time t_i the
  context includes x_{t_i}, strictly between grid points only future
  observations.
- **Observation model, explicit:** `x_t | z_t ~ N(z_t[0], σ_obs²)` — learned
  scalar σ_obs through softplus with a ~1bp floor. Failure modes named: σ_obs
  inflating (latents ignored) vs σ_obs → 0 (likelihood divergence); the floor
  plus β-annealing blocks both, and σ_obs is logged every epoch.
- **q(z0):** diagonal Gaussian head on the reverse-GRU state at t=0 (a
  function of the whole window); prior p(z0) a learned global diagonal
  Gaussian — the unconditional initial law that generative sampling and
  unconditional pricing draw from.

## Stage 3 — The module (`models/latent_sde.py`)

Built against the torchsde contract **verified from installed source, not
guessed** (`_core/base_sde.py`, `_core/sdeint.py`):

- Method names: drift → `f` (posterior), diffusion → `g`, prior drift → `h`
  (`RenameMethodsSDE` defaults).
- `SDELogqp` augments the state with a channel whose drift is
  `0.5‖(f−h)/g‖²` and whose diffusion is **zero** — the expectation-form
  Girsanov integrand, so the returned log-ratio increments are nonnegative
  **pathwise**, not just on average.
- `sdeint(..., logqp=True)` returns `(ys, log_ratio_increments)` shaped
  `(T, batch, d)` and `(T−1, batch)`.

Smoke tests (4): forward shapes on batch 8 × seq 50; KL terms finite and
nonnegative (pathwise, per sample); **identity check** that prior and
posterior reference the same diffusion object; f/h/g signature/shape contract.
Result: `4 passed in 5.56s`; full suite 29/29.

## Stage 4 — Training loop (`models/train_latent.py`) and two init pathologies

Loss `= −recon + β·(KL_path + KL_z0)`, β ramping 0→1 linearly over
`KL_ANNEAL_EPOCHS` (config flag), all terms logged separately per epoch.
Data: the existing `build_regime_splits()` pipeline (same basket, same
strictly-pre-2017 train window), cut into window-relative 64-day windows,
stride 5, temporal per-ticker validation split.

The smoke run (48 train / 16 val windows, 6 epochs) caught two genuine bugs:

1. **First run: recon ≈ −1.26×10⁷ and σ_obs frozen at 0.00161.** Cause:
   softplus-of-random-MLP initializes the diffusion ≈0.7/√yr (≈4× real equity
   vol) so prior paths wander O(1) in log-price; and `raw_obs_noise` started
   in softplus's saturated tail where its gradient direction is then crushed
   by global grad-clipping. Fix (in the module, documented): bias the
   diffusion head to ~0.15/√yr at init, start both drifts near zero, start
   σ_obs at ~1%.
2. **Second run: recon ≈ −3.1×10⁵ → −1.8×10⁵, plateaued.** Cause: the random
   `qz0_net` Linear put the posterior z0 at O(±1) while every window starts
   at exactly 0 — a constant offset a near-zero-drift path cannot recover
   from. Fix: z0 head initialized centered at 0 with logstd −3.

Third (clean) smoke run:

```
ep 0  beta 0.00 | recon -3306.53 | KL_path 0.006 | KL_z0 5.006 | val(-ELBO) 3012.30
ep 1  beta 0.33 | recon -2527.08 | KL_path 0.009 | KL_z0 5.018 | val(-ELBO) 2040.18
ep 5  beta 1.00 | recon -2661.03 | KL_path 0.000 | KL_z0 5.035 | val(-ELBO) 2456.33
```

`KL_z0 ≈ 5.03` matched the analytic Gaussian–Gaussian KL for the
initialization exactly (≈2.5/dim × 2). `KL_path ≈ 0` at this stage is by
design (both drifts start near zero), not collapse.

## Stage 5 — Collapse detection, the knobs question, and the full run

**Added to the loop:** a loud per-epoch alarm when pathwise KL <
`KL_COLLAPSE_THRESHOLD` (config, 0.05) after annealing completes, and a
per-epoch `corr(z1, realized vol)` on validation — the model is never shown
realized vol, so a sustained correlation is direct evidence the free
coordinate became a volatility state. At the end, the same correlation is
computed for the model's instantaneous price-vol `g(z)[0]` along the
posterior path (invariant to z1's arbitrary sign/scale). The alarm was
verified to fire on the (by-construction collapsed) smoke run.

**Collapse knobs, in the order to try them** (recorded for this architecture,
whose decoder is identity+noise — so "collapse" here specifically means *z1
goes unused* while z0 must track the data):

1. slower/cyclical annealing (free, no objective distortion);
2. freeze/cap σ_obs early — the decoder's only cheat channel;
3. free bits / per-term KL floors (distorts the objective; per-dim floors on
   the path term require recomputing the integrand from our own f/h/g since
   logqp returns it summed over dims);
4. strengthen the inference path (wider ctx/GRU, realized-vol features into
   the encoder — legal, it's a function of the observed path);
5. architectural surgery: make g0 depend only on (z1, t) — guarantees usage,
   but changes the model class, and the unconstrained model discovering vol
   on its own is the stronger result.
   (β_max < 1 rejected: pricing simulates from the prior; a posterior-
   flattered generative model is what we can't afford.)

**Full training** (2,400 train / 256 val windows, 120 epochs, ~7s/epoch after
measuring 0.12s per batch-of-128): healthy throughout — no collapse alarm
after annealing. Selected epochs:

```
ep   0  beta 0.00 | recon -2230.84 | KL_path  0.751 | KL_z0 6.057 | val(-ELBO) 1510.65 | corr(z1,rv) -0.061
ep  11  beta 0.37 | recon    86.53 | KL_path 36.715 | KL_z0 7.966 | val(-ELBO)   37.93 | corr(z1,rv) -0.268
ep  30  beta 1.00 | recon   162.05 | KL_path 12.651 | KL_z0 3.096 | val(-ELBO) -154.52 | corr(z1,rv) -0.185
ep  59  beta 1.00 | recon   183.50 | KL_path 20.698 | KL_z0 1.899 | val(-ELBO) -169.13 | corr(z1,rv) -0.114
ep 119  beta 1.00 | recon   188.09 | KL_path 23.270 | KL_z0 2.036 | val(-ELBO) -172.07 | corr(z1,rv) -0.156
```

Final (best checkpoint, validation), σ_obs converged to 0.0129:

| metric | value |
|---|---|
| reconstruction NLL / window | **−192.73** |
| KL pathwise (logqp) | **18.93** |
| KL z0 | **1.89** |
| corr(z1, realized vol) | **−0.174** |
| corr(g(z)[0], realized vol) | **+0.374** (sign/scale-invariant) |
| best val −ELBO | −172.81 |

Read honestly: the latent channel is alive (KL_path ≈ 19–23, alarm silent),
and the model's instantaneous vol tracks realized vol it was never shown at
+0.37 pooled correlation. z1's raw correlation is modest and negative (sign
is arbitrary); the vol information is partly encoded nonlinearly through
g(z), which is why the invariant diagnostic was added.

## Stage 6 — Evaluation harness (built; see status note)

`analysis/eval_latent.py`, three parts with metric definitions stated up
front, plus `pricing/latent_pricing.py` for part 3:

1. **OOS NLL** on the same held-out segment (last 15% of each ticker,
   2010–16). v1: exact conditional Gaussian transition NLL per increment.
   Latent: ELBO and IWAE-64 **lower bounds** on the *joint* path likelihood
   per increment — with the importance weights using the full pathwise
   Girsanov Radon-Nikodym derivative including the martingale term `∫u·dW`
   (torchsde's logqp provides only the expectation-form integrand, which is
   not the pathwise density ratio). Stated plainly in the code: these are
   different definitions; the latent number carries a structural handicap
   (bound + observation-noise terms), so latent>v1 would be decisive while
   v1>latent is partly attributable to the definition.
2. **Path realism** vs real out-of-sample SPY 2017–2022: ACF of |r| and r² at
   multiple lags, excess kurtosis, P(|r|>3σ), QQ plot — both models simulated
   from their physical/prior dynamics with frozen time features ("days since
   window start" has no economic meaning).
3. **Regime RMSE** regenerated with a Latent SDE column under the same
   one-knob (vol-shift) protocol as v1.

**Status note:** the harness was built mid-session and its execution deferred
(first by the Q-measure design question in Stage 7, then by the export). It was
run in a later turn of this session — one bug surfaced and was fixed (a grad
leak: the latent prior-path simulation built `z0` from `nn.Parameter`s outside
`no_grad` before calling `.numpy()`). Results below are the actual run; they are
reported with no spin, including where the original model wins.

**1. Out-of-sample NLL** (per daily increment, lower better): MLE exact
conditional **−2.859** vs latent ELBO **−2.730** / IWAE-64 **−2.809**. The MLE
model wins — though the comparison is not apples-to-apples (exact conditional
density vs a lower bound on a joint likelihood that also pays for observation
noise; the handicap is on the latent side).

**2. Regime RMSE** vs empirical option prices (same single vol-level knob for
both neural models):

| Regime | Black-Scholes | Heston | Neural SDE (MLE) | Latent SDE |
|---|---|---|---|---|
| calm | 0.0475 | 0.0065 | 0.0322 | 0.1174 |
| crisis | 0.5900 | 0.0835 | 0.5573 | 1.0808 |
| bear | 0.0749 | 0.0086 | 0.0707 | 0.2969 |

The latent model is the **worst of the four in every regime, worse than plain
Black-Scholes.** Under one multiplicative knob and an unconditional prior `v0`,
its smile is mis-shaped; the added flexibility hurt under this thin
calibration. This is the clearest place the original model wins.

**3. Path realism** (simulate from each prior vs real SPY 2017–22): real
returns cluster (ACF|r|₁ ≈ 0.41) and are fat-tailed (excess kurtosis ≈ 13.4);
both models are nearly memoryless (ACF ≈ 0.005–0.014) and near-Gaussian
(kurtosis 0.30 MLE / 0.05 latent). The latent clusters *slightly* more than the
MLE model but has *slightly* thinner tails — neither is close to real data.

Honest summary: the latent model demonstrably learned a volatility state
(+0.37 corr, arbitrage-free pricing verified) but does **not** beat the simpler
MLE model on out-of-sample likelihood or regime pricing under the current
training budget and one-knob calibration. The martingale battery (below) is the
other executed validation.

## Stage 7 — Risk-neutral pricing with a latent vol state

The v1 trick (replace drift with `r − ½σ(Y)²`) does not transfer: σ is now a
hidden coordinate. Three candidate approaches were laid out:

1. **Drift-swap on the price coordinate only; λ_vol ≡ 0** (assumption stated:
   zero market price of volatility risk — the same λ that Heston calibration
   absorbs invisibly into κ*, θ*). Exact martingale, no extra data; blind to
   the empirically negative variance risk premium.
2. **Filtering for the initial condition** (orthogonal to the measure): price
   = E over p(z_t | observed history). Pipeline: encode trailing window →
   sample q(z0) → integrate the posterior SDE through the window → terminal
   states are the filtering posterior (smoothing at the boundary = filtering)
   → pin the price coordinate to the live spot → Q-simulate from the mixture.
3. **Calibrate λ to market option prices** (the desk answer): imports real
   risk premia, but on a single chain snapshot λ becomes a garbage-collector
   for model misspecification — and calibrating the *diffusion* to options
   would be remodeling, not a measure change.

**Default adopted: 1 + 2.** Beyond identifiability, a project-specific
argument: the regime benchmark's target prices are bootstrapped from realized
returns *re-centred to the risk-neutral drift* — they contain **no vol risk
premium by construction**, so λ=0 is the measure that matches the evaluation
target, not merely the convenient one.

Implementation (`pricing/latent_pricing.py`): Q-drift `r − ½g0(z)²` on
coordinate 0 (using the *shifted* g0 when the level-match knob is engaged,
mirroring v1), prior drift on the vol coordinate, diffusion untouched;
`filtered_state_samples()` for conditional pricing; initial vol states
duplicated across antithetic halves so each ±ε pair shares its v0.

**Martingale sanity test** (`tests/test_latent_pricing.py`). A property worth
recording: with drift `r − ½g0²` and g0 measurable at each step's start, the
log-space Euler scheme makes the discounted price an **exact discrete-time
martingale** (lognormal MGF per step) — so the test demands agreement within
pure Monte Carlo error (5 standard errors), no discretization slack. It runs
on an *untrained* model first (the property must hold for any weights, by
construction), then the trained checkpoint. 100,000 paths each:

```
untrained, prior v0     E[disc S_T] =  99.9976   (stderr 0.0036, err = 0.66 se)   PASSED
untrained, filtered v0  E[disc S_T] = 100.0004   (stderr 0.0036, err = 0.11 se)   PASSED
untrained, pinned v0    E[disc S_T] =  99.9977   (stderr 0.0037, err = 0.64 se)   PASSED
trained checkpoint      E[disc S_T] =  99.9942   (stderr 0.0049, err = 1.20 se)   PASSED
```

## State at export time

- New files this session: `models/latent_sde.py`, `models/train_latent.py`,
  `pricing/latent_pricing.py`, `analysis/eval_latent.py`,
  `tests/test_latent_sde.py`, `tests/test_latent_pricing.py`, config flags
  (`LATENT_WINDOW_LEN/STRIDE`, `KL_ANNEAL_EPOCHS`, `KL_COLLAPSE_THRESHOLD`),
  this document.
- Test suite: **33 passing** (25 pre-existing + 8 from this session).
- Trained checkpoint: `outputs/checkpoints/latent_sde.pt` (best val −ELBO
  −172.81), plus `latent_training.png` / `latent_vol_tracking.png`.
- Evaluation: executed (Stage 6 above). Headline finding, no spin — the
  original MLE model wins on OOS likelihood and on regime pricing RMSE; the
  latent model's genuine result is the interpretable, arbitrage-free vol state,
  not better prices. Tables also saved to `outputs/eval_latent_results.md`.
- Open next steps (to close the gap): calibrate the risk-neutral dynamics to an
  option chain instead of one vol knob; condition `v0` on each regime's recent
  window; consider correlated noise and `d > 2`.
