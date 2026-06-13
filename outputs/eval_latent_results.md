# Latent SDE vs v1 MLE — evaluation results

## 1. Out-of-sample NLL (per daily increment; lower is better)

- v1 MLE exact conditional NLL: **-2.8594**
- latent ELBO-bound NLL: **-2.7297**
- latent IWAE-64-bound NLL: **-2.8092**
- Caveat: exact-conditional vs lower-bound-joint-with-obs-noise — not the same definition; see analysis/eval_latent.py docstring.

## 2. Path realism vs real SPY 2017-2022

| metric | real | v1 MLE | latent |
|---|---|---|---|
| acf_abs_1 | 0.4092 | 0.0047 | 0.0136 |
| acf_abs_2 | 0.4893 | 0.0052 | 0.0101 |
| acf_abs_3 | 0.4192 | 0.0099 | 0.0175 |
| acf_abs_5 | 0.3743 | 0.0109 | 0.0167 |
| acf_abs_10 | 0.3278 | 0.0095 | 0.0121 |
| acf_abs_21 | 0.1416 | 0.0006 | 0.0081 |
| acf_sq_1 | 0.4647 | 0.0045 | 0.0159 |
| acf_sq_5 | 0.3292 | 0.0109 | 0.0176 |
| acf_sq_21 | 0.0409 | 0.0000 | 0.0083 |
| kurtosis | 13.4049 | 0.3004 | 0.0459 |
| tail_3sig | 0.0159 | 0.0039 | 0.0031 |

## 3. Regime RMSE (empirical option prices)

| Regime | Realised vol | Black-Scholes | Heston (COS) | Neural SDE v1 | Latent SDE |
|---|---|---|---|---|---|
| calm | 12.8% | 0.0475 | 0.0065 | 0.0322 | 0.1174 |
| crisis | 48.8% | 0.5900 | 0.0835 | 0.5573 | 1.0808 |
| bear | 24.2% | 0.0749 | 0.0086 | 0.0707 | 0.2969 |