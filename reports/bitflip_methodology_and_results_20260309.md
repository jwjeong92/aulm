# Bit-Flip Risk Methodology, Current Results, and Next Steps

Date: 2026-03-09

Implementation update: 2026-03-10

This document is the detailed working note for the current bit-flip robustness study in this repository. It consolidates:

- the problem statement
- the exact formulas currently used
- the meaning of each symbol
- which parts are already implemented
- what the current experiments say
- what is still missing before converting the scores into an actual protection budget

This file is intentionally more detailed than `reports/bitflip_experiments_20260309.md`.

## 1. Goal

We want to protect a quantized model against random bit errors while spending as little protection cost as possible.

The practical target is not:

- "reduce layer MSE"

but:

- "reduce task degradation under a given bit error process"

Examples of task degradation are:

- loss increase
- perplexity increase
- accuracy drop
- margin collapse

The main difficulty is that the direct relation

- layer output MSE -> accuracy drop

is often weak, noisy, and highly layer-dependent. The current code and experiments are trying to replace that with a more structured pipeline:

1. estimate local bit sensitivity
2. rank bits / weights / modules by expected damage
3. later convert that ranking into protection allocation

## 2. Repository Artifacts

### Existing summary report

- `reports/bitflip_experiments_20260309.md`

### Main experiment script for the earlier study

- `scripts/bitflip_hypothesis_experiment.py`

This script currently supports:

- `single_flip`
- `layer_full_lsb`

### New reusable bit-risk utilities

- `utils/bitflip_risk.py`

### New analytic ECC residual-BER utility

- `utils/ecc_residual.py`

### New decoder-layer multiplier estimator

- `scripts/estimate_decoder_layer_multiplier.py`

### RTN quantization path

- `lib/quantization/quantizer.py`
- `lib/quantization/weight_quant.py`

### New int8 asymmetric RTN risk script

- `scripts/rtn_int8_asym_bitflip_risk.py`

### New matched int8 sparse layer-evaluation script

- `scripts/rtn_int8_sparse_layer_eval.py`

This script is intended for the next matched-setting phase:

- stay in int8 asymmetric RTN
- select sparse top-risk bit faults within each decoder layer
- inject them into the quantized model
- measure `delta_loss` / `delta_ppl`
- export per-layer sparse-fault trial CSVs for multiplier fitting

### Relevant result files

- `cache/bitflip_hypothesis/20260309_042829_summary.json`
- `cache/bitflip_hypothesis/20260309_042829_corr_trials.csv`
- `cache/bitflip_hypothesis/20260309_042829_group_trials.csv`
- `cache/bitflip_hypothesis/20260309_135619_summary.json`
- `cache/bitflip_hypothesis/20260309_135619_layer_full_lsb_trials.csv`
- `cache/rtn_int8_asym_bitflip_risk/20260309_183147_summary.json`
- `cache/rtn_int8_asym_bitflip_risk/20260309_183147_layer_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk/20260309_183147_top_weight_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk/20260309_183147_top_bit_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk/20260309_183147_bit_position_risk.csv`
- `cache/bitflip_hypothesis_int8/20260309_185745_summary.json`
- `cache/bitflip_hypothesis_int8/20260309_185745_layer_full_lsb_trials.csv`
- `cache/decoder_layer_multiplier/20260309_185850_summary.json`
- `cache/decoder_layer_multiplier/20260309_185850_summary.md`
- `cache/decoder_layer_multiplier/20260309_185850_decoder_layer_multiplier.csv`
- `cache/rtn_int8_asym_bitflip_risk_layeraware/20260309_191702_summary.json`
- `cache/rtn_int8_asym_bitflip_risk_layeraware/20260309_191702_layeraware_layer_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk_layeraware/20260309_191702_layeraware_top_weight_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk_layeraware/20260309_191702_layeraware_top_bit_risk.csv`
- `cache/rtn_int8_group_budget_codeword64_proposal/20260310_023615_summary.json`
- `cache/rtn_int8_group_budget_codeword64_eccanalytic/20260310_140734_summary.json`
- `cache/rtn_int8_group_budget_codeword64_bchauto/20260310_141833_summary.json`

## 3. Quantization Setup

Two different settings appear in the current results:

### Setting A: 4-bit RTN asymmetric per-channel

Used in:

- `single_flip`
- `layer_full_lsb`

### Setting B: 8-bit RTN asymmetric per-channel

Used in:

- `rtn_int8_asym_bitflip_risk.py`
- `layer_full_lsb` rerun stored in `cache/bitflip_hypothesis_int8`

Important: results across Setting A and Setting B should not be mixed as if they were on the same absolute scale.

They can still be compared qualitatively, for example:

- which decoder layers appear dangerous
- whether risk is concentrated in high bits

but not yet quantitatively as a final calibrated predictor.

## 4. Exact Quantization Formula Used Here

The code path is based on `lib/quantization/quantizer.py`.

For a weight tensor `W` in a linear layer:

- "per-channel" in this code means per output channel
- for a 2D linear weight matrix, that means per row

Let:

- `r` be the output-channel index, or row index
- `c` be the input-channel index, or column index
- `w_{r,c}` be the floating-point weight
- `s_r` be the per-row scale
- `z_r` be the per-row zero-point
- `Q_max = 2^b - 1`
- `b` be the quantization bit-width

Then RTN asymmetric quantization is:

```text
q_{r,c} = clamp(round(w_{r,c} / s_r) + z_r, 0, Q_max)
w_hat_{r,c} = s_r * (q_{r,c} - z_r)
```

For the current int8 asymmetric workflow:

```text
b = 8
Q_max = 255
q_{r,c} in {0, 1, ..., 255}
```

Meaning:

- `q_{r,c}` is the stored integer
- `w_hat_{r,c}` is the quantized-dequantized floating-point value actually used by the model
- `s_r` controls how large one integer step is in row `r`
- `z_r` shifts the unsigned integer range so that it can represent asymmetric floating-point intervals

## 5. Bit-Flip Model

For one stored integer `q_{r,c}`, let bit position `k` be counted from LSB:

```text
k = 0, 1, ..., b - 1
mask_k = 2^k
```

A single bit flip is:

```text
q'_{r,c,k} = q_{r,c} XOR mask_k
delta q_{r,c,k} = q'_{r,c,k} - q_{r,c}
```

For unsigned int8 without clipping after XOR:

```text
|delta q_{r,c,k}| = 2^k
```

The corresponding floating-point weight perturbation is:

```text
delta w_{r,c,k} = s_r * delta q_{r,c,k}
```

Therefore:

```text
(delta w_{r,c,k})^2 = s_r^2 * 4^k
```

Meaning:

- higher bits are exponentially more dangerous than lower bits
- larger per-row scale makes every flipped bit in that row more dangerous
- the sign of `delta q` may change with the original stored value, but the squared magnitude only depends on `s_r^2 * 4^k`

## 6. Hessian-Diagonal Proxy Used in the Code

The RTN path uses `collect_rtn_hessian_diagonal()` in `lib/quantization/weight_quant.py`.

For a linear layer, the collector accumulates a diagonal quantity over the input activations to that layer.

Operationally, for input activation `x_n` of sample `n`, the code builds a per-column statistic:

```text
h_c approx (2 / N) * sum_n x_{n,c}^2
```

This is not the full task Hessian.

It is a diagonal proxy derived from the input Gram structure used in GPTQ-style second-order approximations. In practice it serves as:

- a column sensitivity measure
- a local curvature-like importance term

Meaning:

- if `h_c` is large, perturbing weights connected to input column `c` is predicted to be more damaging
- in the current implementation the Hessian proxy is column-wise, not element-wise

This is why:

- scale depends on row `r`
- Hessian proxy depends on column `c`

and the local score factorizes naturally into row and column contributions.

## 7. Local Bit Risk Formula

This is the central current scoring rule.

Let:

- `p_k` be the bit error probability for bit position `k`
- `h_c` be the Hessian-diagonal proxy for column `c`
- `s_r` be the per-row quantization scale
- `delta w_{r,c,k}` be the floating perturbation caused by flipping bit `k`

Then the local expected risk for one bit is:

```text
R_bit(r, c, k) approx (1/2) * h_c * p_k * (delta w_{r,c,k})^2
```

Substituting the RTN int8 asymmetric relation:

```text
R_bit(r, c, k) approx (1/2) * h_c * p_k * s_r^2 * 4^k
```

Meaning of each factor:

- `(1/2)`:
  second-order Taylor coefficient
- `h_c`:
  local sensitivity of that input column
- `p_k`:
  how often bit `k` flips
- `s_r^2`:
  how much one integer jump matters in that output channel
- `4^k`:
  why high bits dominate

Interpretation:

- if two weights have the same `h_c`, the one with larger scale is riskier
- if two weights have the same scale, the one connected to a larger `h_c` column is riskier
- for equal BER across bits, MSB dominates because of `4^k`

## 8. Bit Multiplier

When BER is specified per bit position, the bit-only term is:

```text
M_k = (1/2) * p_k * 4^k
```

Then:

```text
R_bit(r, c, k) approx h_c * s_r^2 * M_k
```

Meaning:

- `M_k` is the global danger weight of bit position `k`
- if all bits share the same BER, the ratio between adjacent bits is exactly 4

For the current int8 run:

```text
p_0 = p_1 = ... = p_7 = 1e-6
```

So:

```text
M_7 : M_6 : ... : M_0 = 4^7 : 4^6 : ... : 1
```

This is why the current result shows:

- bit 7 contributes about 75.0 percent of total risk
- bit 6 contributes about 18.75 percent
- bits 7 and 6 together contribute about 93.75 percent

## 9. Weight-Level Aggregate Risk

To score a weight location `(r, c)` across all bits:

```text
R_weight(r, c) = sum_k R_bit(r, c, k)
```

Therefore:

```text
R_weight(r, c) approx h_c * s_r^2 * sum_k M_k
```

Meaning:

- weight risk is high when a row has a large quantization scale and its column has a large Hessian proxy
- this is the quantity used to rank hotspots for selective protection

In the current implementation this is what `top_weight_risk.csv` is approximating.

## 10. Module-Level Aggregate Risk

For one 2D linear module:

```text
R_module = sum_r sum_c sum_k R_bit(r, c, k)
```

Using the factorized form:

```text
R_module approx (sum_r s_r^2) * (sum_c h_c) * (sum_k M_k)
```

This is exact under the current proxy assumptions because:

- `s_r` depends only on row
- `h_c` depends only on column
- `M_k` depends only on bit position

Meaning:

- module risk is large when the module has a lot of row-scale energy and a lot of column sensitivity mass
- this is what appears in `layer_risk.csv`

This module-level score is currently used as a broad ranking signal.

## 11. Decoder-Layer Aggregate Risk

A decoder layer contains multiple linear modules, such as:

- `self_attn.q_proj`
- `self_attn.k_proj`
- `self_attn.v_proj`
- `self_attn.out_proj`
- `fc1`
- `fc2`

For decoder layer `ell`:

```text
R_decoder(ell) = sum_{modules m in decoder layer ell} R_module(m)
```

Meaning:

- this is the local second-order risk mass contained in that decoder layer
- it does not yet model downstream amplification through the rest of the network

This distinction matters because the experiments show:

- local module risk and final task damage are related
- but decoder-layer depth and downstream propagation matter a lot too

## 12. Decoder-Layer Multiplier: Proposed but Not Yet Calibrated

To bridge the gap between local risk and final task damage, a proposed correction is:

```text
R_final(r, c, k) = alpha_{ell(r,c)} * R_bit(r, c, k)
```

where:

- `ell(r,c)` is the decoder layer containing that weight
- `alpha_ell` is the decoder-layer multiplier

Intuition:

- local score says how bad the bit is locally
- layer multiplier says how strongly damage in that decoder layer is amplified into final task degradation

A direct candidate definition is:

```text
alpha_ell = D_ell / (R_decoder(ell) + epsilon)
```

where:

- `D_ell` is an observed system-level damage metric for faults injected into decoder layer `ell`
- examples of `D_ell`:
  - `delta_loss`
  - `delta_ppl`
  - accuracy drop
- `epsilon` is a small numerical stabilizer

Current status:

- this idea is not yet implemented in the ranking code
- a reusable estimator now exists in `scripts/estimate_decoder_layer_multiplier.py`
- a provisional matched int8 estimate has been generated
- but the multiplier is still based on layer-level full-LSB stress damage, so it remains a calibration aid rather than a final deployment-ready constant

Therefore:

- the concept is useful
- the current data are useful for provisional layer-magnitude correction
- the current data are still not suitable for a final calibrated `alpha_ell`

## 13. Future Protection Objective

The final protection problem should be written in terms of residual BER after protection.

For a protection mode `m`, let:

- `p_k^(m)` be the residual BER of bit `k`
- `c_g(m)` be the protection cost for group `g`
- `R_g(m)` be the residual expected risk of group `g`

Then for a protectable group `g`:

```text
R_g(m) = sum_{(r,c,k) in g} (1/2) * h_c * p_k^(m) * (delta w_{r,c,k})^2
```

The optimization target is:

```text
min_{m_g} sum_g R_g(m_g)
subject to sum_g c_g(m_g) <= B
```

Meaning:

- choose which groups to protect
- choose how strongly to protect them
- minimize total expected risk under a cost budget `B`

The incremental protection benefit is:

```text
Benefit_g(m0 -> m1) = R_g(m0) - R_g(m1)
```

A simple ranking rule is:

```text
Benefit per cost = Benefit_g / (c_g(m1) - c_g(m0))
```

### 13.1 Current Analytic ECC Mode Model

The repository now includes an explicit analytic ECC model for the simplified case:

- codeword size `k`
- correction capability `t`
- i.i.d. Bernoulli raw BER `p`

Let the raw number of errors in one codeword be:

```text
X ~ Binomial(k, p)
```

Then the uncorrectable block probability is:

```text
UBER(k, t, p) = Pr[X > t]
```

and the residual bit error rate is defined as:

```text
RBER(k, t, p) = (1 / k) * sum_{i=t+1}^k i * Pr[X = i]
```

This is now implemented in `utils/ecc_residual.py`.

The current `scripts/rtn_int8_group_budget.py` entry point supports:

- `--mode name:cost:scale_or_profile`
- `--ecc_mode name:k:t`
- `--ecc_mode name:cost:k:t`

For the BCH-style simplification now used in the code, if `k` is a power of two:

```text
parity(k, t) = (log2(k) + 1) * t
```

and `--ecc_mode name:k:t` will infer cost from that parity rule automatically.

Current status:

- the residual BER model is no longer missing at the placeholder level
- the allocation pipeline can now consume analytic `(k, t)` modes directly
- what still remains is a more realistic hardware table beyond the i.i.d. Bernoulli assumption

Current status:

- this is the target methodology
- it is not yet implemented in the repo

## 14. Why Layer MSE Alone Was Not Enough

The original concern was that:

- bits with large `hessian * scale magnitude` seem important
- but it is hard to convert that into a quantitative protection budget
- trying to use layer MSE as a bridge to accuracy drop did not work well

The current experiments support the following interpretation:

### What layer MSE and layer NMSE can do

- they can capture some monotonic relation with task damage under large structured perturbations
- they work better as a stress-test diagnostic than as a precise single-bit predictor

### What they miss

- sign of the final task perturbation
- bit-position asymmetry
- rare catastrophic events from high bits
- downstream amplification that depends on which decoder layer was hit

This is exactly what the experiments show:

- `single_flip` local score predicts damage magnitude weakly but meaningfully
- `layer_full_lsb` reveals strong layer-depth dependence

## 15. Implemented Methodology So Far

### 15.1 Single-Flip Study

Implemented in:

- `scripts/bitflip_hypothesis_experiment.py`

Setting:

- 4-bit RTN asymmetric per-channel
- one sampled bit flip at one sampled weight

Per-trial fields include:

- `layer_name`
- `row`
- `col`
- `bit`
- `q_old`
- `q_new`
- `delta_q`
- `scale`
- `hessian_diag`
- `score`
- `delta_loss`
- `abs_delta_loss`

Current score in that script:

```text
score = h * (delta_w)^2
```

This is the same local idea as above, but without explicitly multiplying by BER because BER is fixed across trials for ranking purposes.

### 15.2 Layer Full-LSB Stress Test

Implemented in:

- `scripts/bitflip_hypothesis_experiment.py`

Setting:

- 4-bit RTN asymmetric per-channel
- for one decoder layer at a time, flip LSB of all weights in all linear modules inside that decoder layer

Measured outputs:

- `delta_loss`
- `delta_ppl`
- decoder output `mse`
- decoder output `nmse`

This experiment is not a realistic BER model.

It is a stress test used to reveal:

- which decoder layers are intrinsically more fragile
- whether decoder output distortion correlates with end-task damage

### 15.3 Int8 Asymmetric RTN Broad Risk Analysis

Implemented in:

- `scripts/rtn_int8_asym_bitflip_risk.py`

Setting:

- 8-bit RTN asymmetric per-channel
- BER specified explicitly
- no actual task evaluation in this script
- purely local second-order risk ranking

Outputs:

- module risk ranking
- bit-position risk breakdown
- top weight hotspots
- top bit hotspots

This script is intended to answer:

- where should protection go first
- which bit positions matter most

### 15.4 Decoder-Layer Multiplier Estimation

Implemented in:

- `scripts/estimate_decoder_layer_multiplier.py`

Inputs:

- one module-level local-risk CSV
- one decoder-layer damage CSV
- one chosen task metric such as `delta_ppl`
- one damage transform such as `abs`

Outputs:

- aggregated decoder-layer local risk CSV
- decoder-layer multiplier CSV
- JSON summary
- Markdown summary

The intended use is:

- estimate which decoder layers are under- or over-estimated by the local risk model
- construct a provisional layer-aware score

```text
score_final(r, c, k) = alpha_rel(layer(r,c)) * score_local(r, c, k)
```

## 16. Current Experimental Results

### 16.1 Single-Flip Result: Magnitude Ranking Works Better Than Sign Prediction

Source:

- `cache/bitflip_hypothesis/20260309_042829_summary.json`

Baseline:

- `baseline_loss = 4.02935791015625`
- `baseline_ppl = 56.22479828254055`
- repeated baseline evaluation had zero variance in this run

Correlation summary:

- `pearson(score, delta_loss) = 0.007835988875427797`
- `spearman(score, delta_loss) = -0.04426458154196918`
- `pearson(score, abs_delta_loss) = 0.11525424826898799`
- `spearman(score, abs_delta_loss) = 0.11055685149266896`

Interpretation:

- the score does not predict whether loss goes up or down
- the score weakly predicts how large the damage magnitude is

Group comparison:

- top-score average absolute delta loss:
  `0.0013742446899414062`
- random average absolute delta loss:
  `3.528594970703125e-05`
- bottom-score average absolute delta loss:
  `3.814697265625e-06`

Approximate ratios:

- top vs random:
  about 39x
- top vs bottom:
  about 360x

Interpretation:

- even if correlation is modest, the ranking is not random
- top-ranked bits are materially more dangerous than random or bottom-ranked ones
- the local quadratic score is useful as a triage mechanism

### 16.2 Layer Full-LSB Result: Decoder Layer Identity Matters a Lot

Source:

- `cache/bitflip_hypothesis/20260309_135619_summary.json`
- `cache/bitflip_hypothesis/20260309_135619_layer_full_lsb_trials.csv`

Global summary:

- `best_delta_ppl_layer = 2`
- `best_delta_ppl = 5312.022979746949`
- `worst_delta_ppl_layer = 9`
- `worst_delta_ppl = 61.545699074797696`
- `mean_delta_ppl = 1801.6442043847044`

Per-layer `delta_ppl`:

```text
layer 0  -> 4167.145224269427
layer 1  -> 4822.678703049046
layer 2  -> 5312.022979746949
layer 3  ->  963.768432257915
layer 4  ->  710.3304223430124
layer 5  -> 1563.2084867484061
layer 6  -> 1224.1488493935265
layer 7  ->  920.5111118668434
layer 8  ->  734.4704326875483
layer 9  ->   61.545699074797696
layer 10 ->  146.7681795973393
layer 11 ->  993.1319315816396
```

Decoder-output distortion correlation:

- `corr(delta_ppl, mean_decoder_mse) = 0.6370773908734768`
- `corr(delta_ppl, mean_decoder_nmse) = 0.6586741158954721`

Interpretation:

- decoder-output distortion is related to task damage
- but the relation is only moderate, not tight enough to be the final allocation target by itself
- which decoder layer is hit is a strong factor

An especially important outcome in this stress test is:

- the simple predictor `target_decoder_layer` alone outperformed several distortion-based engineered feature sets in LOOCV

Interpretation:

- there is a strong layer-depth / downstream-propagation effect
- local MSE or NMSE cannot fully replace that

### 16.3 Int8 Asymmetric RTN Local Risk Result: High Bits Dominate, Hotspots Are Concentrated

Source:

- `cache/rtn_int8_asym_bitflip_risk/20260309_183147_summary.json`
- `cache/rtn_int8_asym_bitflip_risk/20260309_183147_layer_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk/20260309_183147_top_weight_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk/20260309_183147_top_bit_risk.csv`

Setting:

- model: `/raid/LLM/opt-125m`
- quantization: int8 asymmetric RTN per-channel
- Hessian source: quantized model
- BER: `1e-6` for every bit position

Global summary:

- `n_layers = 72` linear modules
- `total_numel = 84,934,656`
- `total_model_risk = 829.3591727190511`

Bit-position breakdown:

```text
bit 7 -> risk fraction 0.7500114442664225
bit 6 -> risk fraction 0.18750286106660563
bit 5 -> risk fraction 0.04687571526665141
bit 4 -> risk fraction 0.011718928816662852
bit 3 -> risk fraction 0.002929732204165713
bit 2 -> risk fraction 0.0007324330510414282
bit 1 -> risk fraction 0.00018310826276035706
bit 0 -> risk fraction 0.000045777065690089265
```

Interpretation:

- bits 7 and 6 account for about 93.75 percent of total surrogate risk
- protecting low bits first would be a poor use of budget under equal BER

Top module:

- `model.decoder.layers.2.self_attn.k_proj`
- `total_risk = 50.79403100603334`

Top weight hotspot:

- `model.decoder.layers.2.fc1`
- row `2768`, col `706`
- `weight_risk = 0.04406983777599884`

Top bit hotspot:

- `model.decoder.layers.2.fc1`
- row `2768`, col `706`, bit `7`
- `delta_w = 0.7384803891181946`
- `bit_risk = 0.033052882678963835`

The top entries are highly concentrated. The first few top-weight rows are:

```text
1  model.decoder.layers.2.fc1          row 2768 col 706
2  model.decoder.layers.2.fc1          row  622 col 706
3  model.decoder.layers.2.self_attn.k_proj row 267 col 119
4  model.decoder.layers.2.fc1          row 1174 col 706
5  model.decoder.layers.2.fc1          row 1115 col 706
6  model.decoder.layers.2.fc1          row 2878 col 706
7  model.decoder.layers.2.self_attn.k_proj row 267 col 706
8  model.decoder.layers.2.fc1          row 1515 col 706
9  model.decoder.layers.2.self_attn.k_proj row 258 col 119
10 model.decoder.layers.2.fc1          row 2417 col 706
```

Interpretation:

- specific columns such as `col 706` are repeatedly selected because the Hessian proxy for those columns is very large
- the corresponding rows also have large scales
- this produces a hotspot structure rather than a uniform spread

### 16.4 Decoder-Layer Aggregate from Int8 Local Risk

Summing module risks within each decoder layer gives:

```text
layer 2  -> 101.40556979052204
layer 4  ->  80.71518634719443
layer 11 ->  77.33040921678597
layer 5  ->  75.18923188430391
layer 10 ->  74.29909776560253
layer 3  ->  72.76691030554645
layer 8  ->  70.53845677061346
layer 9  ->  69.12927236300665
layer 6  ->  68.60402283322716
layer 7  ->  68.5958146761076
layer 1  ->  35.55714763289736
layer 0  ->  35.22805313324357
```

Interpretation:

- layer 2 is clearly high risk in both the local-risk run and the earlier stress test
- layers 0 and 1 are low by local int8 risk but very high by the 4-bit full-LSB stress test

This is a critical observation:

- local second-order risk and final task damage are not the same object
- local risk gives good within-layer and within-module ranking
- decoder-layer amplification still needs a separate correction

### 16.5 Cross-Experiment Comparison

If we directly compare:

- int8 local decoder-layer aggregate risk
- 4-bit full-LSB `delta_ppl`

the correlation is weak and even negative:

```text
pearson approx -0.2753
spearman approx -0.0909
```

This should not be over-interpreted.

It does not mean the local risk score is useless.

It means the two experiments are mismatched:

- different quantization bit-width
- different fault pattern
- different observable
- different granularity

What can still be trusted:

- local score is useful for ranking hotspots
- high bits dominate
- layer identity matters strongly

What cannot yet be trusted:

- a direct one-shot conversion from current local risk to final `delta_ppl`

### 16.6 Matched Int8 Layer Full-LSB Stress Test

Source:

- `cache/bitflip_hypothesis_int8/20260309_185745_summary.json`
- `cache/bitflip_hypothesis_int8/20260309_185745_layer_full_lsb_trials.csv`

This rerun uses:

- 8-bit RTN asymmetric per-channel
- the same model family as the int8 local-risk run
- `layer_full_lsb`, meaning the LSB of all weights is flipped within one decoder layer at a time

Baseline:

- `baseline_loss = 3.759674072265625`
- `baseline_ppl = 42.934430176522746`

Global summary:

- `best_delta_ppl_layer = 7`
- `best_delta_ppl = 0.10232160990580041`
- `worst_delta_ppl_layer = 8`
- `worst_delta_ppl = -0.1216809437894284`
- `mean_delta_ppl = 0.007575773584269048`

Absolute per-layer `delta_ppl` magnitudes:

```text
layer 0  -> 0.03798058091625478
layer 1  -> 0.08918983940799308
layer 2  -> 0.06687502471239526
layer 3  -> 0.041907688400257825
layer 4  -> 0.039325643919227105
layer 5  -> 0.013100549184798638
layer 6  -> 0.01703669200044544
layer 7  -> 0.10232160990580041
layer 8  -> 0.1216809437894284
layer 9  -> 0.01179067416766344
layer 10 -> 0.005241339208112095
layer 11 -> 0.0026204296843417296
```

Interpretation:

- unlike the 4-bit full-LSB stress test, the 8-bit full-LSB perturbation is extremely small
- this is expected because int8 LSB flips have tiny `delta_w`
- therefore this experiment says more about fine-grained layer amplification under tiny perturbations than about catastrophic failure

Decoder-output distortion correlation in this int8 LSB test is weak:

- `corr(delta_ppl, mean_decoder_mse) = 0.011470620395378868`
- `corr(delta_ppl, mean_decoder_nmse) = -0.14475960762319232`

Interpretation:

- in this small-perturbation regime, mean decoder MSE/NMSE is not a good layer-level predictor of `delta_ppl`
- this reinforces the view that a direct `layer MSE -> task damage` bridge is unreliable

### 16.7 Provisional Matched Int8 Decoder-Layer Multiplier

Source:

- `cache/decoder_layer_multiplier/20260309_185850_summary.json`
- `cache/decoder_layer_multiplier/20260309_185850_summary.md`
- `cache/decoder_layer_multiplier/20260309_185850_decoder_layer_multiplier.csv`

This estimate uses:

- local risk: `cache/rtn_int8_asym_bitflip_risk/20260309_183147_layer_risk.csv`
- layer damage: `cache/bitflip_hypothesis_int8/20260309_185745_layer_full_lsb_trials.csv`
- damage metric: `abs(delta_ppl)`

The multiplier reported is:

```text
alpha_rel(ell) = (D_ell / sum_j D_j) / (R_ell / sum_j R_j)
```

where:

- `D_ell = abs(delta_ppl_ell)`
- `R_ell` is the aggregated local decoder-layer risk share

Important:

- `alpha_rel` is dimensionless
- `alpha_rel > 1` means the layer is more damaging than its local risk share suggests
- `alpha_rel < 1` means the layer is less damaging than its local risk share suggests
- because `abs(delta_ppl)` is used, this is a magnitude correction, not a sign predictor

Aggregate comparison:

- `pearson(local risk, abs damage) = -0.1580445872548602`
- `spearman(local risk, abs damage) = -0.26573426573426573`

This weak negative correlation means:

- local decoder-layer risk share alone is not enough to explain even tiny int8 LSB layer stress damage
- a layer multiplier is justified if the goal is system-level ranking

Top underestimated layers by local risk:

```text
layer 1 -> alpha_rel 3.788808027281863
layer 8 -> alpha_rel 2.6056180217692466
layer 7 -> alpha_rel 2.253117704016398
layer 0 -> alpha_rel 1.62849749229186
layer 2 -> alpha_rel 0.9961305961754617
```

Top overestimated layers by local risk:

```text
layer 11 -> alpha_rel 0.05118424905139697
layer 10 -> alpha_rel 0.10655476141341937
layer 9  -> alpha_rel 0.25762664988614165
layer 5  -> alpha_rel 0.26317703368449274
layer 6  -> alpha_rel 0.3751023828644217
```

Interpretation:

- layers 1, 7, and 8 are more fragile than local risk alone would predict
- layers 10 and 11 are less fragile than local risk alone would predict
- layer 2 remains important, but it is no longer an obvious extreme outlier once tiny-int8-LSB damage is used as the target

Recommended provisional usage:

```text
score_final(r, c, k) = alpha_rel(layer(r,c)) * score_local(r, c, k)
```

This should still be treated as provisional because:

- the damage signal comes from full-layer LSB stress, not sparse random BER
- only one seed is used
- only one damage metric is used so far

### 16.8 Layer-Aware Hotspot Ranking with Matched Int8 Multiplier

Source:

- `cache/rtn_int8_asym_bitflip_risk_layeraware/20260309_191702_summary.json`
- `cache/rtn_int8_asym_bitflip_risk_layeraware/20260309_191702_layeraware_layer_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk_layeraware/20260309_191702_layeraware_top_weight_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk_layeraware/20260309_191702_layeraware_top_bit_risk.csv`

This step takes:

- the original local bit score
- the matched int8 `alpha_rel(layer)`

and forms the provisional layer-aware score:

```text
score_final(r, c, k) = alpha_rel(layer(r,c)) * score_local(r, c, k)
```

The original local top module was:

- `model.decoder.layers.2.self_attn.k_proj`

The new layer-aware top module is:

- `model.decoder.layers.8.self_attn.k_proj`

The original local top weight hotspot was:

- `model.decoder.layers.2.fc1`, row `2768`, col `706`

The new layer-aware top weight hotspot is:

- `model.decoder.layers.1.fc1`, row `140`, col `706`

The original local top bit hotspot was:

- `model.decoder.layers.2.fc1`, row `2768`, col `706`, bit `7`

The new layer-aware top bit hotspot is:

- `model.decoder.layers.1.fc1`, row `140`, col `706`, bit `7`

Key numbers from the new summary:

- local top bit risk:
  `0.033052882678963835`
- layer-aware top bit risk:
  `0.08479932860913324`
- local top weight risk:
  `0.04406983777599884`
- layer-aware top weight risk:
  `0.11306404623208714`

Interpretation:

- once the matched layer multiplier is applied, decoder layer 1 becomes much more important than it looked under purely local risk
- decoder layer 8 also rises sharply at the module level because its `alpha_rel` is greater than 1 and its local module risk was already nontrivial
- decoder layer 2 remains important, but it is no longer the dominant target everywhere

The shift is very strong in the top-50 hotspot distribution:

```text
local top-50 bit hotspots:
  layer 2 -> 46
  layer 4 -> 3
  layer 1 -> 1

layer-aware top-50 bit hotspots:
  layer 1 -> 50
```

The same pattern appears for top-50 weight hotspots:

```text
local top-50 weight hotspots:
  layer 2 -> 46
  layer 4 -> 3
  layer 1 -> 1

layer-aware top-50 weight hotspots:
  layer 1 -> 50
```

Interpretation:

- the local risk model alone was heavily biased toward layer 2 because of very large Hessian-proxy columns there
- after multiplying by `alpha_rel`, the system-level ranking strongly favors layer 1 hotspots
- this is exactly the use case for the decoder-layer multiplier: preserve local within-layer ordering, but correct the cross-layer ranking

Important caveat:

- the multiplier currently comes from `abs(delta_ppl)` under a full-layer LSB stress test
- therefore the resulting layer-aware ranking should be treated as a provisional system-level ranking, not a final deployment-calibrated ranking

### 16.9 Matched Int8 Sparse Layer Evaluation

Source:

- `cache/rtn_int8_sparse_layer_eval/20260309_193359_summary.json`
- `cache/rtn_int8_sparse_layer_eval/20260309_193359_layer_sparse_trials.csv`
- `cache/rtn_int8_sparse_layer_eval/20260309_193359_selected_bit_records.csv`
- `cache/rtn_int8_sparse_layer_eval/20260309_193359_applied_sparse_faults.csv`

This run uses a more matched setting than the earlier full-layer-LSB stress test:

- int8 asymmetric RTN per-channel
- local risk computed in the same int8 setting
- for each decoder layer, select the top `64` sparse bit hotspots within that layer
- enforce unique weight locations
- inject those sparse faults and measure actual `delta_loss` and `delta_ppl`

Key observation:

- all selected top-64 sparse faults per layer were `bit 7`

This is consistent with the local risk decomposition and confirms that, under equal BER, the sparse protection problem is effectively dominated by the MSB first.

Layer-level prediction quality is materially better in this matched sparse setting:

- `pearson(predicted_total_bit_risk, delta_loss) = 0.7485280545819484`
- `spearman(predicted_total_bit_risk, delta_loss) = 0.6223776223776224`
- `pearson(predicted_total_bit_risk, delta_ppl) = 0.6894049341547072`
- `spearman(predicted_total_bit_risk, delta_ppl) = 0.6223776223776224`

Most damaging sparse-fault layers by observed `abs(delta_ppl)`:

```text
layer 1 -> predicted 0.9126285983190491, delta_ppl 5391.245314307541
layer 2 -> predicted 1.5627642731106295, delta_ppl 4234.655457147275
layer 3 -> predicted 0.8082122855505496, delta_ppl 329.10741971586555
layer 0 -> predicted 0.05921979330787864, delta_ppl 74.0551603449199
```

Interpretation:

- local risk now tracks actual sparse-fault damage much better than it did in the mismatched comparisons
- but cross-layer ranking is still not fully correct
- layer `1` is more fragile than local risk alone predicts
- layer `0` is also under-estimated by purely local risk
- layers `8`, `9`, and `10` look over-estimated by purely local risk

This is strong evidence that:

- local bit score is useful
- matched sparse evaluation is the right calibration target
- a decoder-layer multiplier is still needed for cross-layer protection allocation

### 16.10 Sparse-Matched Decoder-Layer Multiplier

Source:

- `cache/decoder_layer_multiplier_sparse/20260309_193809_summary.json`
- `cache/decoder_layer_multiplier_sparse/20260309_193809_summary.md`
- `cache/decoder_layer_multiplier_sparse/20260309_193809_decoder_layer_multiplier.csv`

This estimate uses:

- local risk: `cache/rtn_int8_asym_bitflip_risk/20260309_183147_layer_risk.csv`
- damage: `cache/rtn_int8_sparse_layer_eval/20260309_193359_layer_sparse_trials.csv`
- target metric: `abs(delta_ppl)`

Aggregate comparison:

- `pearson(local risk, abs sparse damage) = -0.12147430908406588`
- `spearman(local risk, abs sparse damage) = -0.2517482517482518`

This means:

- decoder-layer aggregate local risk share by itself is still not sufficient
- sparse matched damage remains strongly layer-dependent
- multiplier correction is still justified even after matching the quantization setting

Largest relative multipliers:

```text
layer 1 -> alpha_rel 12.406413224004098
layer 2 -> alpha_rel  3.4169634596251193
layer 3 -> alpha_rel  0.37007334186165347
layer 0 -> alpha_rel  0.17200883221060073
```

Smallest relative multipliers:

```text
layer 10 -> alpha_rel 0.00010690019258722241
layer 8  -> alpha_rel 0.0001186931003514552
layer 9  -> alpha_rel 0.00013355074885897707
layer 11 -> alpha_rel 0.0031044216594388347
```

Interpretation:

- layer `1` is the dominant under-estimated layer in the matched sparse setting
- layer `2` remains strongly important, but no longer dominates once cross-layer amplification is included
- late layers `8` to `10` should receive much less early protection budget than the plain local model would suggest

### 16.11 Sparse-Matched Layer-Aware Ranking and Top-50 Protection Distribution

Source:

- `cache/rtn_int8_asym_bitflip_risk_sparse_layeraware/20260309_193851_summary.json`
- `cache/rtn_int8_asym_bitflip_risk_sparse_layeraware/20260309_193851_layeraware_layer_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk_sparse_layeraware/20260309_193851_layeraware_top_weight_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk_sparse_layeraware/20260309_193851_layeraware_top_bit_risk.csv`

This ranking applies the sparse-matched `alpha_rel(layer)` to the original local bit score:

```text
score_sparse_aware(r, c, k) = alpha_rel_sparse(layer(r,c)) * score_local(r, c, k)
```

Under this updated ranking:

- top layer-aware module:
  `model.decoder.layers.2.self_attn.k_proj`
- top layer-aware weight hotspot:
  `model.decoder.layers.1.fc1`, row `140`, col `706`
- top layer-aware bit hotspot:
  `model.decoder.layers.1.fc1`, row `140`, col `706`, bit `7`

Key values:

- top layer-aware weight risk:
  `0.3702270656714968`
- top layer-aware bit risk:
  `0.27767453623079896`

The top-50 distribution is now extremely concentrated.

Layer distribution:

```text
top-50 layer-aware bit hotspots:
  layer 1 -> 50

top-50 layer-aware weight hotspots:
  layer 1 -> 50
```

Module distribution:

```text
top-50 layer-aware bit hotspots:
  model.decoder.layers.1.fc1 -> 50

top-50 layer-aware weight hotspots:
  model.decoder.layers.1.fc1 -> 50
```

Column distribution inside `model.decoder.layers.1.fc1`:

```text
top-50 layer-aware hotspots:
  col 706 -> 32
  col 355 -> 15
  col 20  ->  3
```

Bit-position distribution:

```text
top-50 layer-aware bit hotspots:
  bit 7 -> 50
```

Interpretation:

- the matched sparse calibration collapses the early protection target onto a very small region
- if the budget is tiny, the best immediate target is no longer "layer 2 broadly"
- it is:
  `decoder layer 1 -> fc1 -> bit 7 -> columns 706, 355, 20`
- layer `2` still matters strongly at the module level and should be the next expansion tier after the first concentrated layer-1 `fc1` protections

This is a much sharper protection prescription than the earlier provisional full-LSB-based ranking.

## 17. What We Can State with High Confidence

### Statement 1

The local score

```text
(1/2) * h_c * p_k * (delta w)^2
```

is a useful magnitude ranking signal.

Evidence:

- top-ranked single-bit flips are much more damaging than random or bottom-ranked ones

### Statement 2

For unsigned asymmetric int8 with equal BER per bit, risk is overwhelmingly concentrated in the top bits.

Evidence:

- bit 7 alone contributes about 75 percent of total surrogate risk
- bits 7 and 6 together contribute about 93.75 percent

### Statement 3

Bit protection should not be allocated uniformly across bits, weights, or modules.

Evidence:

- hotspot concentration in `top_bit_risk.csv` and `top_weight_risk.csv`
- strong variation across modules and decoder layers

### Statement 4

Decoder layer identity adds a large system-level effect beyond local sensitivity.

Evidence:

- full-LSB stress test strongly separates layers 0 to 2 from later layers
- layer index alone predicts stress-test damage better than several decoder distortion summaries

### Statement 5

In the matched int8 sparse setting, local risk becomes a useful layer-level predictor, but still requires cross-layer calibration.

Evidence:

- `pearson(predicted_total_bit_risk, delta_loss) approx 0.7485`
- `pearson(predicted_total_bit_risk, delta_ppl) approx 0.6894`
- layer `1` still outruns layer `2` in actual sparse-fault damage despite lower raw local aggregate

## 18. What Remains Uncertain

### Uncertainty 1

How to map local risk into actual accuracy or PPL drop in a calibrated way.

Reason:

- the matched sparse run is much better aligned, but it still uses one deterministic top-risk selection rule and one seed

### Uncertainty 2

What the correct decoder-layer multiplier should be.

Reason:

- the sparse-matched multiplier is more credible than the earlier full-LSB multiplier
- but it is still based on one seed, one model, one `K=64`, and one damage metric

### Uncertainty 3

How much extra value decoder-output NMSE adds once layer identity is known.

Current evidence suggests:

- some signal exists
- but it is not strong enough yet to displace layer identity as the dominant system-level factor

## 19. Practical Guidance Right Now

If a protection decision had to be made immediately with the current evidence, the safest conclusions are:

### Priority 1

Protect high bits first:

- bit 7 first
- then bit 6
- then bit 5

### Priority 2

For the very first protection budget, focus on:

- `model.decoder.layers.1.fc1`
- within that module, prioritize bit `7`
- inside bit `7`, prioritize columns:
  - `706`
  - `355`
  - `20`

### Priority 3

If the budget extends beyond the first concentrated layer-1 block, expand next to:

- `model.decoder.layers.2.self_attn.k_proj`
- `model.decoder.layers.2.self_attn.q_proj`
- `model.decoder.layers.2.fc1`
- then `model.decoder.layers.1.self_attn.k_proj`
- then `model.decoder.layers.1.self_attn.q_proj`

### Priority 4

If protection budget is extremely small, use the sparse-matched layer-aware hotspot list:

- `cache/rtn_int8_asym_bitflip_risk_sparse_layeraware/20260309_193851_layeraware_top_bit_risk.csv`
- `cache/rtn_int8_asym_bitflip_risk_sparse_layeraware/20260309_193851_layeraware_top_weight_risk.csv`

This is the most direct current list of small-budget candidates.

### Priority 5

Do not spend early budget:

- on low bits
- uniformly across all layers
- on late layers `8`, `9`, and `10`
- based only on plain local aggregate risk without sparse-matched correction

### Compressed Protection List

If the goal is a compact actionable protection order, the current evidence supports:

1. protect `layer 1 / fc1 / bit 7 / col 706`
2. then protect `layer 1 / fc1 / bit 7 / col 355`
3. then protect `layer 1 / fc1 / bit 7 / col 20`
4. then expand to `layer 2 / self_attn.k_proj / bit 7`
5. then expand to `layer 2 / self_attn.q_proj / bit 7`
6. then expand to `layer 2 / fc1 / bit 7`
7. only after that, consider broader layer-1 and layer-2 coverage or bit-6 protection

## 20. Recommended Next Experiments

### Next Experiment A: Repeat Matched Sparse Injection Across Seeds and K

Goal:

- test whether the sparse-matched ranking is stable

Method:

- repeat the matched sparse run across multiple seeds
- vary `faults_per_layer`
- compare top-50 overlap and multiplier stability

This is now the highest priority next step.

### Next Experiment B: Estimate Multipliers Under Random Sparse BER, Not Only Top-Risk Selection

Goal:

- test whether the sparse-matched multiplier transfers beyond deterministic hotspot injection

Method:

- run layer-wise sparse faults in int8 using sampled BER patterns
- compare observed damage `D_ell` against predicted local `R_decoder(ell)`
- fit either:
  - direct ratio multipliers
  - monotone calibration
  - a small regression model using
    - decoder-layer index
    - local risk
    - optional decoder-output distortion

### Next Experiment C: Budget-to-Protection Optimization

Goal:

- turn risk ranking into a real protection policy

Method:

- define protectable groups
  - bit
  - weight block
  - SRAM word
  - ECC codeword
- define cost per mode
- define residual BER per mode
- solve
  - greedy by benefit per cost
  - or knapsack

Status:

- analytic `(k, t)` mode support is implemented
- exact DP allocation for integer costs is implemented
- greedy fallback for non-integer costs is implemented
- the remaining step is to replace placeholder modes with a hardware-relevant ECC family

### Next Experiment D: Margin-Based End Metric

Goal:

- get a tighter bridge to accuracy than raw layer MSE

Method:

- evaluate logit margin variance under fault perturbations
- compare margin crossing probability to actual accuracy drop

## 21. Current To-Do List

### Implement

- repeat matched sparse layer evaluation across seeds and `K`
- estimate decoder-layer multipliers under random sparse BER patterns
- extend analytic ECC mode tables beyond the current BCH-style placeholder
- optional margin-based predictor

### Validate

- repeat experiments across multiple random seeds
- measure top-50 hotspot overlap across seeds and across `faults_per_layer`
- compare FP-collected vs quantized-model-collected Hessian proxy
- test whether top-bit/top-weight hotspots remain stable across seeds

### Analyze

- whether layer `1 fc1` dominance persists across seeds and sparse fault counts
- whether layer `0` under-estimation is caused by downstream amplification or small-sample effects
- whether `fc1` hotspots dominate only in specific layers or across the stack
- how much ranking changes when BER is not uniform across bit positions

## 22. Bottom Line

The current study has already established three important facts:

1. local second-order score is useful for hotspot ranking
2. high bits dominate risk overwhelmingly
3. decoder-layer identity introduces a major downstream amplification effect

It has also established a fourth implementation fact:

4. block exposure, residual risk, sensitive-block classification, and budgeted allocation now work end-to-end with an analytic ECC model

The missing piece is not "whether the local score matters".

The remaining missing piece is:

- how to replace the current simplified Bernoulli + `(k, t)` ECC model with a deployment-relevant hardware characterization

The next phase should therefore not return to plain layer MSE alone.

It should build:

- repeated matched sparse injection
- sparse-calibrated decoder-layer multiplier estimation
- explicit residual-risk-per-cost optimization with realistic ECC mode tables
