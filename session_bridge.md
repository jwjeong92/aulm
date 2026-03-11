# Session Bridge

## Workspace

- Repository: `/home/jwjeong/study/quant_analysis`
- Current focus: quantized LLM weight bit-flip robustness and protection allocation

## What The User Actually Wants

The user does **not** want a workflow centered on repeated fault injection, seed sweeps, or small-model empirical calibration loops.

The real target is:

1. define bit-flip sensitivity at the level of `ECC codeword` or `quantization group`
2. connect that sensitivity to `ECC protection mode`
3. assign protection under a fixed cost budget such as parity length
4. keep the method intuitive, hypothesis-driven, and logically defensible
5. make it realistic for very large models such as `175B`

## Direction Shift

Earlier work in this repo emphasized:

- bit-level hotspot ranking
- weight-level hotspot ranking
- decoder-layer multipliers estimated from sparse fault experiments

That was useful for exploration, but it is **not** the right end goal for the user.

The new direction is:

- analytic block sensitivity
- ECC/codeword or quantization-group aggregation
- residual-risk-per-cost optimization

## Main Design Principle

The user wants an approach that can be justified without large-scale repeated injection experiments.

The intended structure is:

1. compute a local exposure measure from quantization scale and activation sensitivity proxy
2. aggregate that exposure into `quantization groups` and `ECC codewords`
3. define `ECC mode -> residual BER` tables
4. compute residual risk for each block under each mode
5. solve a budgeted allocation problem

## Current Status Update

The core analytic structure is now implemented.

Completed pieces:

- block-level bit exposure vectors `E_b(k)` are emitted by
  `/home/jwjeong/study/quant_analysis/scripts/rtn_int8_group_budget.py`
- mode-wise residual risks `R_b(m)` are computed from those exposure vectors
- sensitive-block classification (`S/A/B`) is emitted in the candidate CSV
- budget allocation supports exact DP for integer costs and greedy fallback otherwise
- analytic ECC residual BER support is implemented in
  `/home/jwjeong/study/quant_analysis/utils/ecc_residual.py`

The ECC model currently supported is:

- i.i.d. Bernoulli raw BER per bit
- block size `k`
- correction capability `t`
- uncorrectable probability:
  `UBER = Pr[Binomial(k, p) > t]`
- residual BER:
  expected residual erroneous bits per codeword bit after decoding

For BCH-style cost modeling, the current assumption is:

- if `k` is a power of two, parity cost can be inferred as
  `parity = (log2(k) + 1) * t`

The current CLI supports both:

- `--ecc_mode name:k:t`
  -> cost inferred by the BCH-style parity rule
- `--ecc_mode name:cost:k:t`
  -> cost overridden manually

## Current Useful Formula Direction

For block `b`, bit position `k`, and ECC mode `m`, the desired structure is:

```text
E_b(k) = sensitivity exposure of block b to bit k
R_b(m) = sum_k E_b(k) * p_k^(m)
```

where:

- `E_b(k)` is block exposure
- `p_k^(m)` is residual BER of bit `k` after applying ECC mode `m`
- `R_b(m)` is residual risk

This separates:

- model-side sensitivity
- hardware-side protection performance

which is exactly the abstraction the user wants.

## Existing Code To Reuse

- `/home/jwjeong/study/quant_analysis/utils/bitflip_risk.py`
- `/home/jwjeong/study/quant_analysis/utils/ecc_residual.py`
- `/home/jwjeong/study/quant_analysis/scripts/rtn_int8_asym_bitflip_risk.py`
- `/home/jwjeong/study/quant_analysis/scripts/rtn_int8_sparse_layer_eval.py`
- `/home/jwjeong/study/quant_analysis/scripts/rtn_int8_group_budget.py`
- `/home/jwjeong/study/quant_analysis/reports/bitflip_methodology_and_results_20260309.md`

## Important Existing Results

Sparse-layer and layer-multiplier experiments were run, but they should now be treated as secondary reference rather than the core method.

Still useful references:

- sparse layer evaluation:
  `/home/jwjeong/study/quant_analysis/cache/rtn_int8_sparse_layer_eval/20260309_193359_summary.json`
- sparse-matched decoder-layer multiplier:
  `/home/jwjeong/study/quant_analysis/cache/decoder_layer_multiplier_sparse/20260309_193809_decoder_layer_multiplier.csv`
- sparse-matched layer-aware ranking:
  `/home/jwjeong/study/quant_analysis/cache/rtn_int8_asym_bitflip_risk_sparse_layeraware/20260309_193851_summary.json`

## New Group/Codeword-Level Outputs

These are more aligned with the user's target:

- 32-weight row-group candidates:
  `/home/jwjeong/study/quant_analysis/cache/rtn_int8_group_budget_qgroup32/20260309_201320_summary.json`
- 64-bit codeword candidates:
  `/home/jwjeong/study/quant_analysis/cache/rtn_int8_group_budget_codeword64/20260309_201354_summary.json`
- 128-bit codeword candidates:
  `/home/jwjeong/study/quant_analysis/cache/rtn_int8_group_budget_codeword128/20260309_203225_summary.json`
- 256-bit codeword candidates:
  `/home/jwjeong/study/quant_analysis/cache/rtn_int8_group_budget_codeword256/20260309_203221_summary.json`
- 64-bit codeword with placeholder analytic modes and allocation:
  `/home/jwjeong/study/quant_analysis/cache/rtn_int8_group_budget_codeword64_proposal/20260310_023615_summary.json`
- 64-bit codeword with explicit analytic ECC mode `(k=64, t=1, cost=10)`:
  `/home/jwjeong/study/quant_analysis/cache/rtn_int8_group_budget_codeword64_eccanalytic/20260310_140734_summary.json`
- 64-bit codeword with BCH-style inferred parity cost:
  `/home/jwjeong/study/quant_analysis/cache/rtn_int8_group_budget_codeword64_bchauto/20260310_141833_summary.json`

## High-Level Observation From Current Group Results

Under simple parity-cost assumptions, finer-grained codewords can capture more surrogate risk per parity bit than coarser blocks.

This is no longer missing at the placeholder level.

What is now available:

- explicit analytic `ECC mode -> residual BER / UBER` modeling under the
  Bernoulli + `(k, t)` assumption
- BCH-style parity-cost inference
- end-to-end allocation runs using those analytic ECC modes

What is still missing for a more realistic hardware model:

- correlated or burst fault processes
- miscorrection modeling
- non-uniform codeword packing or lane-aware BER asymmetry

## What The Next Session Should Do

1. treat the analytic optimization pipeline as the primary method
2. define a realistic ECC mode set beyond the current placeholder `(k, t)` examples
3. decide whether the objective should use:
   - residual BER
   - UBER
   - or another residual-fault metric derived from the decoder model
4. refine packing assumptions between quantization groups and ECC codewords
5. keep the method valid for very large models such as `175B`

## Non-Negotiable Requirements

- no training-level computation
- no dependence on many injection trials
- intuitive and logically clean
- explainable to both model and hardware audiences
- directly tied to `ECC codeword` and `protection budget`
