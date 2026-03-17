# Hybrid Fine-Grained ECC Pilot Plan

Generated on 2026-03-17 to preserve the current implementation plan for follow-up sessions.

## Summary

Implement a single-model pilot that compares `B0` current-system ECC proxy, unary-only, fine-grained product ECC, and hybrid ECC+unary on `OPT-125M + PIQA`.

For this pilot, model `B0` as the existing `int_bch(8192,50)` path as a temporary proxy. The goal is to select one low-cost fine-grained ECC that reduces `post-ECC residual bit count` and `miscorrection` enough that a lower unary level (`u=2` or `u=3`) beats both `B0` and unary-only on the accuracy-overhead frontier.

## Key Changes

- Split the current representation choice into two orthogonal axes:
  - Inner representation stays `int_binary`, `int_unary`, or `mixed`.
  - New outer ECC axis becomes `none`, `b0_proxy_bch`, `hsiao_product64`, or `short_bch_product64`.
- Extend the CLI/config interface in `adaptive_unary_level_mapping/main.py` with:
  - `--outer_ecc`
  - `--outer_ecc_component_bits` default `64`
  - `--outer_ecc_stripe_words` default `8`
  - `--outer_ecc_t` default `2` for `short_bch_product64`
  - `--outer_ecc_policy {detect_first,always_correct}` default `detect_first`
- Keep legacy `repr_type=int_bch` working unchanged for backward compatibility, but do not use it in the new hybrid matrix except as the `B0` proxy backend.
- Add a fine-grained ECC simulation layer before unary decoding in `adaptive_unary_level_mapping/lib/error_injection.py`:
  - `b0_proxy_bch`: reuse current codeword-level BCH behavior unchanged.
  - `hsiao_product64`: row component is `Hsiao SECDED (72,64)` on each 64-bit word, column component is single-parity-check across `8` words, decode order is row then column consistency check.
  - `short_bch_product64`: row component is repo-analytic short-BCH-like `64 payload + 14 parity, t=2`, column component is single-parity-check across `8` words, decode order matches `hsiao_product64`.
  - `detect_first` means: if row correction is not uniquely supported by the stripe check, do not apply the correction; emit a detected-failure/erasure event and leave the bits unmodified.
  - `always_correct` means: apply row correction whenever the row decoder claims a correction, regardless of stripe consistency.
- Extend evaluation in `adaptive_unary_level_mapping/lib/search.py` so every run records ECC intermediate statistics alongside existing task metrics:
  - `mean_residual_bit_count`
  - `p_residual_gt_1`
  - `p_residual_gt_2`
  - `p_residual_gt_4`
  - `miscorrection_rate`
  - `detected_failure_rate`
  - `silent_corruption_rate`
  - `mean_abs_weight_error` after unary decode
- Add one codec-only evaluation entrypoint that runs the outer ECC without loading a model, using Bernoulli bit flips over synthetic stripes. Output should be a compact JSON/CSV summary with the same residual-count and miscorrection statistics as the model path.

## Pilot Matrix

Use `OPT-125M`, `RTN6`, `PIQA`, `limit=200`, `seed list = [0,1,2,3,4]` for all model-level runs except if a backend is too slow, in which case first stage stays codec-only and only the chosen winner reaches model evaluation.

Run in two phases.

### Phase 1: Codec-only selection

- BER grid: `1e-3`, `3e-3`, `1e-2`
- Compare:
  - `B0`: `b0_proxy_bch`
  - `P1`: `hsiao_product64`, `detect_first`
  - `P1-ablation`: `hsiao_product64`, `always_correct`
  - `P2`: `short_bch_product64`, `detect_first`
  - `P2-ablation`: `short_bch_product64`, `always_correct`
- One sensitivity ablation only for the product codes:
  - `component_bits = 128`
  - `stripe_words = 8`
- Selection rule:
  - Choose the product ECC that has lower `p_residual_gt_2` than the other product candidate at `BER=1e-2`
  - Reject any candidate whose `miscorrection_rate` is higher than its `detected_failure_rate`
  - Prefer the lower-overhead candidate if the residual-tail difference is within 10% relative

### Phase 2: Model-level frontier

- Compare:
  - `B0`: `outer_ecc=b0_proxy_bch`, `repr_type=int_binary`
  - `U0`: `outer_ecc=none`, `repr_type=mixed`, `u_bits in {2,3,4}`, `selective_pct in {20,40,100}`
  - `P*`: chosen product ECC from Phase 1, `repr_type=int_binary`
  - `H*`: chosen product ECC + `repr_type=mixed`, `u_bits in {2,3}`, `selective_pct in {10,20,40}`
- Keep `outer_ecc_policy=detect_first` for all model-level runs, plus one ablation at `BER=1e-2` with `always_correct`.
- Acceptance criteria:
  - `H*` must beat both `B0` and the best `U0` point on the `overhead vs PIQA accuracy` frontier at either `BER=3e-3` or `BER=1e-2`
  - `H*` must show lower `mean_abs_weight_error` than `U0` at the same or lower total overhead
  - `H*` must show lower `p_residual_gt_2` than `B0` at the same or lower total overhead

## Test Plan

- Unit tests for each outer ECC backend:
  - Row encode/decode round-trip on error-free input
  - Guaranteed-correctable patterns are corrected
  - Over-capacity patterns are flagged under `detect_first`
  - `always_correct` and `detect_first` diverge on ambiguous multi-error patterns
- Integration tests:
  - Existing runs with `outer_ecc=none` and legacy `repr_type=int_bch` produce unchanged metrics schema and unchanged overhead behavior
  - Hybrid path applies outer ECC before unary decode and records the new ECC stats
- Smoke experiments:
  - Codec-only JSON for all Phase 1 points
  - One model-level run each for `B0`, best `U0`, `P*`, and one `H*` point
- Plot outputs required for the pilot:
  - Accuracy-overhead frontier
  - Residual-count CDF after ECC
  - `|Delta w|` histogram after unary decode

## Assumptions And Defaults

- `B0` is intentionally a temporary `int_bch(8192,50)` proxy for the current-system ECC in this pilot; replacing it with the true baseline ECC is deferred until after the pilot selects the fine-grained candidate.
- `OPT-125M + PIQA` is the only model/task pair in scope for the first implementation pass.
- The default fine-grained product candidate is `64-bit component + 8-word stripe`; `128-bit` is ablation-only.
- `detect_first` is the default decode policy because the hybrid objective values lower residual-count tails more than aggressive correction with silent miscorrection.
- Burst-error experiments are out of scope for the first pass; the first pass uses Bernoulli bit flips only.
