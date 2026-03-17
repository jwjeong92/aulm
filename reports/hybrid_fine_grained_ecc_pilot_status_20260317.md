## Hybrid Fine-Grained ECC Pilot Status

Date: 2026-03-17

### Implemented

- Added an outer-ECC axis to the adaptive unary pipeline:
  - `none`
  - `b0_proxy_bch`
  - `hsiao_product64`
  - `short_bch_product64`
- Added codec-only evaluation for phase-1 selection.
- Added model-level ECC statistics:
  - `mean_residual_bit_count`
  - `p_residual_gt_1`
  - `p_residual_gt_2`
  - `p_residual_gt_4`
  - `miscorrection_rate`
  - `detected_failure_rate`
  - `silent_corruption_rate`
  - `mean_abs_weight_error`
- Added a pilot runner script:
  - `adaptive_unary_level_mapping/scripts/run_hybrid_ecc_pilot.sh`

### Phase 1

- `P1 = hsiao_product64`, `detect_first`
- `P2 = short_bch_product64`, `detect_first`
- At `BER=1e-2`, `P1` and `P2` tied on `p_residual_gt_2 = 2.5e-05`.
- Both `P1` and `P2` had `miscorrection_rate = 0` under `detect_first`.
- `always_correct` reduced average residual but increased miscorrection and worsened the tail metric.
- The selected winner was `P1` because it matched `P2` on the tail metric while using lower overhead.

Interpretation:

- `P1` and `P2` differ only in the row code inside the same product-code geometry.
  - `P1` uses a lighter SECDED-style Hsiao row code.
  - `P2` uses a stronger short-BCH-like row code intended for `t=2`.
- `detect_first` and `always_correct` differ only in decode policy.
  - `detect_first` only applies a row correction when the stripe check supports it.
  - `always_correct` applies row corrections aggressively even when the stripe evidence is ambiguous.
- In this pilot, `detect_first` was the correct choice because the objective emphasized tail suppression and low miscorrection rather than the lowest average residual.

### Phase 2

- `B0` remained very strong under the current proxy.
- The only passing point in the full pilot matrix was unary-only:
  - `U0`, `u=4`, `selective_pct=100`
- `P*` alone was not competitive.
- `H*` did not satisfy the full acceptance criteria, but it did produce meaningful mid-overhead points.

Representative results:

- `BER=3e-3`
  - `H* u=3 p=10`: better than nearby unary-only points in the same overhead band
  - Still not enough to beat the `B0` proxy
- `BER=1e-2`
  - `H* u=3 p=20`: better PIQA than nearby unary-only points with much lower residual tail
  - High-overhead unary-only still reached higher absolute accuracy

### Current Interpretation

- The hybrid idea is promising.
- The strongest signal so far is not "hybrid wins outright", but rather:
  - hybrid can create useful medium-overhead protection points
  - hybrid can beat unary-only at some nearby overheads
- The current `B0` proxy likely makes the acceptance target too strict.

What this means in practice:

- `P*` alone was not enough to replace the `B0` proxy.
- `H*` did beat unary-only at several nearby overhead points, especially around `u=3` and `selective_pct=10 to 20`.
- The right claim from the current data is therefore:
  - fine-grained ECC plus selective unary can be more protection-efficient than unary-only
  - but the current pilot does not show that it beats the present `B0` proxy

### Important Caveat

- `p_residual_gt_*` is computed on encoded values before unary decode.
- `mean_abs_weight_error` is computed after decode in quantized-value space.
- This mismatch explains why better residual tails did not always translate into better decoded error.

Root cause of the mismatch:

- Phase 1 compares backends under the same `int_binary` geometry, so encoded residual tails are a usable surrogate there.
- Phase 2 compares different inner representations with different codeword lengths and different unary decode geometry.
- After that point, residual bit count alone is not enough; where the residual lands matters as much as how many bits remain wrong.
- This is why some `H*` points had much lower `p_residual_gt_2` than nearby unary-only points but did not improve `mean_abs_weight_error` by the same margin.

### Open Items

- `B0 + u=5` was not measured yet.
  - A targeted sweep was attempted.
  - It failed immediately due to GPU out-of-memory because all A100s were already occupied.
  - Based on current overhead trends, the first useful probe region is roughly `selective_pct = 35% to 40%`.
- The current product-code sweep is narrow:
  - mostly `component_bits = 64`
  - `stripe_words = 8`
  - `128-bit` row width only as a limited ablation
- Model-level `P2` was not run in phase 2.
  - The current phase-2 `P*` result is only for the phase-1 winner `P1`.
  - A direct `P1` vs `P2` comparison in phase 2 still remains open if needed.
- The current `short_bch_product64` implementation is still effectively a double-error-correcting proxy.
  - The CLI exposes `--outer_ecc_t`
  - But the row-code construction currently guarantees double-error-correcting behavior, not a full `t=3` or `t=4` BCH decoder

### Recommended Next Steps

- Re-run the baseline comparison with a less optimistic or more realistic `B0` proxy.
- Probe `B0 + u=5` once GPU memory is available again.
- Sweep product-code geometry before drawing broad conclusions:
  - `stripe_words in {4, 8, 16}`
  - `component_bits in {64, 128}`
- Treat `u=3`, `selective_pct=10 to 20` as the main promising hybrid region from this pilot.
