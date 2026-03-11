# TODO

## Now

- [x] Implement a first `surrogate-guided search` prototype in `adaptive_unary_level_mapping/`
- [ ] Keep the existing greedy empirical path only for finalist verification, local refinement, and fallback
- [ ] Resume the frozen Phase 4 matrix only after the faster search path is ready

## Search Prototype Scope

- [x] Define a fast surrogate score using existing group importance plus a level-dependent distortion term
- [x] Generate a small finalist set from the surrogate before running full `lm_eval`
- [x] Preserve the current boundary-constrained level policy format unless a contradiction is found
- [x] Keep `step_pct` and `rollback_steps` as optional local-refinement controls, not the primary global explorer
- [x] Record surrogate rankings and finalist candidates in run artifacts

## Surrogate Sanity Checks

- [x] Check group-importance distribution shape and whether it is sufficiently heavy-tailed
- [ ] Check how `damage_ratio` changes over `u_bits` and `u_bits / W`
- [ ] Check how surrogate risk, overhead, and objective change across `lambda`
- [ ] Check whether adjacent `lambda` values produce distinct policies or redundant repeats
- [ ] Decide whether the current `surrogate_lambdas` grid should be tightened, widened, or replaced by breakpoint-driven candidates

## Phase 4 Evidence

- [ ] Run `E1`: `6-bit` pilot fixed baselines on `OPT-125M`
- [ ] Run `E2`: `6-bit` pilot adaptive search on `OPT-125M`
- [ ] Run `E3`: int8 confirmatory fixed baselines on `OPT-6.7B`
- [ ] Run `E4`: int8 confirmatory adaptive search on `OPT-6.7B`
- [ ] Run `E5`: system projection table assembly
- [ ] Run `E6`: ECC system comparison table assembly

## Required Outputs

- [ ] Confirmatory accuracy-vs-BER results for binary, fixed mixed levels, and adaptive unary
- [ ] Best adaptive policy at `BER = 1e-2`
- [ ] Encoded-width overhead for each candidate policy and the winner
- [ ] Pass/fail margin against the `<= 5%` drop target for each task and BER
- [ ] Robustness-overhead sweet-spot plot
- [ ] Retry/reclaim interpretation using the best confirmatory result

## Paper/Artifact Follow-Up

- [ ] Draft `F1-F3` conceptual figures that do not depend on unfinished experiments
- [ ] Do not change `paper_outline.md`, `paper_method.md`, or `paper_experiment_matrix.md` unless a contradiction is found
- [ ] Update session bridge documents after each meaningful design or execution change

## Secondary Track

- [ ] Repeat matched sparse injection across seeds and `faults_per_layer`
- [ ] Estimate decoder-layer multipliers under random sparse BER patterns
- [ ] Replace placeholder ECC mode tables with hardware-relevant modes
- [ ] Optionally test a margin-based end metric
