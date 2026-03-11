# Paper Experiment Matrix

Date: 2026-03-11
Workspace: `/home/jwjeong/study/quant_analysis`
Status: Phase 3 deliverable

## Purpose

This document freezes the exact experiment matrix needed to validate the paper
claims and populate the required figures and tables listed in
`paper_method.md`.

It has four jobs:

1. map each paper claim to concrete evidence
2. freeze the exact model / BER / baseline / metric / pass-fail protocol
3. separate pilot experiments from paper-critical confirmation experiments
4. define which output files will feed each paper figure and table

## Current Execution Status

The matrix in this file is still valid, but execution is temporarily paused.

Reason:

- the current global empirical greedy search is too slow to be the default
  design-space exploration path
- the project will first redesign the search path and then resume this matrix

Current rule:

- do not change the matrix unless a contradiction is found
- do not launch the full `E1-E6` sequence until the faster search path is ready

## Phase 3 Decisions

The paper will use three evidence tracks.

### Track A. `6-bit` Pilot / Ablation Matrix

This track is for:

- pipeline validation
- trend checking
- candidate-level screening
- qualitative comparison against more aggressive unary expansion

Frozen settings:

- quantization path:
  `RTN`, `w_bits = 6`, `w_groupsize = 128`, `w_asym = true`
- calibration setup:
  `nsamples = 128`, `cal_dataset = wikitext2`, `act_order = false`
- importance metric:
  `gptq_hessian`
- task set:
  `piqa`, `hellaswag`, `arc_easy`
- evaluation limit:
  `lm_eval_limit = 200`
- BER points:
  `1e-4`, `3e-4`, `1e-3`, `3e-3`, `1e-2`
- bit-error seeds:
  `0, 1, 2, 3, 4`
- accuracy-drop tolerance:
  `5%`

Search protocol note:

- `step_pct` and `rollback_steps` define the actual greedy boundary search
- they do not mean the implementation is directly minimizing the analytic risk
  `R_g(u, p)`

Mandatory pilot model:

- `OPT-125M`

Optional pilot replication:

- `OPT-1.3B`

Decision note:

- the current unary search path, hardware proxy table, and repository examples
  are most mature at `w_bits = 6`
- earlier quick experiments moved from `4-bit RTN` to `6-bit RTN` because the
  baseline quality at `4-bit` was too weak for meaningful robustness sweeps
- therefore `6-bit` remains useful as a pilot setting
- but it is not the final paper-faithful quantization setting

### Track B. `int8` Confirmatory Matrix

This is the paper-critical empirical track.

Frozen settings:

- quantization path:
  `RTN`, `w_bits = 8`, `w_groupsize = 128`, `w_asym = true`
- calibration setup:
  `nsamples = 128`, `cal_dataset = wikitext2`, `act_order = false`
- importance metric:
  `gptq_hessian`
- task set:
  `piqa`, `hellaswag`, `arc_easy`
- evaluation limit:
  `lm_eval_limit = 200`
- BER points:
  `1e-4`, `3e-4`, `1e-3`, `3e-3`, `1e-2`
- bit-error seeds:
  `0, 1, 2, 3, 4`
- accuracy-drop tolerance:
  `5%`

Search protocol note:

- the confirmatory track uses the same empirical greedy boundary search style
- analytic risk remains a supporting model, not the directly optimized search
  objective

Mandatory confirmatory model:

- `OPT-6.7B`

Optional confirmatory replication:

- `LLaMA2-7B`

Decision note:

- the paper story is anchored in int8 large-model flash inference
- therefore at least one `7B` int8 model must carry the main paper claim
- the confirmatory baseline family is intentionally smaller than the pilot
  family
- `full unary` is not a mandatory confirmatory baseline at `int8` because
  `m(8) = 255` creates an extreme width penalty and moves the implementation
  into the packed unary path; it should be treated as an analytic extreme or a
  limited comparison point, not as the default required run

### Track C. System Projection and Analytic Comparison

This is the system-supporting track.

- model-size points:
  int8 `7B`, `30B`, `70B`, `100B`
- use:
  capacity, hot-block, reclaim, retry, and ECC-cost interpretation
- no requirement:
  Track C does not require end-to-end `lm_eval` BER sweeps in Phase 4

ECC strong-reference points in Track C:

- `BCH(1KiB, k=8192)` with `10t`, `50t`, `64t`

Decision note:

- strong ECC remains a supporting analytic / system comparison point
- it is not part of the core `lm_eval` baseline family unless a direct
  residual-BER injection path is added later

## Claim-to-Evidence Map

### Claim C1

Statement:

- selective / adaptive unary can meet the `<= 5%` accuracy-drop target at
  `BER = 1e-2`

Required evidence:

- Track B fixed-baseline sweeps
- Track B adaptive-search final verification
- per-task pass/fail summary at `BER = 1e-2`

Track A role:

- supportive only

### Claim C2

Statement:

- unary-based policies offer a better robustness-overhead tradeoff than pure
  binary

Required evidence:

- Track B accuracy-vs-BER curves
- Track B overhead ratios
- `BER = 1e-2` tradeoff comparison across the frozen confirmatory baseline family

Track A role:

- supports level-screening intuition

### Claim C3

Statement:

- a practical sweet spot exists between low-overhead binary and heavier unary
  expansion

Required evidence:

- Track B comparison among `binary`, selected fixed mixed levels, and `adaptive`
- Track A pilot comparison against more aggressive fixed unary baselines
- explicit overhead accounting

### Claim C4

Statement:

- robustness gains are large enough to support the retry / reclaim mitigation
  interpretation

Required evidence:

- best Track B policy and its robustness margin
- Track C retry / reclaim interpretation using the locked system assumptions
- explicit wording that this is a modeled system interpretation, not a
  controller-level closed-loop simulation

## Common Protocol

These settings are shared by Tracks A and B unless explicitly overridden.

### Tasks

- `piqa`
- `hellaswag`
- `arc_easy`

Reason for freezing this set:

- already supported by the current pipeline examples
- covers multiple task styles
- keeps the paper away from a single-benchmark story

### BER Sweep

- `1e-4`
- `3e-4`
- `1e-3`
- `3e-3`
- `1e-2`

Reason for freezing this set:

- matches current configs and code paths
- includes the paper-critical endpoint `1e-2`
- provides enough intermediate points for interpretable robustness curves

### Seed Protocol

- search stage candidate seed list:
  `search_seed_mode = single`, `search_seed = 0`
- final verification seed list:
  `beseed_list = [0, 1, 2, 3, 4]`
- `final_verify_full_seeds = true`

### Pass / Fail Rule

For task `t` and BER `beta`, let:

$$
\mathrm{drop}(t, \beta) :=
1 - \frac{\bar A_t(\beta)}{A_t^{base}}
$$

The policy passes iff:

$$
\bar A_t(\beta) \ge 0.95 \cdot A_t^{base}
\qquad
\text{for all tasks } t \text{ and BER points } \beta
$$

Paper selection rule:

1. prefer passing policies over failing policies
2. among passing policies, prefer smaller overhead ratio `Omega`
3. break ties by larger minimum accuracy margin

Fallback reporting rule if no policy passes:

1. report that the target was not met
2. show the best failing policy by smallest violation at `BER = 1e-2`
3. do not rewrite the claim as success

## Track-Specific Protocol

### Track A. `6-bit` Pilot

- `quant_mode = rtn`
- `w_bits = 6`
- `w_groupsize = 128`
- `w_asym = true`
- `w_clip = false`
- `nsamples = 128`
- `cal_dataset = wikitext2`
- `percdamp = 0.01`
- `act_order = false`
- adaptive level set:
  `0, 2, 3, 4, 5, 6`
- search:
  `step_pct = 5.0`, `rollback_steps = 1`

Actual algorithm summary:

- rank groups by `I_g`
- sweep boundary counts greedily
- verify candidates by BER injection and `lm_eval`

### Track B. `int8` Confirmatory

- `quant_mode = rtn`
- `w_bits = 8`
- `w_groupsize = 128`
- `w_asym = true`
- `w_clip = false`
- `nsamples = 128`
- `cal_dataset = wikitext2`
- `percdamp = 0.01`
- `act_order = false`
- adaptive level set:
  `0, 3, 4, 5`
- search:
  `step_pct = 5.0`, `rollback_steps = 1`

Actual algorithm summary:

- rank groups by `I_g`
- sweep boundary counts greedily
- verify candidates by BER injection and `lm_eval`

Reason for the reduced int8 level set:

- `u = 3` and `u = 4` are the main fixed comparison points
- `u = 5` allows a heavier adaptive option without forcing the `255b`
  full-unary endpoint
- the confirmatory matrix should stay focused on practically defensible points

## Baseline Family

### Track A Pilot Baselines

| ID | Family | Settings | Role |
| --- | --- | --- | --- |
| P0 | Binary | `repr_type = int_binary` | pilot reference baseline |
| P1 | Fixed mixed | `repr_type = mixed`, `u_bits = 2`, `selective_pct = 100` | low-overhead pilot point |
| P2 | Fixed mixed | `repr_type = mixed`, `u_bits = 3`, `selective_pct = 100` | threshold mixed point |
| P3 | Fixed mixed | `repr_type = mixed`, `u_bits = 4`, `selective_pct = 100` | candidate sweet-spot pilot point |
| P4 | Fixed mixed | `repr_type = mixed`, `u_bits = 5`, `selective_pct = 100` | heavier pilot point |
| P5 | Full unary | `repr_type = int_unary` | aggressive unary endpoint where tractable |
| P6 | Adaptive unary | `adaptive-search`, `level_ubits = [0,2,3,4,5,6]` | pilot proposed method |

### Track B Confirmatory Baselines

| ID | Family | Settings | Role |
| --- | --- | --- | --- |
| C0 | Binary | `repr_type = int_binary` | main confirmatory reference |
| C1 | Fixed mixed | `repr_type = mixed`, `u_bits = 3`, `selective_pct = 100` | lower-overhead confirmatory point |
| C2 | Fixed mixed | `repr_type = mixed`, `u_bits = 4`, `selective_pct = 100` | main fixed mixed confirmatory point |
| C3 | Adaptive unary | `adaptive-search`, `level_ubits = [0,3,4,5]` | main proposed method |

### Supporting Analytic Baselines

| ID | Family | Settings | Role |
| --- | --- | --- | --- |
| A0 | ECC weak reference | `BCH 10t` | low-overhead protection point |
| A1 | ECC medium reference | `BCH 50t` | AiF-like stronger-ECC comparison point |
| A2 | ECC strong reference | `BCH 64t` | upper-cost ECC point |
| A3 | Full unary extreme | `m(8) = 255` | analytic upper-overhead reference, not mandatory empirical run |

## Exact Experiment List

### E0. Reuse Existing Supporting Sensitivity Evidence

Purpose:

- support the claim that sensitivity is heterogeneous

Data source:

- `reports/bitflip_experiments_20260309.md`
- `reports/bitflip_methodology_and_results_20260309.md`

Paper use:

- motivation support
- discussion support
- appendix-level evidence if needed

### E1. Track A Fixed-Baseline Pilot Sweep on OPT-125M

Model:

- `/raid/LLM/opt-125m`

Runs:

- P0, P1, P2, P3, P4, P5

Outputs:

- one run directory per baseline
- `metrics/baseline.json`
- `metrics/fixed_eval.json`

Purpose:

- verify pipeline behavior
- screen useful mixed levels
- keep an empirical comparison against a tractable aggressive-unary endpoint

### E2. Track A Adaptive Pilot Search on OPT-125M

Model:

- `/raid/LLM/opt-125m`

Run:

- P6

Outputs:

- `metrics/search_setup.json`
- `metrics/adaptive_candidates.jsonl`
- `metrics/final_policy_eval.json`
- `metrics/best_policy.json`

Purpose:

- confirm the adaptive search path
- compare adaptive vs fixed levels in the pilot environment

### E3. Track B Fixed-Baseline Confirmatory Sweep on OPT-6.7B

Model:

- `/raid/LLM/opt-6.7b`

Runs:

- C0, C1, C2

Outputs:

- one run directory per baseline
- `metrics/baseline.json`
- `metrics/fixed_eval.json`

Purpose:

- establish the main int8 confirmatory baseline family

### E4. Track B Adaptive Confirmatory Search on OPT-6.7B

Model:

- `/raid/LLM/opt-6.7b`

Run:

- C3

Outputs:

- `metrics/search_setup.json`
- `metrics/adaptive_candidates.jsonl`
- `metrics/final_policy_eval.json`
- `metrics/best_policy.json`

Purpose:

- generate the main paper-critical adaptive result

### E5. Track C System Projection Table for `7B / 30B / 70B / 100B`

Purpose:

- support the motivation that large-model flash inference creates structural
  reclaim pressure
- keep the paper's `7B / 30B / 70B / 100B` scale assumptions visible

Inputs:

- locked system assumptions from `paper_session_bridge.md`
- model-size assumptions already frozen in the paper plan

Outputs:

- projection table for model size, block mapping, hot-block interpretation, and
  service gap

### E6. Track C Strong-ECC System Comparison

Purpose:

- support the claim that naive strong ECC is costly

Inputs:

- `10t`, `50t`, `64t` BCH parity overhead
- LSB-only effective data ratio
- AiF-derived area / power reference

Outputs:

- ECC comparison table and short discussion paragraph

### E7. Optional Track A Pilot Replication on OPT-1.3B

Purpose:

- check whether pilot trends are stable beyond `OPT-125M`

Status:

- optional
- run only if `OPT-125M` pilot trends are noisy or ambiguous

### E8. Optional Track B Confirmatory Replication on LLaMA2-7B

Purpose:

- reduce the risk that the main claim depends on one model family

Recommended runs:

- C0, C2, C3

Status:

- optional but valuable

## Required Metrics

### Primary Metrics

1. Per-task mean accuracy at each BER.
2. Per-task standard deviation across bit-error seeds.
3. Relative accuracy drop:
   `1 - mean_accuracy / baseline_accuracy`.
4. Pass / fail against the `5%` threshold.
5. Policy overhead ratio `Omega`.

### Secondary Metrics

1. Task-average mean accuracy across `piqa`, `hellaswag`, `arc_easy`.
2. Minimum pass margin across all task / BER pairs.
3. Encoded-width summary for each fixed or adaptive policy.
4. Effective user-data ratio relative to raw stored width.
5. Hardware proxy counts for representative unary levels.

### Interpretation Metrics

1. Best-policy robustness margin at `BER = 1e-2`.
2. Binary-to-best-policy overhead delta.
3. Comparison between the best policy and heavier unary expansion points.
4. Mapping from best-policy robustness result to retry / reclaim discussion.

## Required Output Files

### Fixed Runs

Each fixed baseline run must retain:

- `config/resolved_args.json`
- `metrics/baseline.json`
- `metrics/fixed_eval.json`
- `logs/run.log`

### Adaptive Runs

Each adaptive run must retain:

- `config/resolved_args.json`
- `metrics/search_setup.json`
- `metrics/adaptive_candidates.jsonl`
- `metrics/final_policy_eval.json`
- `metrics/best_policy.json`
- `logs/run.log`

### Paper-Aggregation Files To Produce Later

These do not exist yet, but the experiment matrix requires them.

- one merged CSV/JSON for pilot sweeps
- one merged CSV/JSON for confirmatory sweeps
- one merged CSV/JSON for main result table assembly
- one short text note mapping the best confirmatory policy to Figure F6
  interpretation

## Figure and Table Data Map

### Main-Body Figures

#### Figure F4. Confirmatory Accuracy vs BER

Data source:

- `fixed_eval.json` from E3
- `final_policy_eval.json` from E4
- optional replication from E8

Need to plot:

- C0, C1, C2, C3
- per task and task-average views

#### Figure F5. Confirmatory Accuracy vs Overhead at `BER = 1e-2`

Data source:

- E3 and E4 outputs at `BER = 1e-2`

Need to plot:

- x-axis:
  overhead ratio `Omega`
- y-axis:
  mean accuracy or relative accuracy drop
- markers:
  C0, C1, C2, C3

#### Figure F6. Retry / Reclaim Interpretation

Data source:

- best confirmatory result from E4
- optional replication from E8
- system projection from E5
- ECC system comparison from E6

Need to show:

- best robustness result
- capacity / retry / reclaim interpretation
- explicit label that the system mapping is modeled, not fully simulated

### Appendix / Supporting Figures

#### Figure AF1. Pilot Accuracy vs BER

Data source:

- `fixed_eval.json` from E1
- `final_policy_eval.json` from E2
- optional E7 replication

Need to show:

- P0 through P6

Purpose:

- support level-screening intuition
- show behavior against a tractable aggressive-unary endpoint

### Main-Body Tables

#### Table T4. Evaluation Baselines

Data source:

- baseline family frozen in this document

Recommended layout:

- separate pilot and confirmatory baseline blocks

#### Table T5. Main Result Summary

Data source:

- E3 and E4 outputs
- optional E8 replication

Columns to fill:

- model
- policy
- BER
- per-task mean accuracy
- task-average accuracy
- relative accuracy drop
- pass/fail
- overhead ratio

### Supporting Tables

#### Table T2. Locked System Assumptions and Derived Numbers

Data source:

- `paper_session_bridge.md`
- E5
- E6

#### Table AT1. Pilot Result Summary

Data source:

- E1
- E2
- optional E7

## Minimum Paper-Critical Run Set

The minimum set required before Phase 4 can claim evidence closure is:

1. E1
2. E2
3. E3
4. E4
5. E5
6. E6

Optional but valuable:

- E7
- E8

## What Is Explicitly Out of Scope For Phase 4

The following are not blockers for the first full evidence pass:

- end-to-end BER sweeps on int8 `30B`, `70B`, `100B`
- int8 full-unary `lm_eval` sweeps as a mandatory baseline
- controller-level reclaim simulation
- residual-BER-driven ECC-vs-unary accuracy matching in `lm_eval`
- post-synthesis unary codec area / power / timing
- correlated-fault and burst-error studies

If added later, they strengthen the paper but should not delay the first
evidence-closure pass.

## Execution Order

Run the next phase in this order:

1. E1: `6-bit` pilot fixed baselines on `OPT-125M`
2. E2: `6-bit` pilot adaptive search on `OPT-125M`
3. E3: int8 confirmatory fixed baselines on `OPT-6.7B`
4. E4: int8 confirmatory adaptive search on `OPT-6.7B`
5. E5: system projection table assembly
6. E6: ECC system comparison table assembly
7. E7: optional pilot replication on `OPT-1.3B`
8. E8: optional confirmatory replication on `LLaMA2-7B`

Reason for this order:

- use the pilot track to validate the experiment path and screen levels
- move the main paper claim to int8 confirmation on a `7B` model
- leave the cross-model replication and analytic table polishing after the main
  confirmatory result exists

## Next Artifact To Produce

The next required artifacts are not new prose documents.

They are the Phase 4 experiment outputs from:

- E1
- E2
- E3
- E4
- E5
- E6

Do not start the full paper draft until at least E3 and E4 have finished and
the main result table can be populated without placeholders.
