# Paper Session Bridge

Date: 2026-03-11
Workspace: `/home/jwjeong/study/quant_analysis`

Primary execution doc:

- `paper_plan.md`
  - read this first for current phase, next task, and session rules

## Purpose

This file is the persistent handoff for turning
`overview.md` into a paper-ready manuscript.

After reading `paper_plan.md`, the next session should be able to read this
file and immediately know:

- what the paper is about
- which assumptions are already fixed
- which claims are already supported
- which claims still need evidence
- what to write next

## Current Paper Direction

The paper is **not** primarily about generic ECC allocation anymore.

The current paper direction is:

- harsh TLC NAND conditions for flash-based LLM inference
- strong ECC is expensive in SSD-level PPA
- weak ECC pushes the system into a read-reclaim vs read-retry tradeoff
- LLMs are not uniformly sensitive to residual bit errors
- selective/adaptive unary-based encoding may reduce effective error damage
  enough to relax retry/reclaim pressure while preserving accuracy

In one sentence:

> Prior flash-based LLM systems mostly avoid harsh TLC conditions through
> SLC/LSB-page/low-BER assumptions; we want a unary-based encoding policy that
> remains accurate under harsher TLC error conditions without paying the full
> cost of naive strong ECC.

## Non-Negotiable Scope

- weight-only focus
- quantized LLM inference
- flash-resident or flash-near execution setting
- harsh TLC reliability regime is central to the motivation
- no training-heavy method
- no story centered on repeated massive fault-injection loops
- paper center should remain `unary-based mitigation`, not drift into a
  separate "ECC allocation only" paper unless explicitly redirected

## Fixed Assumptions

These assumptions are already chosen and should be treated as locked unless the
user explicitly changes them:

- SSD organization:
  `16 channel x 4 chip x 4 plane = 256-way parallelism`
- block size:
  `768 pages/block`
- target service throughput:
  `5 tok/s`
- target service lifetime:
  `3 years`
- reclaim threshold:
  `200K` reads
- endurance budget:
  `3K P/E`
- model sizes:
  int8 `7B`, `30B`, `70B`, `100B`
- ECC candidate family:
  `BCH(1KiB codeword, k=8192 bits)` with `10t`, `50t`, `64t`
- accuracy target:
  maintain accuracy drop within `5%` at `BER = 1e-2`
- placement assumption:
  bandwidth-oriented distributed/striped placement across all available
  parallel units

## Derived Numbers Already Established

These numbers are already computed and should be reused consistently.

### ECC Overhead

For `BCH(1KiB, k=8192)`:

- `10t` parity bits: `140b`, parity overhead `1.709%`
- `50t` parity bits: `700b`, parity overhead `8.545%`
- `64t` parity bits: `896b`, parity overhead `10.938%`

If TLC is used in `LSB-page only` mode:

- effective user-data ratio vs raw TLC:
  - `10t`: about `32.8%`
  - `50t`: about `30.7%`
  - `64t`: about `30.0%`

### Reclaim / Lifetime Math

With:

- reclaim threshold `200K`
- endurance budget `3K P/E`

the total allowed cumulative reads per hot block are:

`200,000 x 3,000 = 6e8`

Under a hot-block full-sweep assumption:

- token-level hot-block read increment:
  `768`
- lifetime token budget per hot block:
  `6e8 / 768 ~= 7.8e5 tokens`

Service demand under `5 tok/s` for `3 years`:

- required tokens:
  `4.73e8`

Gap:

- required / supported:
  about `605x`

Equivalent condition for surviving `3 years`:

- average hot-block read-count increase must stay below about
  `1.27 page/token`

### Parallelism / Model Mapping

Under `256-way parallelism`, even int8 `7B` already exceeds
`1 block x all parallelism` for realistic NAND page sizes.

Useful threshold:

- int8 `7B` would need page size above about `34.8 KiB` to avoid spanning at
  least one full block per parallel unit

Therefore, for practical NAND page sizes, all `7B/30B/70B/100B` settings are
consistent with the "most model-mapped blocks become hot blocks" argument.

### Read-Retry Timing

From `refs/motivation.pdf` slide 7:

- `tR = 30 us`
- `tECC = 561 ns`
- `N_RR = 1 / 3 / 5`

Derived:

- one retry is about `53.5x` the ECC decode time
- approximate per-access read-path times:
  - `N_RR = 1`: `30.561 us`
  - `N_RR = 3`: `90.561 us`
  - `N_RR = 5`: `150.561 us`

Interpretation:

- retry overhead is dominated by extra read latency, not ECC decode logic
- bandwidth degradation from read-retry is therefore structurally important

## Reference Anchors Already Used

These references are already woven into `overview.md` and should stay aligned
with the paper story:

- `refs/Lincoln.pdf`
  - Lincoln uses SLC-like reliable operating conditions plus BCH-style ECC
- `refs/aif.pdf`
  - AiF stores weights in LSB pages and uses BCH-based ECC
  - Figure 5 is already used for the "strong ECC is PPA-expensive" argument
- `refs/cambricon-llm.pdf`
  - Cambricon-LLM uses TLC but remains in a low-BER regime around `1e-4`
- `refs/motivation.pdf`
  - slide 7 is already used for retry latency penalty numbers
- `refs/Leviathan.pdf`
  - use only as supporting intuition for flash-streamed large-model inference,
    not as the main novelty anchor

## What Is Already Written

- `overview.md`
  - motivation and problem framing are now quantitatively grounded
- `paper_outline.md`
  - Phase 1 story-freeze document with title direction, thesis, contribution
    framing, and section plan
- `paper_method.md`
  - canonical method note with notation, unary-aware equations, and explicit
    evidence / figure / table / equation inventory
- `paper_experiment_matrix.md`
  - exact claim-to-evidence map, baseline family, run list, metric rules, and
    figure / table data sources
- `paper_search_strategy_note.md`
  - why the current greedy empirical search is too slow and what search
    redesign path is now preferred
- `proposal.md`
  - analytic exposure / ECC residual-risk formulation
- `proposal_adaptive_unary_level_mapping.md`
  - representation-aware unary / mixed formulation
- `adaptive_unary_level_mapping/`
  - implementation and hardware-cost notes for unary encoding
- `reports/bitflip_experiments_20260309.md`
  - early empirical signs that sensitivity is heterogeneous
- `reports/bitflip_methodology_and_results_20260309.md`
  - detailed methodology state

## Phase 1 Outcome

Phase 1 is now complete.

- story status:
  frozen around `adaptive / selective unary encoding` as the main paper axis
- ECC allocation status:
  supporting background / baseline, not the main thesis
- current working title:
  `Adaptive Unary Encoding for Harsh-TLC Flash-Based LLM Inference`
- next deliverable:
  `paper_method.md`

Current thesis sentence:

> Prior flash-based LLM inference systems mostly avoid harsh TLC conditions
> through SLC/LSB-page/low-BER assumptions; we instead study whether selective
> unary-based weight encoding can preserve LLM accuracy under harsher TLC error
> rates while offering a better robustness-overhead tradeoff than naive
> strong-ECC scaling.

## Phase 2 Outcome

Phase 2 is now complete.

- canonical method status:
  frozen in `paper_method.md`
- notation status:
  paper-facing notation is now defined in one place
- correction status:
  binary-only `4^k` language is explicitly demoted to supporting background
- evidence status:
  missing data, figures, tables, and required equations are explicitly listed
- next deliverable:
  `paper_experiment_matrix.md`

Important consequence:

- future draft sections should inherit method language from `paper_method.md`
- if a later document depends on missing evidence, it should preserve the
  missing-data inventory instead of writing around the gap

## Phase 3 Outcome

Phase 3 is now complete.

- experiment status:
  the exact experiment matrix is frozen in `paper_experiment_matrix.md`
- pilot track:
  `RTN`, `w_bits = 6`, `OPT-125M`,
  tasks `piqa/hellaswag/arc_easy`,
  BER `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`
- confirmatory track:
  `RTN`, `w_bits = 8`, `OPT-6.7B`,
  tasks `piqa/hellaswag/arc_easy`,
  BER `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`
- baseline status:
  pilot uses binary / fixed mixed / full unary / adaptive;
  confirmatory uses binary / fixed `u = 3/4` / adaptive
- supporting analytic track:
  int8 `7B / 30B / 70B / 100B` system projection plus BCH `10t / 50t / 64t`
- next work:
  Phase 4 evidence runs `E1-E6`

Important consequence:

- future sessions should stop debating the experiment scope and start producing
  the frozen outputs
- `6-bit` evidence is now pilot-only
- the main paper claim must rest on the int8 confirmatory track

## Search Strategy Reset

The current search strategy has been reassessed.

What is now considered true:

- the current implementation already showed the proof-of-possibility result we
  needed
- but the current global empirical greedy search is too slow for the main
  design-space exploration path

Therefore:

- keep the current greedy empirical path as
  - finalist verification
  - local refinement
  - fallback
- do not keep using it as the default global explorer

Preferred next direction:

- implement a `surrogate-guided search`
- use fast candidate generation first
- run `lm_eval` only on a small finalist set

Persistent note:

- see `paper_search_strategy_note.md`

Execution consequence:

- the experiment matrix remains frozen and valid
- full-scale Phase 4 execution is temporarily paused until the faster search
  path is ready

## Important Strategic Decision

Do **not** let the paper split into two independent theses:

1. analytic ECC allocation
2. adaptive unary encoding

For the current paper, the primary story should be:

- harsh TLC makes naive ECC unattractive
- model error tolerance opens room for approximate protection
- unary / mixed unary is the mechanism we explore
- selective/adaptive mapping is how we control unary overhead

The ECC allocation work remains valuable, but for this paper it should be:

- supporting background
- a comparison point
- a baseline / appendix direction

unless the user explicitly pivots back.

## Candidate Paper Claims

These are the claims the paper can potentially make. They are separated by
evidence status.

### Claims That Are Already Supportable

1. Existing flash-based LLM systems mostly avoid the harsh TLC regime through
   SLC-like operation, LSB-page restriction, or low-BER assumptions.
2. In harsh TLC, naive strong ECC can be SSD-level PPA-prohibitive.
3. In large-model flash inference, reclaim and retry pressure can become
   structural under realistic service assumptions.
4. Unary / mixed unary reduces decoded single-bit perturbation relative to
   ordinary binary representation.

### Claims That Still Need Final Evidence

1. Selective/adaptive unary can meet the `<=5%` accuracy-drop target at
   `BER = 1e-2`.
2. Unary-based encoding can outperform naive stronger ECC in a more attractive
   accuracy / overhead / retry-pressure tradeoff.
3. There exists a practical unary level policy that is clearly better than pure
   binary and more practical than heavier fixed unary expansion.

These are paper-critical. Do not write them as established facts until the
evaluation section supports them.

## Recommended Paper Structure

### 1. Introduction

Source primarily from:

- `overview.md`

Goal:

- establish harsh TLC problem
- establish strong ECC cost
- establish reclaim/retry dilemma
- position unary-based mitigation as the paper idea

### 2. Background and Motivation

Use:

- `refs/Lincoln.pdf`
- `refs/aif.pdf`
- `refs/cambricon-llm.pdf`
- `refs/motivation.pdf`

Goal:

- explain why prior work does not directly solve harsh TLC
- explain why retry/reclaim matter for flash inference
- explain why model-side error tolerance is relevant

### 3. Problem Definition

Use:

- `proposal.md`
- `overview.md`

Goal:

- define target operating point
- define constraints:
  - BER target
  - accuracy target
  - storage overhead
  - retry / reclaim burden

### 4. Method

Use:

- `proposal_adaptive_unary_level_mapping.md`
- `adaptive_unary_level_mapping/docs/unary_hardware.md`
- `adaptive_unary_level_mapping/README.md`

Goal:

- define binary / mixed / unary representation
- define decoded error under BER
- define policy space over `u_bits`
- define selective / adaptive mapping strategy

### 5. Experimental Setup

Must specify:

- models:
  `7B / 30B / 70B / 100B` when applicable
- quantization:
  int8
- BER sweep:
  include `1e-2`
- accuracy target:
  `<=5%` drop
- baselines:
  binary, fixed mixed levels, adaptive unary, and full unary only where it is
  tractable, plus an optional ECC-strong baseline
- overhead metrics:
  encoded width, effective capacity, hardware cost proxies

### 6. Results

Must answer:

1. Does unary improve robustness at high BER?
2. How much storage overhead does it cost?
3. Which `u_bits` or adaptive policy is best?
4. Is there a useful sweet spot vs heavier unary expansion?
5. Does it help enough to justify reducing retry / reclaim pressure?

### 7. Related Work

Organize by:

- flash-based LLM inference
- ECC / flash reliability
- approximate / error-tolerant model storage
- unary / thermometer coding for robustness

### 8. Discussion and Limitations

Be explicit about:

- retry/reclaim are currently modeled, not fully simulated end-to-end
- some hardware costs are proxy costs, not synthesized final silicon
- correlated faults and burst effects remain future work

## Execution Plan

### Phase 1: Freeze the Paper Story

Deliverable:

- a one-page outline stating the title, one-sentence thesis, and section plan

Required action:

- do not keep reopening the motivation unless a contradiction is found

### Phase 2: Build a Canonical Method Section

Deliverable:

- one clean method document that merges the useful parts of:
  - `proposal.md`
  - `proposal_adaptive_unary_level_mapping.md`

Guideline:

- binary-only ECC formulas from `proposal.md` should not be presented as if
  they fully describe unary / mixed behavior
- unary paper method must use representation-aware perturbation language

### Phase 3: Finalize Evaluation Matrix

Decide and write down:

- exact model list for experiments
- exact BER points
- exact baseline list
- exact metrics
- exact pass/fail condition for the `5%` target

### Phase 4: Close Evidence Gaps

Before claiming paper-level success, verify:

- best unary policy under `BER = 1e-2`
- storage overhead of that policy
- robustness comparison vs binary baseline
- whether the result is strong enough to claim practical mitigation

### Phase 5: Write the Paper Skeleton

Create drafts for:

- abstract
- introduction
- background / motivation
- method
- experimental setup
- results shell with placeholder figure / table slots

## Immediate Next-Session Tasks

The next session should do these in order:

1. read:
   - `paper_plan.md`
   - `paper_session_bridge.md`
   - `paper_search_strategy_note.md`
   - `paper_outline.md`
   - `paper_method.md`
   - `paper_experiment_matrix.md`
   - `proposal_adaptive_unary_level_mapping.md` if notation needs checking
   - `proposal.md` if ECC background wording needs checking
   - `overview.md` if motivation wording needs to be cross-checked
2. design the first `surrogate-guided search` prototype
3. implement that prototype before resuming `E1-E6`
4. preserve the current experiment matrix as the target execution plan
5. keep `paper_outline.md`, `paper_method.md`, and
   `paper_experiment_matrix.md` fixed unless a contradiction is found

## Suggested Title Directions

Possible title directions to test:

- `Adaptive Unary Encoding for Harsh-TLC Flash-Based LLM Inference`
- `Selective Unary Encoding for Error-Resilient Flash-Based LLM Inference`
- `Mitigating Read-Retry and Read-Reclaim Pressure via Adaptive Unary Encoding`

Current preferred working title:

- `Adaptive Unary Encoding for Harsh-TLC Flash-Based LLM Inference`

Do not treat the title as final yet, but keep it centered on:

- harsh TLC
- LLM inference
- unary or adaptive unary
- error resilience

## Open Questions

These questions are still unresolved and should be revisited later, not now:

- should the paper include ECC allocation as a baseline section or omit it from
  the main body?
- which venue style is the intended target?
- should the main metric be accuracy drop only, or a joint
  accuracy / overhead / retry-pressure plot?

## File To Read First Next Time

If a future session has very limited context, read in this order:

1. `paper_plan.md`
2. `paper_session_bridge.md`
3. `paper_search_strategy_note.md`
4. `paper_outline.md`
5. `paper_method.md`
6. `paper_experiment_matrix.md`
7. `proposal_adaptive_unary_level_mapping.md`
8. `proposal.md`
9. `overview.md`

That should be enough to resume work productively.
