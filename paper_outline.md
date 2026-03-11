# Paper Outline

Date: 2026-03-11
Workspace: `/home/jwjeong/study/quant_analysis`
Status: Phase 1 deliverable

## Paper Frame

This paper is centered on `unary-based mitigation for harsh-TLC flash LLM inference`.

- primary mechanism:
  selective / adaptive unary encoding
- supporting background / baseline:
  ECC allocation and stronger-ECC operating points
- explicit non-goal:
  do not position this as a generic ECC allocation paper

## Provisional Title Candidates

1. `Adaptive Unary Encoding for Harsh-TLC Flash-Based LLM Inference`
2. `Selective Unary Encoding for Error-Resilient Flash LLM Inference under Harsh TLC`
3. `Mitigating Read-Retry and Read-Reclaim Pressure via Adaptive Unary Encoding`

Recommended working title for now:

`Adaptive Unary Encoding for Harsh-TLC Flash-Based LLM Inference`

## One-Sentence Thesis

Prior flash-based LLM inference systems mostly avoid harsh TLC conditions through
SLC/LSB-page/low-BER assumptions; we instead study whether selective unary-based
weight encoding can preserve LLM accuracy under harsher TLC error rates while
offering a better robustness-overhead tradeoff than naive strong-ECC scaling.

## Provisional Contribution Framing

These bullets define the intended paper contributions, but any result-dependent
wording must stay provisional until Phase 4 evidence is closed.

1. We formulate harsh-TLC flash LLM inference as a system-level tradeoff where
   naive strong ECC is costly in capacity/area/power, while weaker protection
   increases read-retry and read-reclaim pressure under large-model hot-block
   access patterns.
2. We present a representation-aware unary / mixed-unary perturbation model for
   quantized weights, replacing binary-only `4^k` reasoning with codec-aware
   decoded error `D_u(C_u(q) \oplus \xi) - q`, and define the policy space over
   ordered unary levels `u_bits`.
3. We describe a selective / adaptive unary policy that uses model-side
   sensitivity heterogeneity to decide where unary expansion is worth paying,
   while keeping storage-overhead accounting explicit.
4. We evaluate binary, selected fixed mixed-unary levels, and adaptive unary
   across BER points including `1e-2`, while using full unary only where it is
   still tractable, to determine whether a practical sweet spot exists under
   the `<= 5%` accuracy-drop target.

## Section Outline

### 1. Introduction

Role:

- establish the harsh-TLC problem setting
- show why prior flash-LLM work does not settle this regime
- state the unary-centered thesis and paper contributions

Key points to include:

- prior systems rely on SLC, LSB-only placement, or low BER assumptions
- harsh TLC makes stronger protection necessary but expensive
- weaker protection creates retry / reclaim pressure
- unary-based mitigation is the mechanism explored in this paper

### 2. Background and Motivation

Role:

- ground the paper with quantitative motivation
- connect flash reliability costs to LLM serving access patterns

Key points to include:

- Lincoln / AiF / Cambricon-LLM operating assumptions
- BCH parity overhead and LSB-only effective capacity loss
- AiF Figure 5 area / power cost of strong BCH
- reclaim lifetime gap under the `5 tok/s`, `3 years`, `200K`, `3K P/E` setting
- read-retry timing penalty from `refs/motivation.pdf`

### 3. Problem Definition

Role:

- define the target operating point and optimization target clearly

Key points to include:

- weight-only int8 LLM inference
- flash-resident or flash-near execution
- harsh TLC reliability regime
- target condition: accuracy drop `<= 5%` at `BER = 1e-2`
- compare robustness gain against storage / effective-capacity / hardware-cost
  overhead

### 4. Unary-Based Mitigation Method

Role:

- define the canonical paper method language

Subsections:

4.1 Representation family

- binary, mixed unary, full unary where tractable
- temporal unary codec and nearest-Hamming decoder
- encoded width `m(u)` and overhead growth

4.2 Representation-aware perturbation model

- why binary-only `4^k` is insufficient
- decoded integer perturbation `Delta_u(q, xi)`
- exact / low-BER risk language consistent with current implementation

4.3 Selective / adaptive policy space

- ordered level set `level_ubits`
- group ranking by importance / sensitivity
- boundary-based policy construction over groups

4.4 Overhead and practical constraints

- encoded-width overhead
- optional hardware-cost proxies
- why aggressive unary expansion is not automatically optimal

### 5. Experimental Setup

Role:

- freeze the evaluation matrix used to validate the thesis

Key points to include:

- model list: `7B / 30B / 70B / 100B` when applicable
- quantization: int8 weight-only
- BER sweep including `1e-2`
- baselines: binary, fixed mixed levels, adaptive unary, and full unary only
  where it is tractable, plus an optional strong-ECC comparison point
- metrics: accuracy drop, encoded-width overhead, effective capacity, retry /
  reclaim interpretation metrics
- pass/fail rule tied to the `<= 5%` target

### 6. Results

Role:

- answer the paper-critical open questions without overstating claims

Questions this section must answer:

1. Which policy performs best at `BER = 1e-2`?
2. How much overhead does the best policy require?
3. Is the best policy better than pure binary and more practical than heavier
   unary expansion?
4. Are the gains large enough to support the retry / reclaim mitigation
   interpretation?

### 7. Related Work

Role:

- place the paper against adjacent literatures without diluting the main story

Buckets:

- flash-based LLM inference
- flash reliability and ECC
- approximate / error-tolerant model storage
- unary / thermometer coding for robustness

### 8. Discussion and Limitations

Role:

- narrow the claim scope and make remaining gaps explicit

Key points to include:

- retry / reclaim impact is interpreted from measured robustness, not fully
  simulated end-to-end
- some hardware costs remain proxy estimates
- correlated faults, burst errors, and controller policy details remain future
  work

### 9. Conclusion

Role:

- restate the problem, method, and practical takeaway in one short section

## Writing Guardrails

- keep `adaptive unary encoding` as the main novelty axis
- keep ECC allocation in a supporting role unless later redirected
- avoid presenting binary-only formulas from `proposal.md` as if they fully
  explain mixed / unary behavior
- avoid claiming success at `BER = 1e-2` until the evaluation section proves it

## Next Document To Write

`paper_method.md`

Its job is to merge the useful parts of `proposal.md` and
`proposal_adaptive_unary_level_mapping.md` into one canonical method note for
the paper.
