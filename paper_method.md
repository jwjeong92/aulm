# Paper Method

Date: 2026-03-11
Workspace: `/home/jwjeong/study/quant_analysis`
Status: Phase 2 deliverable

## Purpose

This document is the canonical method note for the paper.

It has two jobs:

1. define the paper's method in one place using notation that matches the
   current unary implementation
2. state explicitly which evidence, figures, tables, and result data are still
   missing before the method can be claimed as validated

## Method Position

The method is centered on `adaptive unary encoding` for harsh-TLC flash-based
LLM inference.

- primary novelty:
  representation-aware unary / mixed-unary mitigation
- supporting background:
  ECC residual-risk language and stronger-ECC operating points
- non-goal:
  do not write this as a generic ECC allocation paper

Result-dependent claims remain provisional until experiments close the evidence
gap at `BER = 1e-2`.

## Scope and Locked Assumptions

This method note assumes:

- weight-only quantized LLM inference
- flash-resident or flash-near execution
- harsh TLC reliability regime
- int8 model settings including `7B / 30B / 70B / 100B`
- target accuracy condition:
  accuracy drop `<= 5%` at `BER = 1e-2`
- SSD organization:
  `16 channel x 4 chip x 4 plane = 256-way parallelism`
- block size:
  `768 pages/block`
- throughput target:
  `5 tok/s`
- service lifetime target:
  `3 years`
- reclaim threshold:
  `200K`
- endurance budget:
  `3K P/E`
- comparison ECC family:
  `BCH(1KiB codeword, k=8192 bits)` with `10t / 50t / 64t`

## Notation

Use the following notation consistently in the paper body.

- `W`:
  original quantized weight bitwidth
- `q in {0, ..., 2^W - 1}`:
  unsigned quantized weight integer
- `g`:
  quantization group
- `w`:
  one weight
- `s_g`:
  quantization scale of group `g`
- `h_c`:
  activation sensitivity proxy for input column `c`
- `alpha_l`:
  optional layer amplification factor for layer `l`
- `u`:
  unary level assigned to one group
- `b(u) = W - u`:
  number of binary bits kept in mixed mode
- `m(u)`:
  encoded width at level `u`
- `C_u` / `D_u`:
  encode / decode maps for level `u`
- `xi in {0,1}^{m(u)}`:
  encoded-space error pattern
- `Delta_u(q, xi)`:
  decoded integer perturbation after encode, corruption, and decode
- `r_w(u, p)`:
  expected risk contribution of weight `w` at raw BER `p`
- `R_g(u, p)`:
  group-level expected risk
- `I_g`:
  importance score used to rank groups for adaptive mapping
- `Omega({u_g})`:
  storage-overhead ratio of a policy

## 1. Supporting Background: Binary Exposure and ECC Risk

This section stays in the paper as supporting background, not as the main
method claim.

For direct binary storage, a flip at binary bit `k` causes:

$$
|\Delta q_k| = 2^k
$$

and therefore:

$$
(\Delta w_k)^2 = s_g^2 4^k
$$

Using the local second-order proxy, per-weight binary exposure is:

$$
e_w(k) = \frac{1}{2} \alpha_{l(w)} h_{c(w)} s_{g(w)}^2 4^k
$$

If a physical protection unit `b` stores multiple weights, the block exposure
seen by the protection layer is:

$$
E_b(k) = \sum_{w \in b} e_w(k)
$$

For an ECC mode `m` with residual error probability `p_k^m`, the supporting
residual-risk view is:

$$
R_b(m) = \sum_k E_b(k) p_k^m
$$

This decomposition is still useful in the paper because it separates:

- model-side exposure
- protection-side residual error probability

But this binary `4^k` language is not sufficient once the stored representation
is unary or mixed.

## 2. Representation Family

The current project uses a unified family of binary, mixed-unary, and full
unary encodings.

Define:

$$
b(u) := W - u
$$

$$
\lambda(u) := 2^u - 1
$$

The encoded width is:

$$
m(u) =
\begin{cases}
W, & u = 0 \\
\lambda(u) + b(u), & 0 < u < W \\
2^W - 1, & u = W
\end{cases}
$$

This means:

- `u = 0` is pure binary
- `0 < u < W` is mixed unary-binary
- `u = W` is full unary

### 2.1 Temporal Unary Codec

For the unary field value `a in {0, ..., 2^u - 1}`, the temporal unary encoder
is:

$$
T_u(a) = 2^a - 1
$$

The decoder chooses the nearest thermometer codeword in Hamming distance and
breaks ties toward the smaller decoded value:

$$
\hat a_u(y) =
\arg\min_{k \in \{0, \dots, 2^u - 1\}} d_H(y, T_u(k))
$$

### 2.2 Mixed Unary-Binary Representation

For mixed level `0 < u < W`, define:

$$
a_u(q) := \left\lfloor \frac{q}{2^{b(u)}} \right\rfloor
$$

$$
r_u(q) := q \bmod 2^{b(u)}
$$

The encoder is:

$$
C_u(q) = \left(T_u(a_u(q)) \ll b(u)\right) \;|\; r_u(q)
$$

The decoder is:

$$
D_u(y) =
\left(\hat a_u\!\left(\left\lfloor \frac{y}{2^{b(u)}} \right\rfloor\right) \ll b(u)\right)
\;|\;
\left(y \bmod 2^{b(u)}\right)
$$

Endpoint cases are:

$$
C_0(q) = q, \qquad D_0(y) = y
$$

$$
C_W(q) = T_W(q), \qquad
D_W(y) = \arg\min_{k \in \{0, \dots, 2^W - 1\}} d_H(y, T_W(k))
$$

## 3. Representation-Aware Perturbation Model

This is the main correction relative to the earlier binary-only formulation.

For unary or mixed storage, the correct perturbation unit is not a binary bit
position. It is the full encoded-space corruption pattern.

Define the decoded integer perturbation:

$$
\Delta_u(q, \xi) := D_u(C_u(q) \oplus \xi) - q
$$

Then the local second-order proxy for one weight becomes:

$$
\ell_w(u, \xi)
\approx
\frac{1}{2} \alpha_{l(w)} h_{c(w)} s_{g(w)}^2
\left(\Delta_u(q_w, \xi)\right)^2
$$

This replaces the binary-only term:

$$
\frac{1}{2} \alpha h s^2 4^k
$$

for any method section meant to match the current implementation.

### 3.1 Expected Risk Under Raw BER

Let `Xi_u(p)` be the encoded-space random bit-flip vector with i.i.d.
Bernoulli coordinates of raw BER `p`.

The expected risk contribution of one weight is:

$$
r_w(u, p)
=
\frac{1}{2} \alpha_{l(w)} h_{c(w)} s_{g(w)}^2
\mathbb E_{\Xi_u(p)}
\left[
\left(\Delta_u(q_w, \Xi_u(p))\right)^2
\right]
$$

Group-level risk is:

$$
R_g(u, p) = \sum_{w \in g} r_w(u, p)
$$

This is the canonical analytic quantity for binary, mixed, and full unary.

### 3.2 Low-BER Single-Bit Approximation

For small `p`, define the single-bit decoded error energy:

$$
\rho_u(q, t) :=
\left(D_u(C_u(q) \oplus e_t) - q\right)^2
$$

Then:

$$
r_w(u, p)
=
\frac{1}{2} \alpha_{l(w)} h_{c(w)} s_{g(w)}^2
\left[
p \sum_{t=0}^{m(u)-1} \rho_u(q_w, t) + O(p^2)
\right]
$$

Important interpretation:

- for `u = 0`, the old binary result is recovered
- for `u > 0`, error significance depends on `q`, `u`, and the decoder
- unary robustness cannot be summarized by `4^k` alone

### 3.3 Qualitative Consequence

For the current temporal unary decoder, the first-order low-BER analysis gives
an important directional fact:

- increasing `u` increases encoded width
- increasing `u` also reduces decoded jump magnitude for single encoded-bit
  errors
- for sufficiently low BER, `u >= 3` is analytically favorable over direct
  binary in first-order expected decoded error energy

This remains a directional method insight, not a final paper claim, because
the actual paper target is the high-BER regime including `BER = 1e-2`, where
multi-bit events matter.

## 4. Adaptive Level Policy

The paper does not force one global encoding level on all groups.

Instead, it searches over an ordered level family:

$$
\mathcal U = \{u_0 < u_1 < \dots < u_{L-1}\}
$$

with the current default set:

$$
\mathcal U_{\mathrm{default}} = \{0, 2, 3, \dots, W\}
$$

Each quantization group `g` receives an importance score `I_g`.
Sort groups in ascending importance:

$$
I_{\pi(1)} \le I_{\pi(2)} \le \dots \le I_{\pi(G)}
$$

Let the search choose nondecreasing boundaries:

$$
0 \le n_0 \le n_1 \le \dots \le n_{L-2} \le G
$$

The adaptive group policy is then:

$$
u_{\pi(r)} =
\begin{cases}
u_0, & 1 \le r \le n_0 \\
u_\ell, & n_{\ell-1} < r \le n_\ell,\ \ell = 1, \dots, L-2 \\
u_{L-1}, & n_{L-2} < r \le G
\end{cases}
$$

Interpretation:

- low-importance groups can absorb more unary expansion
- high-importance groups can stay at safer or lower-overhead levels
- the optimization object is the level-boundary vector, not a free-form
  per-weight code assignment

Important clarification:

- in the current implementation, `I_g` comes from the selected importance
  metric, not from directly computing `R_g(u, p)` for every candidate level
- the default practical choice is `gptq_hessian`
- therefore the current search should be described as
  `importance-ordered boundary search`, not as an exact optimizer of the
  analytic risk

## 5. Overhead and Practical Constraints

If group `g` contains `n_g` weights and is assigned level `u_g`, the exact
storage-overhead ratio is:

$$
\Omega(\{u_g\}) =
\frac{\sum_g n_g m(u_g)}
{\sum_g n_g W}
$$

This is the first required cost metric.

The hardware notes in this repository also provide structural cost proxies for
each unary level:

- encoded width
- unary-field length
- encoder compare bit-slices
- decoder popcount bit-ops
- decoder scan bit-ops

For the current `W = 6` setting, representative points are already known:

- `u = 2`:
  `7b`, `1.167x` storage
- `u = 3`:
  `10b`, `1.667x` storage
- `u = 4`:
  `17b`, `2.833x` storage
- `u = 5`:
  `32b`, `5.333x` storage
- `u = 6`:
  `63b`, `10.500x` storage

These hardware quantities are currently proxy metrics, not post-synthesis
silicon results. The paper must say that explicitly.

Important scope note:

- these `W = 6` numbers are the current pilot / implementation-friendly points
- for `W = 8`, full unary expands to `255b`, so int8 full unary should be
  treated as an analytic extreme or a limited comparison point rather than a
  mandatory confirmatory baseline

## 6. Why Risk Is Still Defined

The analytic risk language is still useful, but its role must be stated
precisely.

In the current paper, `r_w(u, p)` and `R_g(u, p)` are used for:

1. defining the correct representation-aware damage model
2. explaining why unary level `u` changes robustness and overhead
3. motivating why an ordered level policy and group sensitivity ranking are
   sensible
4. providing a surrogate quantity that a future analytic optimizer could use

They are **not** the exact objective directly minimized by the current search
code.

The actual implemented search loop is:

1. compute group importance scores `I_g`
2. sort groups by that importance
3. greedily move level boundaries with `step_pct` and `rollback_steps`
4. evaluate each candidate empirically with BER injection and `lm_eval`
5. keep candidates by pass/fail, overhead, and margin

So the paper should describe the current method as:

- an analytically motivated search space
- plus an empirical greedy boundary search

and not as:

- a closed-form optimizer of `sum_g R_g(u_g, p)`

## 7. Evaluation Interface for the Method

The method section should define how a candidate policy is judged, even though
the full experiment matrix lives in a separate document.

Let `A_t^base` be the clean quantized baseline accuracy for task `t`, and let
`bar{A}_t({u_g}; beta)` be the mean accuracy of policy `{u_g}` at BER `beta`.

With tolerance `tau`, the policy passes when:

$$
\bar{A}_t(\{u_g\}; \beta) \ge (1 - \tau) A_t^{base}
\qquad
\text{for all } t, \beta
$$

The minimum margin is:

$$
M(\{u_g\}) :=
\min_{t, \beta}
\left[
\bar{A}_t(\{u_g\}; \beta) - (1 - \tau) A_t^{base}
\right]
$$

In the current search logic, candidates are compared by:

1. pass before fail
2. smaller `Omega`
3. larger `M`

This is enough method language for the paper. The exact benchmark set, BER
points, and seed schedule belong in `paper_experiment_matrix.md`.

## 8. What The Method Claims and Does Not Claim

What this method can already claim:

- harsh TLC creates a meaningful robustness-overhead design space
- unary / mixed-unary changes decoded perturbation geometry
- adaptive level assignment is a plausible way to exploit heterogeneous group
  sensitivity
- storage and hardware costs can be tracked explicitly through `m(u)` and the
  unary hardware proxy table

What this method cannot yet claim without additional evidence:

- that adaptive unary meets the `<= 5%` target at `BER = 1e-2`
- that adaptive unary beats stronger ECC overall at the system level
- that a specific level such as `u = 4` is the universal sweet spot
- that retry / reclaim pressure is reduced by a quantified amount in a full
  controller-level simulation

## 9. Required Evidence Before Paper Claims Become Validated

This section is mandatory because the current repository does not yet contain
all evidence needed for the final paper.

### 8.1 Core Result Data Still Needed

1. Accuracy-vs-BER results for `binary`, selected fixed mixed levels, and
   `adaptive unary` under the same evaluation protocol for the main
   confirmatory track.
2. The best adaptive policy at `BER = 1e-2`, including its per-group level
   assignment or compact boundary representation.
3. Encoded-width overhead of each candidate policy, especially the winning one.
4. Effective-capacity comparison against binary and strong-ECC reference points.
5. Hardware proxy comparison for candidate levels and the selected adaptive
   policy.
6. Margin-to-threshold numbers showing whether the `<= 5%` target is passed or
   failed for each benchmark and BER point.
7. Sensitivity-distribution evidence showing that heterogeneous group importance
   is strong enough to justify adaptive mapping over one global level.
8. A pilot comparison against heavier unary expansion, including full unary
   where that endpoint is still tractable.

### 8.2 Interpretation Data Still Needed

1. A robustness-overhead tradeoff plot that supports the claim of a practical
   sweet spot between pure binary and heavier unary expansion.
2. A mapping from robustness gain to retry/reclaim interpretation, at least as
   a modeled system-level discussion.
3. If stronger-ECC comparison remains in the main body, a fair comparison point
   that includes capacity loss and hardware proxy cost, not accuracy only.

### 8.3 Data That Can Stay Optional or Appendix-Level

1. Full controller-level reclaim simulation.
2. Correlated-fault or burst-error experiments.
3. Post-synthesis unary codec area, power, and timing numbers.

These are valuable, but they are not required for the minimum viable paper if
the main robustness-overhead story is otherwise well supported.

## 10. Required Figure Inventory

The paper should explicitly plan figures before the data is complete.

### Figure F1. Problem Framing Diagram

Goal:

- show the harsh-TLC dilemma:
  stronger ECC cost vs weaker-protection retry/reclaim burden vs unary-based
  mitigation opportunity

Data dependency:

- mostly conceptual
- can be drafted now

### Figure F2. Representation Illustration

Goal:

- show binary, mixed, and full unary examples for one `W = 6` pilot weight
- visualize how one encoded-bit flip maps to a decoded integer perturbation

Data dependency:

- none beyond the current codec definition
- can be drafted now

### Figure F3. Analytic Perturbation Comparison

Goal:

- compare binary `4^k` intuition with representation-aware unary decoded error
- show why codec-aware modeling is necessary

Data dependency:

- current formulas are enough for a first draft
- optional small numeric examples can be generated later

### Figure F4. Accuracy vs BER

Goal:

- compare the chosen baseline family over BER; in the main body this should use
  the int8 confirmatory set, while pilot full-unary results can move to the
  appendix

Required data:

- actual evaluation results across the chosen benchmarks and BER sweep

### Figure F5. Accuracy vs Overhead at `BER = 1e-2`

Goal:

- show whether a practical sweet spot exists

Required data:

- policy-level accuracy at `1e-2`
- `Omega`
- optionally hardware proxy cost for marker size or annotation

### Figure F6. Retry / Reclaim Interpretation Plot

Goal:

- connect robustness improvement to the system-level motivation

Required data:

- best available robustness results
- modeled retry/reclaim interpretation, not necessarily full simulation

## 11. Required Table Inventory

### Table T1. Prior Flash-LLM Operating Assumptions

Rows:

- Lincoln
- AiF
- Cambricon-LLM

Columns:

- cell mode / placement assumption
- BER regime
- ECC usage
- why it does not fully cover harsh TLC

### Table T2. Locked System Assumptions and Derived Numbers

Rows should include:

- SSD organization
- reclaim threshold
- endurance budget
- throughput target
- lifetime target
- hot-block lifetime token budget
- required service token count
- retry timing numbers

### Table T3. Representation Levels and Cost Proxies

Rows:

- `u = 0, 2, 3, 4, 5, 6`

Columns:

- mode
- encoded width
- storage ratio
- unary length
- hardware proxy counts

### Table T4. Evaluation Baselines

Rows:

- binary
- fixed mixed levels
- full unary where empirically tractable
- adaptive unary
- optional stronger-ECC reference

Columns:

- representation type
- overhead metric
- expected role in comparison

### Table T5. Main Result Summary

Columns:

- policy
- BER point
- mean accuracy
- accuracy drop
- pass/fail against `<= 5%`
- `Omega`
- optional hardware proxy

This table cannot be completed yet and is a required Phase 4 artifact.

## 12. Required Equation Inventory For The Paper Body

The final paper method should keep at least the following equations in the main
body.

1. Binary exposure:
   `e_w(k) = (1/2) alpha h s^2 4^k`
2. Supporting ECC residual-risk view:
   `R_b(m) = sum_k E_b(k) p_k^m`
3. Encoded width:
   `m(u)`
4. Mixed encode/decode:
   `C_u(q)` and `D_u(y)`
5. Representation-aware perturbation:
   `Delta_u(q, xi)`
6. Expected per-weight risk:
   `r_w(u, p)`
7. Adaptive policy boundary definition:
   `u_{pi(r)}`
8. Storage-overhead ratio:
   `Omega({u_g})`
9. Pass/fail rule:
   `bar{A}_t({u_g}; beta) >= (1 - tau) A_t^base`

If the page budget is tight, equations `1` and `2` can be compressed into
background, but equations `3` through `9` should stay visible in the main
method section.

## 13. Current Placeholders and Writing Rules

When this document is converted into paper prose:

- do not write result-dependent claims in definitive language
- mark any mention of the best policy as a placeholder until the data exists
- keep retry/reclaim impact framed as a system interpretation unless full
  simulation is added
- if a figure or table depends on missing data, keep the slot in the outline
  instead of silently dropping it

## Next Document To Write

`paper_experiment_matrix.md`

It should convert the evidence inventory above into an exact run plan:

- model list
- BER points
- baselines
- tasks and seeds
- metrics
- pass/fail rules
- figure/table data sources
