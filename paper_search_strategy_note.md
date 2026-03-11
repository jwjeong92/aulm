# Paper Search Strategy Note

Date: 2026-03-11
Workspace: `/home/jwjeong/study/quant_analysis`
Status: active handoff note

## Purpose

This note captures the current decision about the unary-policy search strategy.

It exists because:

- the current implementation already produced the key proof-of-possibility
  signal
- but the current design-space exploration loop is too slow to be the main
  paper path

## Current Implementation Status

The current implementation has already answered one important question:

- it produced empirical evidence that some unary-based policies can improve
  robustness without exploding space overhead

So the current code is still valuable.

However, it should be treated as:

- a proof-of-possibility implementation
- a verification path
- a fallback baseline

and not as:

- the final search strategy for large-scale design-space exploration

## Why The Current Search Is Too Slow

The current search in `adaptive_unary_level_mapping/lib/search.py` does this:

1. rank groups by importance
2. move level boundaries greedily using `step_pct`
3. evaluate each candidate with BER injection
4. run `lm_eval`
5. repeat across BER points and seeds

The expensive part is not the ranking itself.

The expensive part is:

- repeated empirical candidate evaluation
- where each candidate may require
  `BER x seed x task x full model eval`

This makes the current path too expensive for broad search, especially once the
paper's main evidence moves to int8 `7B` settings.

## Agreed Decision

We should not treat the current global empirical greedy search as the default
long-term solution.

The current search path is now demoted to:

- finalist verification
- local refinement around a small number of candidates
- fallback if the faster path fails

## Current Interpretation of Risk

The analytic risk language remains useful, but not as the exact implemented
search objective.

Current role of risk:

- define representation-aware damage correctly
- explain why unary level changes robustness and overhead
- justify sensitivity-aware search
- provide a future surrogate objective

Current role of search:

- importance-ordered greedy boundary search
- empirical BER injection
- `lm_eval` verification

So, for the current codebase:

- `risk` is an analytic support / surrogate concept
- `search` is an empirical greedy verification loop

These two should not be described as the same thing.

## Proposed New Direction

The agreed next direction is:

- `surrogate-guided search`

The intended structure is:

1. compute a fast group-level score for each candidate unary level
2. generate a small number of candidate policies from that surrogate
3. run full empirical verification only on the finalists
4. optionally use the old greedy search only as local refinement

## Candidate Surrogate Form

The current working idea is:

- keep the existing group importance signal `I_g`
- add a fast representation-dependent distortion term
- combine them into a level score

Conceptually:

$$
S_g(u) \approx I_g \cdot D_g(u)
$$

or, with BER weighting:

$$
S_g(u) \approx I_g \sum_{\beta \in \mathcal B} \omega_\beta D_g(u, \beta)
$$

where:

- `I_g` captures model-side importance
- `D_g(u, beta)` captures expected decoded perturbation under level `u`
- `omega_beta` lets us emphasize the target regime such as `BER = 1e-2`

The exact formula is not frozen yet.

## Proposed Search Flow

The intended search flow is:

1. compute `I_g`
2. estimate fast per-level distortion summaries
3. build a surrogate score table over `(g, u)`
4. solve for one or a few low-cost candidate policies
5. run `lm_eval` only on:
   - binary
   - a few fixed mixed baselines
   - the best surrogate policy
   - one or two neighboring policies
6. if needed, apply local greedy refinement around the best finalist

## What Probably Stays

These parts are still likely useful:

- group importance computation in
  `adaptive_unary_level_mapping/lib/importance.py`
- policy overhead accounting in
  `adaptive_unary_level_mapping/lib/error_injection.py`
- empirical evaluation path in
  `adaptive_unary_level_mapping/lib/search.py`
  as a finalist verifier

## What Likely Changes

These parts are the expected redesign targets:

- the global search loop in
  `adaptive_unary_level_mapping/lib/search.py`
- the meaning of `step_pct` / `rollback_steps`
  from main search controls to optional local refinement controls
- candidate generation so it no longer depends on repeated full `lm_eval`
  during broad exploration

## Open Technical Decisions

These points are still unresolved:

1. how to estimate `D_g(u, beta)` efficiently
   - exact sampled codec perturbation
   - histogram-based approximation
   - single-bit / low-order approximation
2. whether the search should remain boundary-constrained over ordered groups
   or allow more flexible policy classes
3. how to penalize overhead
   - hard budget
   - Lagrangian penalty
   - Pareto candidate generation
4. whether the first prototype should target
   - `6-bit` pilot only
   - or directly `int8` confirmatory settings

## Current Path

The current path is now:

1. keep the paper story and experiment matrix as they are
2. pause large-scale evidence execution
3. redesign the search path first
4. resume the frozen experiment matrix after the faster search path exists

In other words:

- the experiment matrix is still valid
- the execution order is temporarily paused
- the next engineering task is search redesign

## Immediate Next Task

The next session should start by designing and implementing a first
`surrogate-guided search` prototype.

The old greedy empirical path should remain available, but only as:

- finalist verification
- local refinement
- fallback
