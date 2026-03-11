# Workspace Handoff

This repository is the upper-level workspace for the current quantization,
bit-flip, and paper-tracking work.

## Recommended Transfer Structure

- keep this upper repository for:
  - paper notes
  - top-level scripts
  - shared utils
  - experiment methodology code
- keep `adaptive_unary_level_mapping/` as its own GitHub-backed submodule

This avoids duplicating the adaptive-search codebase while still moving the full
workspace context.

## Included In This Workspace Repo

- top-level `paper_*.md`, `proposal*.md`, `overview.md`, `session_bridge.md`
- top-level scripts and utils for bit-flip / ECC studies
- selected reports and references
- `adaptive_unary_level_mapping/` as a submodule
- `lm-evaluation-harness/` as a submodule

## Excluded On Purpose

Generated artifacts are not meant to be versioned here:

- `cache/`
- `logs/`
- `er_coding/runs/`

If another PC also needs previous run artifacts or cached Hessians, copy those
directories separately outside Git.

## Submodule Notes

`adaptive_unary_level_mapping/` points to:

- repo: `https://github.com/jwjeong92/adaptive_unary_level_mapping.git`
- branch hint: `wip/surrogate-search-handoff-20260311`

That nested repo contains the current surrogate-search implementation and its
compact active docs.

## Clone / Restore

After this upper repository is pushed to GitHub, restore it like this:

```bash
git clone <upper-repo-url>
cd quant_analysis
git submodule update --init --recursive
cd adaptive_unary_level_mapping
git switch wip/surrogate-search-handoff-20260311
```

## Current Active Docs

For the adaptive-search thread, start here:

- `adaptive_unary_level_mapping/docs/active_todo.md`
- `adaptive_unary_level_mapping/docs/paper_execution_guardrails.md`
- `adaptive_unary_level_mapping/docs/work_handoff_20260311.md`

For the upper workspace thread, the main notes are:

- `paper_plan.md`
- `paper_method.md`
- `paper_experiment_matrix.md`
- `paper_search_strategy_note.md`
- `TODO.md`
