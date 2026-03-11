# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the root entry point for model loading, quantization, bit-flip injection, perplexity evaluation, and `lm_eval` runs. Core implementations live in `lib/quantization/`, `lib/gptq/`, and `lib/smoothquant/`; shared helpers are in `utils/`. Reproducible experiment runners live in `scripts/` and are named by method or study, for example `scripts/rtn_int8_group_budget.py`.

`adaptive_unary_level_mapping/` is a separate experimental pipeline for adaptive unary coding search, with its own `main.py`, `configs/`, `lib/`, and `scripts/`. `lm-evaluation-harness/` is a vendored submodule with its own tests and lint configuration. Research notes and outputs live in `reports/`, `refs/`, and top-level `paper_*.md` files. Generated artifacts belong in `cache/` and `logs/` and should not be committed.

## Build, Test, and Development Commands
`bash scripts/run.sh 0 <hf-model>` runs the main quantization pipeline on GPU 0.

`bash scripts/run_gptq.sh 0 <hf-model>` runs the GPTQ-based path with cached checkpoints and bit-flip injection.

`python adaptive_unary_level_mapping/main.py prebuild-cuda` builds the adaptive search CUDA extension.

`python adaptive_unary_level_mapping/main.py adaptive-search --config adaptive_unary_level_mapping/configs/opt125m_rtn6_piqa_search.yaml` runs the adaptive search workflow.

`pytest lm-evaluation-harness/tests -q` runs the harness test suite; use `-k <pattern>` for targeted checks.

## Coding Style & Naming Conventions
Use Python with 4-space indentation, snake_case for modules/functions, and descriptive long-form CLI flags such as `--gptq_act_order`. Follow existing script naming: lowercase, underscore-separated, method-first names. Prefer `logging` for long-running experiments so results land in log files; keep comments brief and only where control flow is non-obvious. For changes inside `lm-evaluation-harness/`, follow its local `pyproject.toml` and run `ruff check lm-evaluation-harness`.

## Testing Guidelines
There is no root-level pytest suite or coverage gate yet. For root or adaptive pipeline changes, run a smoke test with a small model and constrained evaluation, for example `python main.py --model_path facebook/opt-125m --tasks piqa --limit 10 --eval_ppl false`. For `lm-evaluation-harness/` changes, add or update `tests/test_*.py` cases and run the relevant `pytest` target.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects, for example `Add GPTQ checkpoint cache and tagged logging`. Keep commits focused on one feature or experiment. In pull requests, include the exact reproduction command, model/task/seed settings, expected output paths, and any metric deltas. If a change touches generated figures or reports, link the updated artifact path instead of committing caches or checkpoints. Never commit `cache/`, `logs/`, model weights, or Hugging Face tokens.
