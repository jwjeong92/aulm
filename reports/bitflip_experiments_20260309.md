# Bit-Flip Experiments Summary (2026-03-09)

## 1) Code Changes

### RTN + act_order support
- Added Hessian-diagonal collection for RTN path and enabled RTN `act_order` based column permutation.
- Files:
  - `lib/quantization/weight_quant.py`
  - `main.py`

### New experiment entrypoint
- Added standalone script:
  - `scripts/bitflip_hypothesis_experiment.py`
- Supported modes:
  - `single_flip`: one-weight one-bit flip trials
  - `layer_full_lsb`: for each decoder layer, flip LSB of **all** weights in all linear modules in that decoder layer
- Added options:
  - `--bit_mode {random,msb}`
  - `--sampling_strategy {uniform_flat,stratified_layer}`
  - `--baseline_repeats`
  - `--experiment_mode {single_flip,layer_full_lsb}`

### Runner script
- Added RTN+act_order runner:
  - `scripts/run_rtn_actorder.sh`

---

## 2) Experiment Outputs

### A. Initial single-flip run (insufficient signal)
- Summary: `cache/bitflip_hypothesis/20260309_041637_summary.json`
- Key outcome:
  - Correlation metrics were `NaN` because `corr_trials` deltas were all zero.

### B. Improved single-flip run (stratified + larger eval)
- Summary: `cache/bitflip_hypothesis/20260309_042829_summary.json`
- Trials:
  - `cache/bitflip_hypothesis/20260309_042829_corr_trials.csv`
  - `cache/bitflip_hypothesis/20260309_042829_group_trials.csv`

#### Config highlights
- `eval_nsamples=64`, `corr_trials=512`, `sampling_strategy=stratified_layer`
- `baseline_repeats=10`

#### Key results
- Noise floor:
  - baseline repeated losses were identical (`std=0`, `max_abs_step_delta=0`)
- Correlation:
  - `score vs delta_loss` is near zero (sign prediction weak)
  - `score vs abs(delta_loss)` is positive:
    - pearson: `0.1153`
    - spearman: `0.1106`
- Group comparison (`abs(delta_loss)`):
  - top-score: `0.001374`
  - random: `0.0000353`
  - bottom-score: `0.00000381`
  - Direction is consistent with hypothesis (`top > random > bottom`).

#### Interpretation
- The score behaves as a **magnitude predictor** (damage size), not a sign predictor.
- This matches the quadratic-form intuition: `hessian * (delta_w^2)`.

### C. Decoder-layer full LSB sweep (requested methodology)
- Summary: `cache/bitflip_hypothesis/20260309_123958_summary.json`
- Trials CSV:
  - `cache/bitflip_hypothesis/20260309_123958_layer_full_lsb_trials.csv`

#### Method
- Baseline: RTN quantized model
- For each decoder layer `i` (0~11):
  - Flip LSB for all weights in all linear modules in decoder layer `i`
  - Measure:
    - `delta_ppl`
    - decoder-layer output NMSE (`nmse_decoder_layer_j`)
    - mean NMSE across decoder layers

#### Global summary
- Baseline PPL: `56.2248`
- Mean `delta_ppl`: `+1801.6442`
- Most sensitive by `delta_ppl`: layer `2` (`+5312.0230`)
- Least sensitive by `delta_ppl`: layer `9` (`+61.5457`)
- `corr(delta_ppl, mean_decoder_nmse)` ≈ `0.659`

#### Per-layer `delta_ppl` (injected layer -> delta_ppl)
- `0 -> +4167.1452`
- `1 -> +4822.6787`
- `2 -> +5312.0230`
- `3 -> +963.7684`
- `4 -> +710.3304`
- `5 -> +1563.2085`
- `6 -> +1224.1488`
- `7 -> +920.5111`
- `8 -> +734.4704`
- `9 -> +61.5457`
- `10 -> +146.7682`
- `11 -> +993.1319`

#### Interpretation
- Strong sensitivity differences exist across decoder layers under full-LSB corruption.
- Early/mid layers (0,1,2) were highly sensitive in this run.
- Layer-output distortion (mean NMSE) and end-task degradation (`delta_ppl`) are positively related.

---

## 3) Notes and Caveats
- `layer_full_lsb` is a stress test and may be harsher than realistic bit-flip rates.
- For statistical robustness, repeat with multiple seeds and report mean/std across seeds.
- For deployment realism, add sparse corruption settings (`fixed K` or `flip ratio`) per layer.

---

## 4) Repro Commands

### single_flip
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bitflip_hypothesis_experiment.py \
  --model_path /raid/LLM/opt-125m \
  --experiment_mode single_flip \
  --bits_w 4 --sym_w false --groupsize_w -1 \
  --hessian_on fp \
  --eval_dataset wikitext2 --eval_seqlen 512 --eval_nsamples 64 \
  --corr_trials 512 \
  --sampling_strategy stratified_layer \
  --baseline_repeats 10 \
  --out_dir ./cache/bitflip_hypothesis
```

### layer_full_lsb
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bitflip_hypothesis_experiment.py \
  --model_path /raid/LLM/opt-125m \
  --experiment_mode layer_full_lsb \
  --bits_w 4 --sym_w false --groupsize_w -1 \
  --eval_dataset wikitext2 --eval_seqlen 512 --eval_nsamples 64 \
  --baseline_repeats 10 \
  --out_dir ./cache/bitflip_hypothesis
```
