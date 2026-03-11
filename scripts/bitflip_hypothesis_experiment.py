#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.quantization.quantizer import Quantizer, quantize_to_int
from lib.quantization.weight_quant import (
    collect_rtn_hessian_diagonal,
    iter_quantized_linears,
    quantize_nearest,
)
from lib.gptq.modelutils import find_layers
from utils.common import set_seed, str2bool
from utils.data_utils import get_test_tokens
from utils.import_model import model_from_hf_path


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)

    sorted_vals = values[order]
    change = np.r_[True, sorted_vals[1:] != sorted_vals[:-1], True]
    idx = np.flatnonzero(change)
    for start, end in zip(idx[:-1], idx[1:]):
        if end - start > 1:
            avg_rank = 0.5 * (start + end - 1)
            ranks[order[start:end]] = avg_rank
    return ranks


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def spearman_corr(x, y):
    return pearson_corr(rankdata(x), rankdata(y))


def build_eval_inputs(args):
    toks = get_test_tokens(
        args.eval_dataset,
        seed=args.seed,
        seqlen=args.eval_seqlen,
        model=args.model_path,
        cache_dir=args.cache_dir,
    )
    total = toks.numel() // args.eval_seqlen
    n = min(args.eval_nsamples, total)
    if n <= 0:
        raise ValueError("eval_nsamples is too small for the chosen dataset/seqlen.")
    toks = toks[0, : n * args.eval_seqlen].view(n, args.eval_seqlen)
    return [toks[i, :].cuda().view(1, -1) for i in range(n)]


@torch.no_grad()
def eval_avg_loss(model, eval_inputs):
    loss_fct = torch.nn.CrossEntropyLoss().cuda()
    acc = 0.0
    for inp in eval_inputs:
        out = model(
            inp,
            use_cache=False,
            output_hidden_states=False,
            output_attentions=False,
        )[0]
        shift_logits = out[:, :-1, :].contiguous()
        shift_labels = inp[:, 1:]
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        acc += float(loss.item())
    return acc / len(eval_inputs)


def eval_noise_floor(model, eval_inputs, repeats):
    if repeats <= 0:
        return {
            "repeats": 0,
            "losses": [],
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "max_abs_step_delta": float("nan"),
        }

    losses = [eval_avg_loss(model, eval_inputs) for _ in range(repeats)]
    arr = np.asarray(losses, dtype=np.float64)
    if len(arr) > 1:
        step = np.diff(arr)
        max_abs_step = float(np.max(np.abs(step)))
    else:
        max_abs_step = 0.0

    return {
        "repeats": int(repeats),
        "losses": [float(x) for x in losses],
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "max_abs_step_delta": max_abs_step,
    }


def get_decoder_layers(model, args):
    if "llama" in args.model_path:
        return model.model.layers
    if "opt" in args.model_path:
        return model.model.decoder.layers
    raise NotImplementedError(f"Unsupported model path for decoder layers: {args.model_path}")


def get_decoder_layer_linears(model, args, target_layer_idx):
    decoder_layers = get_decoder_layers(model, args)
    target_layer = decoder_layers[target_layer_idx]
    subset = find_layers(target_layer)
    if "llama" in args.model_path:
        prefix = f"model.layers.{target_layer_idx}"
    else:
        prefix = f"model.decoder.layers.{target_layer_idx}"

    linears = []
    for local_name, module in subset.items():
        full_name = f"{prefix}.{local_name}" if local_name else prefix
        linears.append((full_name, module))
    return linears


@torch.no_grad()
def collect_decoder_outputs_cache(model, eval_inputs, decoder_layers, cache_dtype=torch.float16):
    n_layers = len(decoder_layers)
    cache = []

    def make_hook(i, bucket):
        def hook(_, inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            bucket[i] = tensor.detach().to(cache_dtype).cpu()
        return hook

    for inp in eval_inputs:
        bucket = [None] * n_layers
        handles = []
        for i, layer in enumerate(decoder_layers):
            handles.append(layer.register_forward_hook(make_hook(i, bucket)))
        try:
            model(
                inp,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
            )
        finally:
            for h in handles:
                h.remove()

        if any(x is None for x in bucket):
            raise RuntimeError("Failed to collect all decoder layer outputs.")
        cache.append(bucket)
    return cache


@torch.no_grad()
def eval_loss_and_decoder_distortion(model, eval_inputs, decoder_layers, baseline_cache):
    n_layers = len(decoder_layers)
    numerator = np.zeros((n_layers,), dtype=np.float64)
    denominator = np.zeros((n_layers,), dtype=np.float64)
    element_count = np.zeros((n_layers,), dtype=np.float64)
    loss_fct = torch.nn.CrossEntropyLoss().cuda()
    acc_loss = 0.0

    for sample_idx, inp in enumerate(eval_inputs):
        bucket = [None] * n_layers

        def make_hook(i):
            def hook(_, inpt, out):
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                bucket[i] = tensor.detach().to(torch.float32).cpu()
            return hook

        handles = []
        for i, layer in enumerate(decoder_layers):
            handles.append(layer.register_forward_hook(make_hook(i)))
        try:
            output = model(
                inp,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
            )[0]
        finally:
            for h in handles:
                h.remove()

        shift_logits = output[:, :-1, :].contiguous()
        shift_labels = inp[:, 1:]
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        acc_loss += float(loss.item())

        for i in range(n_layers):
            cur = bucket[i]
            base = baseline_cache[sample_idx][i].to(torch.float32)
            if cur is None:
                raise RuntimeError(f"Missing decoder output for layer {i}, sample {sample_idx}.")
            diff = cur - base
            numerator[i] += float(torch.sum(diff * diff).item())
            denominator[i] += float(torch.sum(base * base).item())
            element_count[i] += float(diff.numel())

    mse = numerator / np.maximum(element_count, 1.0)
    nmse = numerator / np.maximum(denominator, 1e-12)
    avg_loss = acc_loss / len(eval_inputs)
    return {
        "loss": float(avg_loss),
        "mse": mse.tolist(),
        "mean_mse": float(np.mean(mse)),
        "nmse": nmse.tolist(),
        "mean_nmse": float(np.mean(nmse)),
    }


@torch.no_grad()
def apply_full_lsb_flip_to_modules(modules, args):
    backups = {}
    flipped_weights = 0

    for full_name, module in modules:
        weight = module.weight.data
        backups[full_name] = weight.detach().clone()
        shape = weight.shape
        quant_input = weight

        if args.groupsize_w > 0:
            if quant_input.shape[-1] % args.groupsize_w != 0:
                raise ValueError(
                    f"groupsize_w mismatch for {full_name}: "
                    f"last_dim={quant_input.shape[-1]}, groupsize_w={args.groupsize_w}"
                )
            quant_input = quant_input.reshape(-1, args.groupsize_w)

        quantizer = Quantizer()
        quantizer.configure(args.bits_w, perchannel=True, sym=args.sym_w, mse=False)
        quantizer.find_params(quant_input, weight=True)
        qint = quantize_to_int(quant_input, quantizer.scale, quantizer.zero, quantizer.maxq)
        qint = torch.bitwise_xor(qint, torch.ones_like(qint, dtype=torch.int32))
        qW = quantizer.scale * (qint.to(quantizer.scale.dtype) - quantizer.zero)
        module.weight.data = qW.reshape(shape).to(module.weight.data.dtype)
        flipped_weights += int(weight.numel())

    return backups, flipped_weights


@torch.no_grad()
def restore_modules_from_backups(modules, backups):
    for full_name, module in modules:
        module.weight.data.copy_(backups[full_name])


def build_layer_infos(model, args, hessian_diagonal):
    infos = []
    total_numel = 0

    for layer_name, linear in iter_quantized_linears(model, args):
        if layer_name not in hessian_diagonal:
            continue

        W = linear.weight.data.detach()
        if W.dim() != 2:
            logging.warning(f"Skip non-2D weight: {layer_name}, shape={tuple(W.shape)}")
            continue

        rows, cols = W.shape
        hdiag = hessian_diagonal[layer_name]
        if hdiag.numel() != cols:
            logging.warning(
                f"Skip Hessian mismatch: {layer_name}, hdiag={hdiag.numel()}, cols={cols}"
            )
            continue

        quantizer = Quantizer()
        quantizer.configure(args.bits_w, perchannel=True, sym=args.sym_w, mse=False)

        quant_input = W
        if args.groupsize_w > 0:
            if cols % args.groupsize_w != 0:
                logging.warning(
                    f"Skip layer due to groupsize mismatch: {layer_name}, cols={cols}, groupsize={args.groupsize_w}"
                )
                continue
            quant_input = quant_input.reshape(-1, args.groupsize_w)

        quantizer.find_params(quant_input, weight=True)
        qint = quantize_to_int(
            quant_input, quantizer.scale, quantizer.zero, quantizer.maxq
        ).reshape(rows, cols)

        info = {
            "layer_name": layer_name,
            "module": linear,
            "rows": rows,
            "cols": cols,
            "numel": rows * cols,
            "hdiag": hdiag.detach().cpu().to(torch.float32),
            "qint": qint.detach().cpu().to(torch.int32),
        }

        if args.groupsize_w > 0:
            num_groups = cols // args.groupsize_w
            info["num_groups"] = num_groups
            info["scale_vec"] = quantizer.scale.detach().view(-1).cpu().to(torch.float32)
            info["zero_vec"] = quantizer.zero.detach().view(-1).cpu().to(torch.float32)
        else:
            info["scale_vec"] = quantizer.scale.detach().view(-1).cpu().to(torch.float32)
            info["zero_vec"] = quantizer.zero.detach().view(-1).cpu().to(torch.float32)

        infos.append(info)
        total_numel += rows * cols

    if not infos:
        raise RuntimeError("No eligible 2D linear layers found for bit-flip experiment.")
    return infos, total_numel


def pick_layer_and_coord(infos, total_numel, rng):
    flat = rng.randrange(total_numel)
    base = 0
    for idx, info in enumerate(infos):
        nxt = base + info["numel"]
        if flat < nxt:
            local = flat - base
            row = local // info["cols"]
            col = local % info["cols"]
            return idx, row, col
        base = nxt
    raise RuntimeError("Failed to pick layer index.")


def get_scale_zero(info, row, col, groupsize_w):
    if groupsize_w > 0:
        group_idx = row * info["num_groups"] + (col // groupsize_w)
        scale = float(info["scale_vec"][group_idx].item())
        zero = float(info["zero_vec"][group_idx].item())
    else:
        scale = float(info["scale_vec"][row].item())
        zero = float(info["zero_vec"][row].item())
    return scale, zero


def pick_bit(args, rng):
    if args.bit_mode == "msb":
        return args.bits_w - 1
    return rng.randrange(args.bits_w)


def make_candidate_from_coord(infos, args, layer_idx, row, col, bit):
    info = infos[layer_idx]
    mask = 1 << bit

    q_old = int(info["qint"][row, col].item())
    q_new = q_old ^ mask
    delta_q = q_new - q_old

    scale, zero = get_scale_zero(info, row, col, args.groupsize_w)
    h = float(info["hdiag"][col].item())
    delta_w = scale * float(delta_q)
    score = h * (delta_w * delta_w)

    return {
        "layer_idx": layer_idx,
        "layer_name": info["layer_name"],
        "row": int(row),
        "col": int(col),
        "bit": int(bit),
        "mask": int(mask),
        "q_old": int(q_old),
        "q_new": int(q_new),
        "delta_q": int(delta_q),
        "scale": float(scale),
        "zero": float(zero),
        "hessian_diag": float(h),
        "score": float(score),
        "bit_factor_4b": float(4**bit),
    }


def make_candidate(infos, total_numel, args, rng):
    layer_idx, row, col = pick_layer_and_coord(infos, total_numel, rng)
    bit = pick_bit(args, rng)
    return make_candidate_from_coord(infos, args, layer_idx, row, col, bit)


def make_candidate_stratified_layer(infos, args, rng, layer_idx):
    info = infos[layer_idx]
    row = rng.randrange(info["rows"])
    col = rng.randrange(info["cols"])
    bit = pick_bit(args, rng)
    return make_candidate_from_coord(infos, args, layer_idx, row, col, bit)


def generate_candidates(infos, total_numel, args, rng, n):
    if n <= 0:
        return []

    strategy = args.sampling_strategy
    if strategy == "uniform_flat":
        return [make_candidate(infos, total_numel, args, rng) for _ in range(n)]

    if strategy == "stratified_layer":
        layer_ids = list(range(len(infos)))
        rng.shuffle(layer_ids)
        candidates = []
        for t in range(n):
            lid = layer_ids[t % len(layer_ids)]
            if (t + 1) % len(layer_ids) == 0:
                rng.shuffle(layer_ids)
            candidates.append(make_candidate_stratified_layer(infos, args, rng, lid))
        return candidates

    raise ValueError(f"Unknown sampling strategy: {strategy}")


@torch.no_grad()
def run_single_flip_trial(model, infos, candidate, baseline_loss):
    info = infos[candidate["layer_idx"]]
    module = info["module"]
    row = candidate["row"]
    col = candidate["col"]
    q_new = candidate["q_new"]
    scale = candidate["scale"]
    zero = candidate["zero"]

    weight = module.weight.data
    old_val = weight[row, col].detach().clone()
    new_val = scale * (float(q_new) - zero)
    weight[row, col] = torch.tensor(new_val, device=weight.device, dtype=weight.dtype)

    return old_val


@torch.no_grad()
def restore_single_flip(model, infos, candidate, old_val):
    info = infos[candidate["layer_idx"]]
    module = info["module"]
    row = candidate["row"]
    col = candidate["col"]
    module.weight.data[row, col] = old_val


def evaluate_candidates(model, infos, eval_inputs, baseline_loss, candidates):
    rows = []
    for idx, cand in enumerate(candidates):
        old_val = run_single_flip_trial(model, infos, cand, baseline_loss)
        try:
            loss = eval_avg_loss(model, eval_inputs)
            delta = loss - baseline_loss
        finally:
            restore_single_flip(model, infos, cand, old_val)

        row = dict(cand)
        row["trial_idx"] = idx
        row["loss"] = float(loss)
        row["delta_loss"] = float(delta)
        row["abs_delta_loss"] = float(abs(delta))
        rows.append(row)
    return rows


def summarize_correlations(rows):
    pred = np.array([r["score"] for r in rows], dtype=np.float64)
    delta = np.array([r["delta_loss"] for r in rows], dtype=np.float64)
    abs_delta = np.array([abs(r["delta_loss"]) for r in rows], dtype=np.float64)
    h = np.array([r["hessian_diag"] for r in rows], dtype=np.float64)
    s2 = np.array([(r["scale"] ** 2) for r in rows], dtype=np.float64)
    b = np.array([r["bit_factor_4b"] for r in rows], dtype=np.float64)

    summary = {
        "n_trials": int(len(rows)),
        "n_nonzero_delta": int(np.count_nonzero(delta)),
        "n_nonzero_abs_delta": int(np.count_nonzero(abs_delta)),
        "pearson_score_vs_delta": pearson_corr(pred, delta),
        "spearman_score_vs_delta": spearman_corr(pred, delta),
        "pearson_score_vs_abs_delta": pearson_corr(pred, abs_delta),
        "spearman_score_vs_abs_delta": spearman_corr(pred, abs_delta),
        "spearman_hessian_vs_delta": spearman_corr(h, delta),
        "spearman_hessian_vs_abs_delta": spearman_corr(h, abs_delta),
        "spearman_scale2_vs_delta": spearman_corr(s2, delta),
        "spearman_scale2_vs_abs_delta": spearman_corr(s2, abs_delta),
        "spearman_bitfactor_vs_delta": spearman_corr(b, delta),
        "spearman_bitfactor_vs_abs_delta": spearman_corr(b, abs_delta),
    }
    return summary


def select_group_candidates(pool, group_size, seed):
    pool_sorted = sorted(pool, key=lambda x: x["score"], reverse=True)
    k = min(group_size, len(pool_sorted))
    top = pool_sorted[:k]
    bottom = pool_sorted[-k:]

    rng = random.Random(seed)
    random_group = list(pool)
    rng.shuffle(random_group)
    random_group = random_group[:k]
    return top, random_group, bottom


def save_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def extract_decoder_metric_array(row, key_prefix):
    values = []
    idx = 0
    while True:
        key = f"{key_prefix}_{idx}"
        if key not in row:
            break
        values.append(float(row[key]))
        idx += 1
    return np.asarray(values, dtype=np.float64)


def add_engineered_decoder_features(row):
    target_idx = int(row["target_decoder_layer"])
    mse = extract_decoder_metric_array(row, "mse_decoder_layer")
    nmse = extract_decoder_metric_array(row, "nmse_decoder_layer")
    if mse.size == 0 or nmse.size == 0:
        return

    downstream_mse = mse[target_idx:]
    downstream_nmse = nmse[target_idx:]
    rel_weights = np.arange(1, downstream_mse.size + 1, dtype=np.float64)
    depth_weights = np.arange(1, mse.size + 1, dtype=np.float64)

    row.update(
        {
            "last_decoder_mse": float(mse[-1]),
            "last_decoder_nmse": float(nmse[-1]),
            "max_decoder_mse": float(np.max(mse)),
            "max_decoder_nmse": float(np.max(nmse)),
            "downstream_mean_decoder_mse": float(np.mean(downstream_mse)),
            "downstream_mean_decoder_nmse": float(np.mean(downstream_nmse)),
            "downstream_sum_decoder_mse": float(np.sum(downstream_mse)),
            "downstream_sum_decoder_nmse": float(np.sum(downstream_nmse)),
            "downstream_weighted_decoder_mse": float(np.sum(rel_weights * downstream_mse)),
            "downstream_weighted_decoder_nmse": float(np.sum(rel_weights * downstream_nmse)),
            "depth_weighted_decoder_mse": float(np.sum(depth_weights * mse)),
            "depth_weighted_decoder_nmse": float(np.sum(depth_weights * nmse)),
        }
    )


def fit_linear_1d(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0.0:
        return y_mean, 0.0

    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = float(y_mean - slope * x_mean)
    return intercept, slope


def loocv_linear_summary(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(x.size)
    if n <= 1:
        return {
            "n": n,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
        }

    preds = np.zeros((n,), dtype=np.float64)
    for idx in range(n):
        mask = np.ones((n,), dtype=bool)
        mask[idx] = False
        intercept, slope = fit_linear_1d(x[mask], y[mask])
        preds[idx] = intercept + slope * x[idx]

    residual = y - preds
    ss_res = float(np.sum(residual * residual))
    centered = y - float(np.mean(y))
    ss_tot = float(np.sum(centered * centered))
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
    }


def fit_linear_multi(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return float("nan"), np.array([], dtype=np.float64)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = np.column_stack([np.ones((y.size,), dtype=np.float64), x])
    coeffs = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coeffs[0]), coeffs[1:]


def loocv_linear_multi_summary(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n = int(y.size)
    if n <= 1:
        return {
            "n": n,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
        }

    preds = np.zeros((n,), dtype=np.float64)
    for idx in range(n):
        mask = np.ones((n,), dtype=bool)
        mask[idx] = False
        intercept, slopes = fit_linear_multi(x[mask], y[mask])
        preds[idx] = intercept + float(np.dot(x[idx], slopes))

    residual = y - preds
    ss_res = float(np.sum(residual * residual))
    centered = y - float(np.mean(y))
    ss_tot = float(np.sum(centered * centered))
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
    }


def summarize_scalar_predictor(rows, feature_key, target_key):
    x = np.array([row[feature_key] for row in rows], dtype=np.float64)
    y = np.array([row[target_key] for row in rows], dtype=np.float64)
    intercept, slope = fit_linear_1d(x, y)
    summary = {
        "feature_key": feature_key,
        "target_key": target_key,
        "n": int(len(rows)),
        "pearson": pearson_corr(x, y),
        "spearman": spearman_corr(x, y),
        "linear_intercept": float(intercept),
        "linear_slope": float(slope),
    }
    summary.update(loocv_linear_summary(x, y))
    return summary


def choose_better_predictor(lhs, rhs):
    lhs_r2 = lhs["r2"]
    rhs_r2 = rhs["r2"]
    if math.isnan(lhs_r2):
        return rhs
    if math.isnan(rhs_r2):
        return lhs
    return lhs if lhs_r2 >= rhs_r2 else rhs


def summarize_feature_set(rows, feature_keys, target_key):
    x = np.column_stack(
        [np.array([row[key] for row in rows], dtype=np.float64) for key in feature_keys]
    )
    y = np.array([row[target_key] for row in rows], dtype=np.float64)
    intercept, slopes = fit_linear_multi(x, y)
    summary = {
        "feature_keys": list(feature_keys),
        "target_key": target_key,
        "n": int(len(rows)),
        "linear_intercept": float(intercept),
        "linear_coefficients": {
            key: float(slopes[idx]) for idx, key in enumerate(feature_keys)
        },
    }
    summary.update(loocv_linear_multi_summary(x, y))
    return summary


def summarize_layer_full_lsb_predictors(rows):
    if not rows:
        return {}

    layer_count = len([k for k in rows[0].keys() if k.startswith("nmse_decoder_layer_")])
    summary = {}
    for target_key in ("delta_loss", "delta_ppl"):
        mean_nmse = summarize_scalar_predictor(rows, "mean_decoder_nmse", target_key)
        mean_mse = summarize_scalar_predictor(rows, "mean_decoder_mse", target_key)

        best_nmse = None
        best_mse = None
        for idx in range(layer_count):
            nmse_summary = summarize_scalar_predictor(
                rows, f"nmse_decoder_layer_{idx}", target_key
            )
            mse_summary = summarize_scalar_predictor(
                rows, f"mse_decoder_layer_{idx}", target_key
            )
            best_nmse = nmse_summary if best_nmse is None else choose_better_predictor(best_nmse, nmse_summary)
            best_mse = mse_summary if best_mse is None else choose_better_predictor(best_mse, mse_summary)

        summary[target_key] = {
            "mean_decoder_nmse": mean_nmse,
            "mean_decoder_mse": mean_mse,
            "best_single_nmse_layer": best_nmse,
            "best_single_mse_layer": best_mse,
        }
    return summary


def summarize_engineered_layer_predictors(rows):
    if not rows:
        return {}

    candidate_sets = {
        "target_only": ["target_decoder_layer"],
        "last_mse": ["last_decoder_mse"],
        "last_nmse": ["last_decoder_nmse"],
        "downstream_weighted_mse": ["downstream_weighted_decoder_mse"],
        "downstream_weighted_nmse": ["downstream_weighted_decoder_nmse"],
        "depth_weighted_mse": ["depth_weighted_decoder_mse"],
        "depth_weighted_nmse": ["depth_weighted_decoder_nmse"],
        "target_plus_last_mse": ["target_decoder_layer", "last_decoder_mse"],
        "target_plus_last_nmse": ["target_decoder_layer", "last_decoder_nmse"],
        "target_plus_downstream_weighted_mse": [
            "target_decoder_layer",
            "downstream_weighted_decoder_mse",
        ],
        "target_plus_downstream_weighted_nmse": [
            "target_decoder_layer",
            "downstream_weighted_decoder_nmse",
        ],
        "target_plus_last_plus_downstream_mse": [
            "target_decoder_layer",
            "last_decoder_mse",
            "downstream_weighted_decoder_mse",
        ],
        "target_plus_last_plus_downstream_nmse": [
            "target_decoder_layer",
            "last_decoder_nmse",
            "downstream_weighted_decoder_nmse",
        ],
    }

    summary = {}
    for target_key in ("delta_loss", "delta_ppl"):
        per_set = {}
        best = None
        for label, feature_keys in candidate_sets.items():
            result = summarize_feature_set(rows, feature_keys, target_key)
            per_set[label] = result
            best = result if best is None else choose_better_predictor(best, result)

        summary[target_key] = {
            "candidate_feature_sets": per_set,
            "best_feature_set": best,
        }
    return summary


def run_layer_full_lsb_sweep(model, args, eval_inputs, baseline_loss, baseline_ppl):
    decoder_layers = get_decoder_layers(model, args)
    n_layers = len(decoder_layers)

    logging.info(
        f"Collecting baseline decoder outputs cache for {n_layers} decoder layers x {len(eval_inputs)} samples..."
    )
    baseline_cache = collect_decoder_outputs_cache(
        model, eval_inputs, decoder_layers, cache_dtype=torch.float16
    )

    rows = []
    for target_idx in range(n_layers):
        modules = get_decoder_layer_linears(model, args, target_idx)
        backups, flipped_weights = apply_full_lsb_flip_to_modules(modules, args)
        try:
            distortion = eval_loss_and_decoder_distortion(
                model, eval_inputs, decoder_layers, baseline_cache
            )
        finally:
            restore_modules_from_backups(modules, backups)

        loss = distortion["loss"]
        ppl = math.exp(loss)
        row = {
            "target_decoder_layer": int(target_idx),
            "num_linear_modules": int(len(modules)),
            "num_flipped_weights": int(flipped_weights),
            "loss": float(loss),
            "delta_loss": float(loss - baseline_loss),
            "ppl": float(ppl),
            "delta_ppl": float(ppl - baseline_ppl),
            "mean_decoder_mse": float(distortion["mean_mse"]),
            "mean_decoder_nmse": float(distortion["mean_nmse"]),
        }
        for j, v in enumerate(distortion["mse"]):
            row[f"mse_decoder_layer_{j}"] = float(v)
        for j, v in enumerate(distortion["nmse"]):
            row[f"nmse_decoder_layer_{j}"] = float(v)
        add_engineered_decoder_features(row)
        rows.append(row)

        logging.info(
            f"[layer {target_idx}] delta_ppl={row['delta_ppl']:.6f}, "
            f"mean_decoder_mse={row['mean_decoder_mse']:.6e}, "
            f"mean_decoder_nmse={row['mean_decoder_nmse']:.6e}, "
            f"flipped_weights={flipped_weights}"
        )

    if not rows:
        raise RuntimeError("No rows produced for layer_full_lsb sweep.")

    delta_ppl_vals = np.array([r["delta_ppl"] for r in rows], dtype=np.float64)
    mean_mse_vals = np.array([r["mean_decoder_mse"] for r in rows], dtype=np.float64)
    mean_nmse_vals = np.array([r["mean_decoder_nmse"] for r in rows], dtype=np.float64)
    summary = {
        "n_decoder_layers": int(n_layers),
        "best_delta_ppl_layer": int(rows[int(np.argmax(delta_ppl_vals))]["target_decoder_layer"]),
        "best_delta_ppl": float(np.max(delta_ppl_vals)),
        "worst_delta_ppl_layer": int(rows[int(np.argmin(delta_ppl_vals))]["target_decoder_layer"]),
        "worst_delta_ppl": float(np.min(delta_ppl_vals)),
        "mean_delta_ppl": float(np.mean(delta_ppl_vals)),
        "corr_delta_loss_vs_mean_decoder_mse": pearson_corr(
            [r["mean_decoder_mse"] for r in rows],
            [r["delta_loss"] for r in rows],
        ),
        "corr_delta_ppl_vs_mean_decoder_mse": pearson_corr(
            [r["mean_decoder_mse"] for r in rows],
            [r["delta_ppl"] for r in rows],
        ),
        "corr_delta_loss_vs_mean_decoder_nmse": pearson_corr(
            [r["mean_decoder_nmse"] for r in rows],
            [r["delta_loss"] for r in rows],
        ),
        "corr_delta_ppl_vs_mean_decoder_nmse": pearson_corr(
            [r["mean_decoder_nmse"] for r in rows],
            [r["delta_ppl"] for r in rows],
        ),
        "max_mean_decoder_mse_layer": int(rows[int(np.argmax(mean_mse_vals))]["target_decoder_layer"]),
        "max_mean_decoder_mse": float(np.max(mean_mse_vals)),
        "max_mean_decoder_nmse_layer": int(rows[int(np.argmax(mean_nmse_vals))]["target_decoder_layer"]),
        "max_mean_decoder_nmse": float(np.max(mean_nmse_vals)),
        "predictor_summary": summarize_layer_full_lsb_predictors(rows),
        "engineered_predictor_summary": summarize_engineered_layer_predictors(rows),
    }
    return rows, summary


def build_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--out_dir", type=str, default="./cache/bitflip_hypothesis")
    parser.add_argument(
        "--experiment_mode",
        type=str,
        choices=["single_flip", "layer_full_lsb"],
        default="single_flip",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda_graph", type=str2bool, default=False)

    parser.add_argument("--bits_w", type=int, default=4)
    parser.add_argument("--sym_w", type=str2bool, default=False)
    parser.add_argument("--groupsize_w", type=int, default=-1)
    parser.add_argument("--bit_mode", type=str, choices=["random", "msb"], default="random")
    parser.add_argument("--gptq_act_order", type=str2bool, default=False)

    parser.add_argument("--gptq_dataset", type=str, default="c4")
    parser.add_argument("--gptq_nsamples", type=int, default=128)
    parser.add_argument("--gptq_seqlen", type=int, default=2048)
    parser.add_argument("--hessian_on", type=str, choices=["fp", "quant"], default="fp")

    parser.add_argument("--eval_dataset", type=str, default="wikitext2")
    parser.add_argument("--eval_seqlen", type=int, default=512)
    parser.add_argument("--eval_nsamples", type=int, default=8)

    parser.add_argument("--corr_trials", type=int, default=64)
    parser.add_argument("--group_compare", type=str2bool, default=True)
    parser.add_argument("--group_pool_size", type=int, default=2000)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        choices=["uniform_flat", "stratified_layer"],
        default="stratified_layer",
    )
    parser.add_argument("--baseline_repeats", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment script.")
    if args.bits_w <= 0:
        raise ValueError(f"bits_w must be positive, got {args.bits_w}.")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("Loading model...")
    model = model_from_hf_path(
        args.model_path,
        use_cuda_graph=args.use_cuda_graph,
        device_map="auto",
    ).eval()

    hessian_diagonal_fp = None
    if args.experiment_mode == "single_flip" and args.hessian_on == "fp":
        logging.info("Collect Hessian diagonal on fp model.")
        hessian_diagonal_fp = collect_rtn_hessian_diagonal(model, args, dev="cuda")

    logging.info("Applying RTN quantization (baseline, no random bitflip injection).")
    original_prob = getattr(args, "w_bitflip_prob", 0.0)
    original_act_order = args.gptq_act_order
    args.w_bitflip_prob = 0.0
    args.gptq_act_order = False
    quantize_nearest(model, args, dev="cuda", hessian_diagonal=None)
    args.w_bitflip_prob = original_prob
    args.gptq_act_order = original_act_order

    eval_inputs = build_eval_inputs(args)
    noise_floor = eval_noise_floor(model, eval_inputs, args.baseline_repeats)
    baseline_loss = noise_floor["mean"]
    baseline_ppl = math.exp(baseline_loss)
    logging.info(
        f"Baseline(mean over {noise_floor['repeats']} runs) "
        f"loss={baseline_loss:.6f}, ppl={baseline_ppl:.4f}, "
        f"noise_std={noise_floor['std']:.6e}, "
        f"max_abs_step_delta={noise_floor['max_abs_step_delta']:.6e}"
    )

    ts = build_timestamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": ts,
        "model_path": args.model_path,
        "seed": args.seed,
        "experiment_mode": args.experiment_mode,
        "bits_w": args.bits_w,
        "bit_mode": args.bit_mode,
        "sym_w": bool(args.sym_w),
        "groupsize_w": args.groupsize_w,
        "gptq_dataset": args.gptq_dataset,
        "gptq_nsamples": args.gptq_nsamples,
        "gptq_seqlen": args.gptq_seqlen,
        "eval_dataset": args.eval_dataset,
        "eval_nsamples": args.eval_nsamples,
        "eval_seqlen": args.eval_seqlen,
        "baseline_noise_floor": noise_floor,
        "baseline_loss": baseline_loss,
        "baseline_ppl": baseline_ppl,
    }

    if args.experiment_mode == "single_flip":
        hessian_diagonal = None
        if args.hessian_on == "fp":
            hessian_diagonal = hessian_diagonal_fp
        elif args.hessian_on == "quant":
            logging.info("Collect Hessian diagonal on quantized model.")
            hessian_diagonal = collect_rtn_hessian_diagonal(model, args, dev="cuda")

        infos, total_numel = build_layer_infos(model, args, hessian_diagonal)
        logging.info(
            f"Prepared {len(infos)} layers for bitflip trials, total elements={total_numel}."
        )

        rng = random.Random(args.seed + 13)
        corr_candidates = generate_candidates(
            infos, total_numel, args, rng, args.corr_trials
        )
        corr_rows = evaluate_candidates(
            model, infos, eval_inputs, baseline_loss, corr_candidates
        )
        corr_summary = summarize_correlations(corr_rows)
        logging.info("Correlation summary:")
        for k, v in corr_summary.items():
            logging.info(f"{k}: {v:.6f}" if not math.isnan(v) else f"{k}: nan")

        group_summary = {}
        group_rows = []
        if args.group_compare:
            pool = generate_candidates(
                infos, total_numel, args, rng, args.group_pool_size
            )
            top, rnd, bottom = select_group_candidates(pool, args.group_size, args.seed + 29)
            top_rows = evaluate_candidates(model, infos, eval_inputs, baseline_loss, top)
            rnd_rows = evaluate_candidates(model, infos, eval_inputs, baseline_loss, rnd)
            bottom_rows = evaluate_candidates(model, infos, eval_inputs, baseline_loss, bottom)

            for r in top_rows:
                r["group"] = "top"
            for r in rnd_rows:
                r["group"] = "random"
            for r in bottom_rows:
                r["group"] = "bottom"
            group_rows = top_rows + rnd_rows + bottom_rows

            def avg_delta(rows):
                return float(np.mean([x["delta_loss"] for x in rows]))

            group_summary = {
                "top_avg_delta_loss": avg_delta(top_rows),
                "top_avg_abs_delta_loss": float(np.mean([abs(x["delta_loss"]) for x in top_rows])),
                "random_avg_delta_loss": avg_delta(rnd_rows),
                "random_avg_abs_delta_loss": float(
                    np.mean([abs(x["delta_loss"]) for x in rnd_rows])
                ),
                "bottom_avg_delta_loss": avg_delta(bottom_rows),
                "bottom_avg_abs_delta_loss": float(
                    np.mean([abs(x["delta_loss"]) for x in bottom_rows])
                ),
                "top_nonzero_abs_delta": int(sum(abs(x["delta_loss"]) > 0 for x in top_rows)),
                "random_nonzero_abs_delta": int(sum(abs(x["delta_loss"]) > 0 for x in rnd_rows)),
                "bottom_nonzero_abs_delta": int(sum(abs(x["delta_loss"]) > 0 for x in bottom_rows)),
                "top_size": len(top_rows),
                "random_size": len(rnd_rows),
                "bottom_size": len(bottom_rows),
            }
            logging.info("Group comparison summary:")
            for k, v in group_summary.items():
                logging.info(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

        corr_csv = out_dir / f"{ts}_corr_trials.csv"
        save_csv(corr_rows, corr_csv)

        group_csv = None
        if group_rows:
            group_csv = out_dir / f"{ts}_group_trials.csv"
            save_csv(group_rows, group_csv)

        summary.update(
            {
                "hessian_on": args.hessian_on,
                "sampling_strategy": args.sampling_strategy,
                "corr_trials": args.corr_trials,
                "correlations": corr_summary,
                "group_compare": bool(args.group_compare),
                "group_summary": group_summary,
                "corr_csv": str(corr_csv),
                "group_csv": str(group_csv) if group_csv is not None else None,
            }
        )
    else:
        layer_rows, layer_summary = run_layer_full_lsb_sweep(
            model, args, eval_inputs, baseline_loss, baseline_ppl
        )
        layer_csv = out_dir / f"{ts}_layer_full_lsb_trials.csv"
        save_csv(layer_rows, layer_csv)
        summary.update(
            {
                "bit_mode": "lsb_fixed",
                "layer_full_lsb_summary": layer_summary,
                "layer_full_lsb_csv": str(layer_csv),
            }
        )

    summary_path = out_dir / f"{ts}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved summary to {summary_path}")
    if args.experiment_mode == "single_flip":
        logging.info(f"Saved correlation rows to {summary.get('corr_csv')}")
        if summary.get("group_csv") is not None:
            logging.info(f"Saved group rows to {summary.get('group_csv')}")
    else:
        logging.info(f"Saved layer sweep rows to {summary.get('layer_full_lsb_csv')}")


if __name__ == "__main__":
    main()
