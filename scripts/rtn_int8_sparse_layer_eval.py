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

from lib.quantization.weight_quant import collect_rtn_hessian_diagonal, quantize_nearest
from scripts.bitflip_hypothesis_experiment import (
    build_eval_inputs,
    collect_decoder_outputs_cache,
    eval_avg_loss,
    eval_loss_and_decoder_distortion,
    eval_noise_floor,
    get_decoder_layers,
    pearson_corr,
    spearman_corr,
)
from utils.bitflip_risk import (
    build_rtn_int8_perchannel_layer_infos,
    extract_decoder_layer_idx,
    resolve_bit_error_probs,
    select_top_bit_risk_records,
)
from utils.common import set_seed, str2bool
from utils.import_model import model_from_hf_path


def setup_logging(logfile):
    handlers = [logging.StreamHandler()]
    if logfile != "none":
        logfile_path = Path(logfile)
        logfile_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile_path, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def build_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_bit_error_prob_by_bit(raw):
    if raw is None:
        return None
    raw = raw.strip()
    if not raw or raw.lower() == "none":
        return None
    return [float(token) for token in raw.split(",")]


def configure_rtn_args(args):
    args.bits_w = 8
    args.sym_w = False
    args.groupsize_w = -1
    args.gptq_act_order = False
    args.w_bitflip_prob = 0.0


def group_infos_by_decoder_layer(layer_infos):
    grouped = {}
    for info in layer_infos:
        decoder_layer = extract_decoder_layer_idx(info.layer_name)
        if decoder_layer is None:
            continue
        grouped.setdefault(int(decoder_layer), []).append(info)
    return dict(sorted(grouped.items()))


def select_layer_bit_records(
    layer_infos,
    bit_error_probs,
    faults_per_layer,
    unique_weight_faults,
    overfetch_factor,
):
    if faults_per_layer <= 0:
        return []

    bits = int(torch.as_tensor(bit_error_probs).numel())
    overfetch = max(faults_per_layer, faults_per_layer * max(overfetch_factor, 1))
    if unique_weight_faults:
        overfetch = max(overfetch, faults_per_layer * bits)

    candidates = select_top_bit_risk_records(layer_infos, bit_error_probs, overfetch)
    if not unique_weight_faults:
        selected = candidates[:faults_per_layer]
    else:
        selected = []
        seen_weights = set()
        for row in candidates:
            key = (row["layer_name"], int(row["row"]), int(row["col"]))
            if key in seen_weights:
                continue
            seen_weights.add(key)
            selected.append(row)
            if len(selected) >= faults_per_layer:
                break

    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = int(rank)
    return selected


def aggregate_selected_bit_records(rows):
    aggregated = {}
    for row in rows:
        key = (row["layer_name"], int(row["row"]), int(row["col"]))
        bucket = aggregated.setdefault(
            key,
            {
                "layer_name": row["layer_name"],
                "decoder_layer": int(row["decoder_layer"]),
                "row": int(row["row"]),
                "col": int(row["col"]),
                "combined_mask": 0,
                "selected_bits": [],
                "selected_bit_count": 0,
                "predicted_total_bit_risk": 0.0,
                "predicted_total_base_bit_risk": 0.0,
            },
        )
        bucket["combined_mask"] ^= int(row["mask"])
        bucket["selected_bits"].append(int(row["bit"]))
        bucket["selected_bit_count"] += 1
        bucket["predicted_total_bit_risk"] += float(row["bit_risk"])
        bucket["predicted_total_base_bit_risk"] += float(
            row.get("base_bit_risk", row["bit_risk"])
        )

    aggregated_rows = []
    for bucket in aggregated.values():
        bucket["selected_bits"] = ",".join(
            str(bit) for bit in sorted(bucket["selected_bits"])
        )
        aggregated_rows.append(bucket)
    aggregated_rows.sort(
        key=lambda item: item["predicted_total_bit_risk"],
        reverse=True,
    )
    return aggregated_rows


@torch.no_grad()
def apply_aggregated_bitflips(layer_info_map, aggregated_rows):
    backups = []
    applied_rows = []

    for row in aggregated_rows:
        info = layer_info_map[row["layer_name"]]
        module = info.module
        weight = module.weight.data
        target_row = int(row["row"])
        target_col = int(row["col"])
        old_val = weight[target_row, target_col].detach().clone()
        q_old = int(info.qint[target_row, target_col].item())
        q_new = q_old ^ int(row["combined_mask"])
        scale = float(info.scale_vec[target_row].item())
        zero = float(info.zero_vec[target_row].item())
        delta_q = int(q_new - q_old)
        delta_w = float(scale * delta_q)
        new_val = float(scale * (q_new - zero))

        weight[target_row, target_col] = weight.new_tensor(new_val)
        backups.append(
            {
                "module": module,
                "row": target_row,
                "col": target_col,
                "old_val": old_val,
            }
        )

        applied = dict(row)
        applied.update(
            {
                "q_old": int(q_old),
                "q_new": int(q_new),
                "delta_q": int(delta_q),
                "delta_w": float(delta_w),
                "scale": float(scale),
                "zero": float(zero),
            }
        )
        applied_rows.append(applied)

    return backups, applied_rows


@torch.no_grad()
def restore_aggregated_bitflips(backups):
    for row in backups:
        row["module"].weight.data[row["row"], row["col"]].copy_(row["old_val"])


def summarize_sparse_layer_rows(rows):
    if not rows:
        return {}

    predicted = np.asarray(
        [row["predicted_total_bit_risk"] for row in rows],
        dtype=np.float64,
    )
    delta_loss = np.asarray([row["delta_loss"] for row in rows], dtype=np.float64)
    abs_delta_loss = np.abs(delta_loss)
    delta_ppl = np.asarray([row["delta_ppl"] for row in rows], dtype=np.float64)
    abs_delta_ppl = np.abs(delta_ppl)

    return {
        "n_decoder_layers": int(len(rows)),
        "max_predicted_risk_layer": int(
            rows[int(np.argmax(predicted))]["target_decoder_layer"]
        ),
        "max_predicted_risk": float(np.max(predicted)),
        "max_abs_delta_loss_layer": int(
            rows[int(np.argmax(abs_delta_loss))]["target_decoder_layer"]
        ),
        "max_abs_delta_loss": float(np.max(abs_delta_loss)),
        "max_abs_delta_ppl_layer": int(
            rows[int(np.argmax(abs_delta_ppl))]["target_decoder_layer"]
        ),
        "max_abs_delta_ppl": float(np.max(abs_delta_ppl)),
        "pearson_predicted_vs_delta_loss": pearson_corr(predicted, delta_loss),
        "spearman_predicted_vs_delta_loss": spearman_corr(predicted, delta_loss),
        "pearson_predicted_vs_abs_delta_loss": pearson_corr(predicted, abs_delta_loss),
        "spearman_predicted_vs_abs_delta_loss": spearman_corr(predicted, abs_delta_loss),
        "pearson_predicted_vs_delta_ppl": pearson_corr(predicted, delta_ppl),
        "spearman_predicted_vs_delta_ppl": spearman_corr(predicted, delta_ppl),
        "pearson_predicted_vs_abs_delta_ppl": pearson_corr(predicted, abs_delta_ppl),
        "spearman_predicted_vs_abs_delta_ppl": spearman_corr(predicted, abs_delta_ppl),
    }


def run_sparse_layer_eval(
    model,
    args,
    eval_inputs,
    layer_infos,
    bit_error_probs,
    baseline_loss,
    baseline_ppl,
):
    layer_info_map = {info.layer_name: info for info in layer_infos}
    infos_by_decoder = group_infos_by_decoder_layer(layer_infos)

    decoder_layers = None
    baseline_cache = None
    if args.collect_decoder_distortion:
        decoder_layers = get_decoder_layers(model, args)
        logging.info(
            f"Collecting baseline decoder outputs cache for {len(decoder_layers)} decoder layers "
            f"x {len(eval_inputs)} samples..."
        )
        baseline_cache = collect_decoder_outputs_cache(
            model,
            eval_inputs,
            decoder_layers,
            cache_dtype=torch.float16,
        )

    selected_fault_rows = []
    applied_fault_rows = []
    trial_rows = []

    for decoder_layer, decoder_infos in infos_by_decoder.items():
        selected = select_layer_bit_records(
            decoder_infos,
            bit_error_probs,
            args.faults_per_layer,
            args.unique_weight_faults,
            args.selection_overfetch_factor,
        )
        if not selected:
            logging.warning(f"No sparse faults selected for decoder layer {decoder_layer}.")
            continue

        predicted_total_bit_risk = float(sum(row["bit_risk"] for row in selected))
        predicted_total_base_bit_risk = float(
            sum(row.get("base_bit_risk", row["bit_risk"]) for row in selected)
        )
        for row in selected:
            copied = dict(row)
            copied["target_decoder_layer"] = int(decoder_layer)
            selected_fault_rows.append(copied)

        aggregated = aggregate_selected_bit_records(selected)
        backups, applied = apply_aggregated_bitflips(layer_info_map, aggregated)
        try:
            if args.collect_decoder_distortion:
                distortion = eval_loss_and_decoder_distortion(
                    model,
                    eval_inputs,
                    decoder_layers,
                    baseline_cache,
                )
                loss = float(distortion["loss"])
            else:
                distortion = None
                loss = float(eval_avg_loss(model, eval_inputs))
        finally:
            restore_aggregated_bitflips(backups)

        for row in applied:
            copied = dict(row)
            copied["target_decoder_layer"] = int(decoder_layer)
            applied_fault_rows.append(copied)

        ppl = float(math.exp(loss))
        trial_row = {
            "target_decoder_layer": int(decoder_layer),
            "selection_method": (
                "top_bit_risk_unique_weights"
                if args.unique_weight_faults
                else "top_bit_risk"
            ),
            "num_linear_modules": int(len(decoder_infos)),
            "requested_faults": int(args.faults_per_layer),
            "selected_bit_records": int(len(selected)),
            "applied_unique_weights": int(len(aggregated)),
            "predicted_total_bit_risk": float(predicted_total_bit_risk),
            "predicted_total_base_bit_risk": float(predicted_total_base_bit_risk),
            "predicted_mean_bit_risk": float(
                predicted_total_bit_risk / max(len(selected), 1)
            ),
            "predicted_max_bit_risk": float(
                max(row["bit_risk"] for row in selected)
            ),
            "loss": float(loss),
            "delta_loss": float(loss - baseline_loss),
            "ppl": float(ppl),
            "delta_ppl": float(ppl - baseline_ppl),
        }
        if distortion is not None:
            trial_row["mean_decoder_mse"] = float(distortion["mean_mse"])
            trial_row["mean_decoder_nmse"] = float(distortion["mean_nmse"])
            for idx, value in enumerate(distortion["mse"]):
                trial_row[f"mse_decoder_layer_{idx}"] = float(value)
            for idx, value in enumerate(distortion["nmse"]):
                trial_row[f"nmse_decoder_layer_{idx}"] = float(value)
        trial_rows.append(trial_row)

        logging.info(
            f"[layer {decoder_layer}] selected_bits={len(selected)}, "
            f"unique_weights={len(aggregated)}, "
            f"predicted_risk={predicted_total_bit_risk:.6e}, "
            f"delta_ppl={trial_row['delta_ppl']:.6e}"
        )

    summary = summarize_sparse_layer_rows(trial_rows)
    return trial_rows, selected_fault_rows, applied_fault_rows, summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./cache/rtn_int8_sparse_layer_eval",
    )
    parser.add_argument("--logfile", type=str, default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda_graph", type=str2bool, default=False)

    parser.add_argument("--gptq_dataset", type=str, default="c4")
    parser.add_argument("--gptq_nsamples", type=int, default=128)
    parser.add_argument("--gptq_seqlen", type=int, default=2048)
    parser.add_argument("--hessian_on", type=str, choices=["fp", "quant"], default="quant")

    parser.add_argument("--eval_dataset", type=str, default="wikitext2")
    parser.add_argument("--eval_seqlen", type=int, default=512)
    parser.add_argument("--eval_nsamples", type=int, default=64)
    parser.add_argument("--baseline_repeats", type=int, default=5)

    parser.add_argument("--bit_error_prob", type=float, default=1e-6)
    parser.add_argument("--bit_error_prob_by_bit", type=str, default="none")
    parser.add_argument("--faults_per_layer", type=int, default=64)
    parser.add_argument("--unique_weight_faults", type=str2bool, default=True)
    parser.add_argument("--selection_overfetch_factor", type=int, default=8)
    parser.add_argument("--collect_decoder_distortion", type=str2bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this workflow.")
    if args.faults_per_layer <= 0:
        raise ValueError(
            f"faults_per_layer must be positive, got {args.faults_per_layer}."
        )

    setup_logging(args.logfile)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    configure_rtn_args(args)
    bit_error_probs = resolve_bit_error_probs(
        bits=args.bits_w,
        bit_error_prob=args.bit_error_prob,
        bit_error_prob_by_bit=parse_bit_error_prob_by_bit(args.bit_error_prob_by_bit),
    )

    logging.info("Loading model...")
    model = model_from_hf_path(
        args.model_path,
        use_cuda_graph=args.use_cuda_graph,
        device_map="auto",
    ).eval()

    if args.hessian_on == "fp":
        logging.info("Collecting RTN Hessian diagonal on fp model.")
        hessian_diagonal = collect_rtn_hessian_diagonal(model, args, dev="cuda")
        logging.info("Applying 8-bit asymmetric per-channel RTN quantization.")
        quantizer_states = quantize_nearest(model, args, dev="cuda", hessian_diagonal=None)
    else:
        logging.info("Applying 8-bit asymmetric per-channel RTN quantization.")
        quantizer_states = quantize_nearest(model, args, dev="cuda", hessian_diagonal=None)
        logging.info("Collecting RTN Hessian diagonal on quantized model.")
        hessian_diagonal = collect_rtn_hessian_diagonal(model, args, dev="cuda")

    layer_infos, total_numel = build_rtn_int8_perchannel_layer_infos(
        model=model,
        args=args,
        hessian_diagonal=hessian_diagonal,
        quantizer_states=quantizer_states,
    )
    logging.info(
        f"Prepared {len(layer_infos)} quantized linear layers for sparse evaluation "
        f"(total_numel={total_numel})."
    )

    eval_inputs = build_eval_inputs(args)
    noise_floor = eval_noise_floor(model, eval_inputs, args.baseline_repeats)
    baseline_loss = float(noise_floor["mean"])
    baseline_ppl = float(math.exp(baseline_loss))
    logging.info(
        f"Baseline(mean over {noise_floor['repeats']} runs) "
        f"loss={baseline_loss:.6f}, ppl={baseline_ppl:.6f}, "
        f"noise_std={noise_floor['std']:.6e}, "
        f"max_abs_step_delta={noise_floor['max_abs_step_delta']:.6e}"
    )

    trial_rows, selected_fault_rows, applied_fault_rows, sparse_summary = run_sparse_layer_eval(
        model=model,
        args=args,
        eval_inputs=eval_inputs,
        layer_infos=layer_infos,
        bit_error_probs=bit_error_probs,
        baseline_loss=baseline_loss,
        baseline_ppl=baseline_ppl,
    )

    timestamp = build_timestamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_csv = out_dir / f"{timestamp}_layer_sparse_trials.csv"
    selected_csv = out_dir / f"{timestamp}_selected_bit_records.csv"
    applied_csv = out_dir / f"{timestamp}_applied_sparse_faults.csv"
    summary_path = out_dir / f"{timestamp}_summary.json"

    save_csv(trial_rows, trial_csv)
    save_csv(selected_fault_rows, selected_csv)
    save_csv(applied_fault_rows, applied_csv)

    summary = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "seed": args.seed,
        "hessian_on": args.hessian_on,
        "bits_w": args.bits_w,
        "sym_w": bool(args.sym_w),
        "groupsize_w": args.groupsize_w,
        "bit_error_prob": float(args.bit_error_prob),
        "bit_error_prob_by_bit": [float(x) for x in bit_error_probs.tolist()],
        "gptq_dataset": args.gptq_dataset,
        "gptq_nsamples": args.gptq_nsamples,
        "gptq_seqlen": args.gptq_seqlen,
        "eval_dataset": args.eval_dataset,
        "eval_nsamples": args.eval_nsamples,
        "eval_seqlen": args.eval_seqlen,
        "baseline_noise_floor": noise_floor,
        "baseline_loss": float(baseline_loss),
        "baseline_ppl": float(baseline_ppl),
        "faults_per_layer": int(args.faults_per_layer),
        "unique_weight_faults": bool(args.unique_weight_faults),
        "selection_overfetch_factor": int(args.selection_overfetch_factor),
        "collect_decoder_distortion": bool(args.collect_decoder_distortion),
        "n_layers": int(len(layer_infos)),
        "total_numel": int(total_numel),
        "layer_sparse_summary": sparse_summary,
        "layer_sparse_csv": str(trial_csv),
        "selected_bit_records_csv": str(selected_csv),
        "applied_sparse_faults_csv": str(applied_csv),
        "top_trial_by_abs_delta_ppl": (
            max(trial_rows, key=lambda row: abs(row["delta_ppl"])) if trial_rows else None
        ),
        "top_trial_by_predicted_risk": (
            max(trial_rows, key=lambda row: row["predicted_total_bit_risk"])
            if trial_rows
            else None
        ),
    }

    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    logging.info(f"Saved sparse layer trial rows to {trial_csv}")
    logging.info(f"Saved selected bit records to {selected_csv}")
    logging.info(f"Saved applied sparse faults to {applied_csv}")
    logging.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
