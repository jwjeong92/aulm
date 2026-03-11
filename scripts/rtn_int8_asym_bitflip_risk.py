#!/usr/bin/env python3
import argparse
import csv
import json
import logging
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
from utils.bitflip_risk import (
    build_rtn_int8_perchannel_layer_infos,
    resolve_bit_error_probs,
    select_top_bit_risk_records,
    select_top_weight_risk_records,
    summarize_layer_risks,
    summarize_model_bit_positions,
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


def load_decoder_layer_multipliers(path, key):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    multipliers = {}
    for row in rows:
        multipliers[int(row["decoder_layer"])] = float(row[key])
    return multipliers


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--out_dir", type=str, default="./cache/rtn_int8_asym_bitflip_risk")
    parser.add_argument("--logfile", type=str, default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda_graph", type=str2bool, default=False)

    parser.add_argument("--gptq_dataset", type=str, default="c4")
    parser.add_argument("--gptq_nsamples", type=int, default=128)
    parser.add_argument("--gptq_seqlen", type=int, default=2048)
    parser.add_argument("--hessian_on", type=str, choices=["fp", "quant"], default="quant")

    parser.add_argument("--bit_error_prob", type=float, default=1e-6)
    parser.add_argument("--bit_error_prob_by_bit", type=str, default="none")
    parser.add_argument("--topk_weights", type=int, default=256)
    parser.add_argument("--topk_bits", type=int, default=256)
    parser.add_argument("--decoder_layer_multiplier_csv", type=str, default="none")
    parser.add_argument("--multiplier_key", type=str, default="relative_multiplier")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this workflow.")

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
    decoder_layer_multipliers = None
    if args.decoder_layer_multiplier_csv != "none":
        decoder_layer_multipliers = load_decoder_layer_multipliers(
            args.decoder_layer_multiplier_csv,
            args.multiplier_key,
        )
        logging.info(
            f"Loaded decoder-layer multipliers from {args.decoder_layer_multiplier_csv} "
            f"using key={args.multiplier_key}."
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
    layer_rows = summarize_layer_risks(layer_infos, bit_error_probs)
    bit_position_rows = summarize_model_bit_positions(layer_infos, bit_error_probs)
    top_weight_rows = select_top_weight_risk_records(
        layer_infos, bit_error_probs, args.topk_weights
    )
    top_bit_rows = select_top_bit_risk_records(layer_infos, bit_error_probs, args.topk_bits)

    adjusted_layer_rows = None
    adjusted_top_weight_rows = None
    adjusted_top_bit_rows = None
    if decoder_layer_multipliers is not None:
        adjusted_layer_rows = summarize_layer_risks(
            layer_infos,
            bit_error_probs,
            decoder_layer_multipliers=decoder_layer_multipliers,
        )
        adjusted_top_weight_rows = select_top_weight_risk_records(
            layer_infos,
            bit_error_probs,
            args.topk_weights,
            decoder_layer_multipliers=decoder_layer_multipliers,
        )
        adjusted_top_bit_rows = select_top_bit_risk_records(
            layer_infos,
            bit_error_probs,
            args.topk_bits,
            decoder_layer_multipliers=decoder_layer_multipliers,
        )

    timestamp = build_timestamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_csv = out_dir / f"{timestamp}_layer_risk.csv"
    bit_csv = out_dir / f"{timestamp}_top_bit_risk.csv"
    weight_csv = out_dir / f"{timestamp}_top_weight_risk.csv"
    bitpos_csv = out_dir / f"{timestamp}_bit_position_risk.csv"
    summary_path = out_dir / f"{timestamp}_summary.json"
    adjusted_layer_csv = None
    adjusted_weight_csv = None
    adjusted_bit_csv = None

    save_csv(layer_rows, layer_csv)
    save_csv(top_bit_rows, bit_csv)
    save_csv(top_weight_rows, weight_csv)
    save_csv(bit_position_rows, bitpos_csv)
    if adjusted_layer_rows is not None:
        adjusted_layer_csv = out_dir / f"{timestamp}_layeraware_layer_risk.csv"
        adjusted_weight_csv = out_dir / f"{timestamp}_layeraware_top_weight_risk.csv"
        adjusted_bit_csv = out_dir / f"{timestamp}_layeraware_top_bit_risk.csv"
        save_csv(adjusted_layer_rows, adjusted_layer_csv)
        save_csv(adjusted_top_weight_rows, adjusted_weight_csv)
        save_csv(adjusted_top_bit_rows, adjusted_bit_csv)

    total_model_risk = float(sum(row["total_risk"] for row in layer_rows))
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
        "n_layers": int(len(layer_infos)),
        "total_numel": int(total_numel),
        "total_model_risk": total_model_risk,
        "layer_csv": str(layer_csv),
        "bit_position_csv": str(bitpos_csv),
        "top_weight_csv": str(weight_csv),
        "top_bit_csv": str(bit_csv),
        "top_layer": layer_rows[0] if layer_rows else None,
        "top_weight": top_weight_rows[0] if top_weight_rows else None,
        "top_bit": top_bit_rows[0] if top_bit_rows else None,
        "bit_position_summary": bit_position_rows,
    }
    if adjusted_layer_rows is not None:
        summary["decoder_layer_multiplier_csv"] = args.decoder_layer_multiplier_csv
        summary["multiplier_key"] = args.multiplier_key
        summary["layeraware_layer_csv"] = str(adjusted_layer_csv)
        summary["layeraware_top_weight_csv"] = str(adjusted_weight_csv)
        summary["layeraware_top_bit_csv"] = str(adjusted_bit_csv)
        summary["layeraware_total_model_risk"] = float(
            sum(row["total_risk"] for row in adjusted_layer_rows)
        )
        summary["layeraware_top_layer"] = adjusted_layer_rows[0] if adjusted_layer_rows else None
        summary["layeraware_top_weight"] = (
            adjusted_top_weight_rows[0] if adjusted_top_weight_rows else None
        )
        summary["layeraware_top_bit"] = adjusted_top_bit_rows[0] if adjusted_top_bit_rows else None

    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    logging.info(
        f"Saved RTN int8 asymmetric bitflip risk artifacts to {out_dir} "
        f"(layers={len(layer_rows)}, top_weights={len(top_weight_rows)}, top_bits={len(top_bit_rows)})."
    )
    if adjusted_layer_rows is not None:
        logging.info(
            "Saved layer-aware artifacts "
            f"(layers={len(adjusted_layer_rows)}, top_weights={len(adjusted_top_weight_rows)}, "
            f"top_bits={len(adjusted_top_bit_rows)})."
        )
    logging.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
