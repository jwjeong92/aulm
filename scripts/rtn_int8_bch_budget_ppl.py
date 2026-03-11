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
from scripts.bitflip_hypothesis_experiment import build_eval_inputs, eval_avg_loss
from scripts.rtn_int8_group_budget import (
    aggregate_group_candidates,
    allocate_budget,
    classify_sensitive_blocks,
    configure_rtn_args,
    load_decoder_layer_multipliers,
    resolve_protection_modes,
    sanitize_name,
)
from utils.bitflip_risk import bit_exposure_multipliers, build_rtn_int8_perchannel_layer_infos
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


def parse_int_list(raw):
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def parse_float_list(raw):
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def build_bch_mode_specs(codeword_bits, t_values):
    specs = []
    for t in t_values:
        if int(t) <= 0:
            continue
        specs.append(f"bch_t{int(t)}:{int(codeword_bits)}:{int(t)}")
    return specs


def count_total_flat_chunk_groups(layer_infos, group_size_weights):
    total = 0
    for info in layer_infos:
        total += int(math.ceil((info.rows * info.cols) / group_size_weights))
    return int(total)


def resolve_budget_values(raw_budget_values, budget_fractions, total_groups, strongest_cost):
    if raw_budget_values.lower() != "auto":
        return [float(x) for x in parse_float_list(raw_budget_values)]

    full_strong_cost = float(total_groups * strongest_cost)
    budgets = []
    for fraction in budget_fractions:
        budget = float(full_strong_cost * fraction)
        if fraction > 0.0:
            budget = float(max(1.0, round(budget)))
        budgets.append(float(budget))
    deduped = []
    seen = set()
    for budget in budgets:
        key = round(float(budget), 12)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(float(budget))
    return deduped


def count_mask_hits(mask, bits):
    if mask.numel() == 0:
        return 0
    total = 0
    for bit in range(bits):
        total += int(torch.bitwise_and(torch.bitwise_right_shift(mask, bit), 1).sum().item())
    return total


def mix_seed(base_seed, *values):
    mask = (1 << 63) - 1
    state = int(base_seed) & mask
    for value in values:
        state = (state * 6364136223846793005 + int(value) + 1442695040888963407) & mask
    return int(state)


def sample_target_mask(numel, bit_error_probs, device, seed):
    probs = [float(x) for x in bit_error_probs]
    target_mask = torch.zeros((numel,), dtype=torch.int32, device=device)
    hit_count = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    for bit, bit_error_prob in enumerate(probs):
        if bit_error_prob <= 0.0:
            continue
        hits = (
            torch.rand(
                (numel,),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            < bit_error_prob
        ).to(torch.int32)
        hit_count += int(hits.sum().item())
        target_mask = torch.bitwise_or(
            target_mask,
            torch.bitwise_left_shift(hits, bit),
        )
    return target_mask, int(hit_count)


def restore_quantized_weights(layer_infos):
    with torch.no_grad():
        for info in layer_infos:
            weight = info.module.weight.data
            device = weight.device
            qint = info.qint.to(device=device, dtype=torch.int32)
            scale = info.scale_vec.to(device=device, dtype=torch.float32).view(info.rows, 1)
            zero = info.zero_vec.to(device=device, dtype=torch.float32).view(info.rows, 1)
            restored = scale * (qint.to(torch.float32) - zero)
            weight.copy_(restored.to(weight.dtype))


def build_selected_group_mode_map(allocation_rows):
    mapping = {}
    for row in allocation_rows:
        mapping.setdefault(row["layer_name"], {})[int(row["group_index"])] = row["selected_mode"]
    return mapping


def apply_mode_aware_faults(
    layer_infos,
    group_size_weights,
    base_bit_error_probs,
    allocation_rows,
    mode_probs_by_name,
    trial_seed,
):
    selected_group_map = build_selected_group_mode_map(allocation_rows)
    total_bit_hits = 0
    total_targets = 0
    selected_groups = 0
    bits = len(base_bit_error_probs)
    maxq = (1 << bits) - 1

    with torch.no_grad():
        for layer_idx, info in enumerate(layer_infos):
            weight = info.module.weight.data
            device = weight.device
            qint = info.qint.to(device=device, dtype=torch.int32)
            qflat_orig = qint.reshape(-1)

            base_mask, base_hits = sample_target_mask(
                qflat_orig.numel(),
                base_bit_error_probs,
                device=device,
                seed=mix_seed(trial_seed, layer_idx, 0),
            )
            qflat_faulted = torch.bitwise_xor(qflat_orig, base_mask).clamp_(0, maxq)
            total_bit_hits += int(base_hits)
            total_targets += int(torch.count_nonzero(base_mask).item())

            layer_selected = selected_group_map.get(info.layer_name, None)
            if layer_selected:
                for group_index, mode_name in sorted(layer_selected.items()):
                    start = int(group_index * group_size_weights)
                    end = int(min(start + group_size_weights, qflat_orig.numel()))
                    if start >= end:
                        continue
                    total_bit_hits -= count_mask_hits(base_mask[start:end], bits)
                    mode_mask, mode_hits = sample_target_mask(
                        end - start,
                        mode_probs_by_name[mode_name],
                        device=device,
                        seed=mix_seed(trial_seed, layer_idx, group_index, 1),
                    )
                    qflat_faulted[start:end] = torch.bitwise_xor(
                        qflat_orig[start:end],
                        mode_mask,
                    ).clamp_(0, maxq)
                    total_bit_hits += int(mode_hits)
                    total_targets += int(torch.count_nonzero(mode_mask).item()) - int(
                        torch.count_nonzero(base_mask[start:end]).item()
                    )
                    selected_groups += 1

            qint_faulted = qflat_faulted.view(info.rows, info.cols)
            scale = info.scale_vec.to(device=device, dtype=torch.float32).view(info.rows, 1)
            zero = info.zero_vec.to(device=device, dtype=torch.float32).view(info.rows, 1)
            faulted = scale * (qint_faulted.to(torch.float32) - zero)
            weight.copy_(faulted.to(weight.dtype))

    return {
        "selected_groups": int(selected_groups),
        "total_bit_hits": int(total_bit_hits),
        "total_targets_with_any_flip": int(total_targets),
    }


def summarize_trials(budget, allocation_summary, trial_rows):
    if not trial_rows:
        return {
            "budget": float(budget),
            "n_trials": 0,
        }

    losses = np.asarray([row["faulted_loss"] for row in trial_rows], dtype=np.float64)
    ppls = np.asarray([row["faulted_ppl"] for row in trial_rows], dtype=np.float64)
    bit_hits = np.asarray([row["total_bit_hits"] for row in trial_rows], dtype=np.float64)
    return {
        "budget": float(budget),
        "allocation_method": allocation_summary["allocation_method"] if allocation_summary else "none",
        "selected_groups": int(allocation_summary["selected_groups"]) if allocation_summary else 0,
        "budget_used": float(allocation_summary["budget_used"]) if allocation_summary else 0.0,
        "predicted_total_benefit": (
            float(allocation_summary["total_benefit"]) if allocation_summary else 0.0
        ),
        "predicted_model_residual_risk": (
            float(allocation_summary["estimated_model_residual_risk"])
            if allocation_summary
            else float("nan")
        ),
        "n_trials": int(len(trial_rows)),
        "mean_faulted_loss": float(np.mean(losses)),
        "std_faulted_loss": float(np.std(losses)),
        "mean_faulted_ppl": float(np.mean(ppls)),
        "std_faulted_ppl": float(np.std(ppls)),
        "mean_total_bit_hits": float(np.mean(bit_hits)),
        "std_total_bit_hits": float(np.std(bit_hits)),
        "best_faulted_ppl": float(np.min(ppls)),
        "worst_faulted_ppl": float(np.max(ppls)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./cache/rtn_int8_bch_budget_ppl",
    )
    parser.add_argument("--logfile", type=str, default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda_graph", type=str2bool, default=False)

    parser.add_argument("--gptq_dataset", type=str, default="c4")
    parser.add_argument("--gptq_nsamples", type=int, default=128)
    parser.add_argument("--gptq_seqlen", type=int, default=2048)
    parser.add_argument("--hessian_on", type=str, choices=["fp", "quant"], default="quant")

    parser.add_argument("--bit_error_prob", type=float, default=1e-2)
    parser.add_argument("--codeword_bits", type=int, default=8192)
    parser.add_argument("--ecc_t_values", type=str, default="0,8,11,16,24,32,64")
    parser.add_argument("--candidate_topk", type=int, default=0)
    parser.add_argument("--candidate_metric", type=str, default="baseline_risk")
    parser.add_argument("--critical_coverage", type=float, default=0.80)
    parser.add_argument("--important_coverage", type=float, default=0.15)
    parser.add_argument("--max_exact_dp_states", type=int, default=5000000)
    parser.add_argument("--budget_values", type=str, default="auto")
    parser.add_argument("--budget_fractions", type=str, default="0,0.01,0.05,0.10")
    parser.add_argument("--decoder_layer_multiplier_csv", type=str, default="none")
    parser.add_argument("--multiplier_key", type=str, default="relative_multiplier")

    parser.add_argument("--eval_dataset", type=str, default="wikitext2")
    parser.add_argument("--eval_seqlen", type=int, default=512)
    parser.add_argument("--eval_nsamples", type=int, default=16)
    parser.add_argument("--ppl_trials", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this workflow.")
    if args.ppl_trials <= 0:
        raise ValueError(f"ppl_trials must be positive, got {args.ppl_trials}.")

    setup_logging(args.logfile)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    configure_rtn_args(args)
    group_size_weights = int(args.codeword_bits // args.bits_w)
    if args.codeword_bits % args.bits_w != 0:
        raise ValueError(
            f"codeword_bits={args.codeword_bits} is not divisible by bits_w={args.bits_w}."
        )

    base_bit_error_probs = [float(args.bit_error_prob)] * int(args.bits_w)
    t_values = parse_int_list(args.ecc_t_values)
    ecc_mode_specs = build_bch_mode_specs(args.codeword_bits, t_values)
    modes = resolve_protection_modes(base_bit_error_probs, [], ecc_mode_specs)
    bit_template = [float(x) for x in bit_exposure_multipliers(args.bits_w).tolist()]
    mode_column_map = {mode.name: sanitize_name(mode.name) for mode in modes}
    strongest_mode_cost = max(mode.cost for mode in modes)

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
    total_groups = count_total_flat_chunk_groups(layer_infos, group_size_weights)
    candidate_topk = int(args.candidate_topk) if args.candidate_topk > 0 else int(total_groups)
    budget_values = resolve_budget_values(
        raw_budget_values=args.budget_values,
        budget_fractions=parse_float_list(args.budget_fractions),
        total_groups=total_groups,
        strongest_cost=strongest_mode_cost,
    )

    logging.info(
        f"Aggregating flat_chunk groups with group_size_weights={group_size_weights} "
        f"from {len(layer_infos)} layers (candidate_topk={candidate_topk}, total_groups={total_groups})."
    )
    candidate_rows, group_summary = aggregate_group_candidates(
        layer_infos=layer_infos,
        group_kind="flat_chunk",
        group_size_weights=group_size_weights,
        decoder_layer_multipliers=decoder_layer_multipliers,
        candidate_topk=candidate_topk,
        candidate_metric=args.candidate_metric,
        modes=modes,
        bit_exposure_template=bit_template,
        mode_column_map=mode_column_map,
    )
    classification_summary = classify_sensitive_blocks(
        candidate_rows,
        critical_coverage=args.critical_coverage,
        important_coverage=args.important_coverage,
    )

    eval_inputs = build_eval_inputs(args)
    clean_loss = float(eval_avg_loss(model, eval_inputs))
    clean_ppl = float(torch.exp(torch.tensor(clean_loss)).item())
    logging.info(f"Clean quantized baseline: loss={clean_loss:.6f}, ppl={clean_ppl:.6f}")

    mode_probs_by_name = {
        mode.name: [float(x) for x in mode.residual_bit_error_probs]
        for mode in modes
    }

    timestamp = build_timestamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = out_dir / f"{timestamp}_group_candidates.csv"
    save_csv(candidate_rows, candidate_csv)

    budget_rows = []
    trial_rows = []

    for budget_index, budget in enumerate(budget_values):
        logging.info(f"Evaluating budget={budget:.6f}")
        allocation_csv = out_dir / f"{timestamp}_budget_{budget_index:02d}_allocation.csv"
        if budget > 0.0:
            allocation_rows, allocation_summary = allocate_budget(
                candidate_rows=candidate_rows,
                modes=modes,
                budget=budget,
                mode_column_map=mode_column_map,
                total_model_baseline_risk=group_summary["total_model_baseline_risk"],
                max_exact_dp_states=args.max_exact_dp_states,
            )
        else:
            allocation_rows = []
            allocation_summary = {
                "allocation_method": "none",
                "budget": float(budget),
                "budget_used": 0.0,
                "remaining_budget": float(budget),
                "eligible_groups": 0,
                "selected_groups": 0,
                "total_benefit": 0.0,
                "candidate_baseline_risk": float(group_summary["candidate_total_baseline_risk"]),
                "candidate_reducible_risk": float(group_summary["candidate_total_reducible_risk"]),
                "eligible_candidate_baseline_risk": 0.0,
                "eligible_candidate_reducible_risk": 0.0,
                "candidate_residual_risk": float(group_summary["candidate_total_baseline_risk"]),
                "estimated_model_residual_risk": float(group_summary["total_model_baseline_risk"]),
                "mode_counts": [],
            }
        if allocation_rows:
            save_csv(allocation_rows, allocation_csv)

        current_trial_rows = []
        for trial_idx in range(args.ppl_trials):
            trial_seed = int(args.seed + trial_idx)
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            fault_stats = apply_mode_aware_faults(
                layer_infos=layer_infos,
                group_size_weights=group_size_weights,
                base_bit_error_probs=base_bit_error_probs,
                allocation_rows=allocation_rows,
                mode_probs_by_name=mode_probs_by_name,
                trial_seed=trial_seed,
            )
            faulted_loss = float(eval_avg_loss(model, eval_inputs))
            faulted_ppl = float(torch.exp(torch.tensor(faulted_loss)).item())
            restore_quantized_weights(layer_infos)
            row = {
                "budget_index": int(budget_index),
                "budget": float(budget),
                "trial": int(trial_idx),
                "trial_seed": int(trial_seed),
                "allocation_method": allocation_summary["allocation_method"],
                "selected_groups": int(allocation_summary["selected_groups"]),
                "budget_used": float(allocation_summary["budget_used"]),
                "predicted_total_benefit": float(allocation_summary["total_benefit"]),
                "predicted_model_residual_risk": float(
                    allocation_summary["estimated_model_residual_risk"]
                ),
                "faulted_loss": float(faulted_loss),
                "faulted_ppl": float(faulted_ppl),
                "delta_loss_vs_clean": float(faulted_loss - clean_loss),
                "delta_ppl_vs_clean": float(faulted_ppl - clean_ppl),
                "total_bit_hits": int(fault_stats["total_bit_hits"]),
                "total_targets_with_any_flip": int(fault_stats["total_targets_with_any_flip"]),
            }
            trial_rows.append(row)
            current_trial_rows.append(row)
            logging.info(
                f"[budget={budget:.6f} trial={trial_idx}] "
                f"faulted_ppl={faulted_ppl:.6f} delta_ppl={row['delta_ppl_vs_clean']:.6f} "
                f"bit_hits={row['total_bit_hits']}"
            )
            torch.cuda.empty_cache()

        budget_row = summarize_trials(
            budget=budget,
            allocation_summary=allocation_summary,
            trial_rows=current_trial_rows,
        )
        budget_row.update(
            {
                "budget_index": int(budget_index),
                "clean_loss": float(clean_loss),
                "clean_ppl": float(clean_ppl),
                "allocation_csv": str(allocation_csv) if allocation_rows else None,
            }
        )
        budget_rows.append(budget_row)

    budget_csv = out_dir / f"{timestamp}_budget_sweep.csv"
    trials_csv = out_dir / f"{timestamp}_ppl_trials.csv"
    save_csv(budget_rows, budget_csv)
    save_csv(trial_rows, trials_csv)

    summary = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "seed": args.seed,
        "bit_error_prob": float(args.bit_error_prob),
        "codeword_bits": int(args.codeword_bits),
        "group_size_weights": int(group_size_weights),
        "ecc_t_values": [int(x) for x in t_values],
        "ecc_mode_specs": ecc_mode_specs,
        "resolved_modes": [
            {
                "name": mode.name,
                "cost": float(mode.cost),
                "mode_source": mode.mode_source,
                "cost_source": mode.cost_source,
                "ecc_block_size": int(mode.ecc_block_size) if mode.ecc_block_size is not None else None,
                "ecc_correction_capability": (
                    int(mode.ecc_correction_capability)
                    if mode.ecc_correction_capability is not None
                    else None
                ),
                "residual_bit_error_probs": [float(x) for x in mode.residual_bit_error_probs],
                "uncorrectable_probabilities": (
                    [float(x) for x in mode.uncorrectable_probabilities]
                    if mode.uncorrectable_probabilities is not None
                    else None
                ),
            }
            for mode in modes
        ],
        "budget_values": [float(x) for x in budget_values],
        "budget_fractions": [float(x) for x in parse_float_list(args.budget_fractions)],
        "candidate_topk": int(candidate_topk),
        "total_groups": int(total_groups),
        "n_layers": int(len(layer_infos)),
        "total_numel": int(total_numel),
        "group_summary": group_summary,
        "classification_summary": classification_summary,
        "clean_loss": float(clean_loss),
        "clean_ppl": float(clean_ppl),
        "group_candidates_csv": str(candidate_csv),
        "budget_csv": str(budget_csv),
        "trials_csv": str(trials_csv),
        "top_budget_by_mean_ppl": min(budget_rows, key=lambda row: row["mean_faulted_ppl"])
        if budget_rows
        else None,
    }
    if args.decoder_layer_multiplier_csv != "none":
        summary["decoder_layer_multiplier_csv"] = args.decoder_layer_multiplier_csv
        summary["multiplier_key"] = args.multiplier_key

    summary_path = out_dir / f"{timestamp}_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    logging.info(f"Saved group candidates to {candidate_csv}")
    logging.info(f"Saved budget sweep to {budget_csv}")
    logging.info(f"Saved trial results to {trials_csv}")
    logging.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
