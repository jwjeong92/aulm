#!/usr/bin/env python3
import argparse
import csv
import heapq
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.quantization.weight_quant import collect_rtn_hessian_diagonal, quantize_nearest
from utils.bitflip_risk import (
    bit_exposure_multipliers,
    build_rtn_int8_perchannel_layer_infos,
    extract_decoder_layer_idx,
    resolve_bit_error_probs,
)
from utils.common import set_seed, str2bool
from utils.ecc_residual import infer_bch_parity_bits, residual_profile_from_kt
from utils.import_model import model_from_hf_path


@dataclass
class ProtectionMode:
    name: str
    cost: float
    residual_bit_error_probs: list[float]
    residual_risk_per_unit_mass: float
    mode_source: str
    cost_source: str | None = None
    ecc_block_size: int | None = None
    ecc_correction_capability: int | None = None
    uncorrectable_probabilities: list[float] | None = None


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


def load_decoder_layer_multipliers(path, key):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    multipliers = {}
    for row in rows:
        multipliers[int(row["decoder_layer"])] = float(row[key])
    return multipliers


def sanitize_name(name):
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()
    return sanitized or "mode"


def parse_mode_spec(spec, base_bit_error_probs):
    name, raw_cost, raw_profile = spec.split(":", 2)
    cost = float(raw_cost)
    if cost <= 0.0:
        raise ValueError(f"Protection mode cost must be positive: {spec}")

    if "," in raw_profile:
        residual = [float(token) for token in raw_profile.split(",")]
        if len(residual) != len(base_bit_error_probs):
            raise ValueError(
                f"Expected {len(base_bit_error_probs)} residual probabilities in {spec}, "
                f"got {len(residual)}."
            )
    else:
        scale = float(raw_profile)
        if scale < 0.0:
            raise ValueError(f"Residual BER scale must be non-negative: {spec}")
        residual = [float(scale * x) for x in base_bit_error_probs]
    return name, cost, residual


def parse_ecc_mode_spec(spec):
    tokens = spec.split(":")
    if len(tokens) == 3:
        name, raw_k, raw_t = tokens
        k = int(raw_k)
        t = int(raw_t)
        cost = float(infer_bch_parity_bits(k, t))
        cost_source = "bch_parity"
    elif len(tokens) == 4:
        name, raw_cost, raw_k, raw_t = tokens
        cost = float(raw_cost)
        if cost <= 0.0:
            raise ValueError(f"ECC mode cost must be positive: {spec}")
        k = int(raw_k)
        t = int(raw_t)
        cost_source = "manual"
    else:
        raise ValueError(
            "ECC mode spec must be 'name:k:t' or 'name:cost:k:t', "
            f"got {spec}."
        )
    if k <= 0:
        raise ValueError(f"ECC block size k must be positive: {spec}")
    if t < 0:
        raise ValueError(f"ECC correction capability t must be non-negative: {spec}")
    return name, cost, k, t, cost_source


def build_protection_mode(
    *,
    name,
    cost,
    residual_bit_error_probs,
    mode_source,
    cost_source=None,
    ecc_block_size=None,
    ecc_correction_capability=None,
    uncorrectable_probabilities=None,
):
    residual_bit_error_probs = [float(x) for x in residual_bit_error_probs]
    exposure_template = bit_exposure_multipliers(len(residual_bit_error_probs)).tolist()
    return ProtectionMode(
        name=name,
        cost=float(cost),
        residual_bit_error_probs=residual_bit_error_probs,
        residual_risk_per_unit_mass=float(
            sum(x * p for x, p in zip(exposure_template, residual_bit_error_probs))
        ),
        mode_source=mode_source,
        cost_source=cost_source,
        ecc_block_size=ecc_block_size,
        ecc_correction_capability=ecc_correction_capability,
        uncorrectable_probabilities=(
            [float(x) for x in uncorrectable_probabilities]
            if uncorrectable_probabilities is not None
            else None
        ),
    )


def resolve_protection_modes(base_bit_error_probs, mode_specs, ecc_mode_specs):
    modes = [
        build_protection_mode(
            name="none",
            cost=0.0,
            residual_bit_error_probs=base_bit_error_probs,
            mode_source="baseline",
            cost_source="baseline",
        )
    ]
    mode_names = {"none"}

    for spec in mode_specs:
        name, cost, residual = parse_mode_spec(spec, base_bit_error_probs)
        if name in mode_names:
            raise ValueError(f"Duplicate protection mode name: {name}")
        modes.append(
            build_protection_mode(
                name=name,
                cost=cost,
                residual_bit_error_probs=residual,
                mode_source="manual",
                cost_source="manual",
            )
        )
        mode_names.add(name)

    for spec in ecc_mode_specs:
        name, cost, k, t, cost_source = parse_ecc_mode_spec(spec)
        if name in mode_names:
            raise ValueError(f"Duplicate protection mode name: {name}")
        residual, uncorrectable = residual_profile_from_kt(k, t, base_bit_error_probs)
        modes.append(
            build_protection_mode(
                name=name,
                cost=cost,
                residual_bit_error_probs=residual,
                mode_source="ecc_analytic",
                cost_source=cost_source,
                ecc_block_size=int(k),
                ecc_correction_capability=int(t),
                uncorrectable_probabilities=uncorrectable,
            )
        )
        mode_names.add(name)
    return modes


def resolve_group_size_weights(args):
    if args.codeword_bits > 0:
        if args.codeword_bits % args.bits_w != 0:
            raise ValueError(
                f"codeword_bits={args.codeword_bits} is not divisible by bits_w={args.bits_w}."
            )
        derived = args.codeword_bits // args.bits_w
        if args.group_size_weights > 0 and args.group_size_weights != derived:
            raise ValueError(
                f"group_size_weights={args.group_size_weights} conflicts with "
                f"codeword_bits={args.codeword_bits} -> {derived} weights."
            )
        return int(derived)

    if args.group_size_weights <= 0:
        raise ValueError("Either group_size_weights or codeword_bits must be positive.")
    return int(args.group_size_weights)


def maybe_push_candidate(heap, topk, row, score_key):
    score = float(row[score_key])
    if topk <= 0:
        return

    entry = (
        score,
        float(row["baseline_risk"]),
        int(row["decoder_layer"]),
        row["layer_name"],
        int(row["group_index"]),
        row,
    )
    if len(heap) < topk:
        heapq.heappush(heap, entry)
        return
    if entry[:5] <= heap[0][:5]:
        return
    heapq.heapreplace(heap, entry)


def finalize_heap_rows(heap, score_key):
    rows = [entry[5] for entry in heap]
    rows.sort(
        key=lambda item: (
            float(item[score_key]),
            float(item["baseline_risk"]),
            float(item["reducible_risk"]),
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row["candidate_rank"] = idx
    return rows


def summarize_top_rows(rows):
    if not rows:
        return {}

    by_decoder = {}
    by_layer = {}
    for row in rows:
        decoder_layer = int(row["decoder_layer"])
        by_decoder[decoder_layer] = by_decoder.get(decoder_layer, 0) + 1
        layer_name = row["layer_name"]
        by_layer[layer_name] = by_layer.get(layer_name, 0) + 1

    decoder_rows = [
        {"decoder_layer": int(layer), "count": int(count)}
        for layer, count in sorted(by_decoder.items(), key=lambda item: (-item[1], item[0]))
    ]
    layer_rows = [
        {"layer_name": layer_name, "count": int(count)}
        for layer_name, count in sorted(
            by_layer.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "decoder_layer_counts": decoder_rows,
        "layer_name_counts": layer_rows[:20],
    }


def iter_row_chunk_groups(info, group_size_weights, decoder_layer_multiplier):
    scale_sq = info.scale_vec.to(torch.float64).pow(2).numpy()
    hdiag = info.hdiag.to(torch.float64).numpy()
    prefix = np.concatenate(([0.0], np.cumsum(hdiag)))
    starts = np.arange(0, info.cols, group_size_weights, dtype=np.int64)
    ends = np.minimum(starts + group_size_weights, info.cols)
    hdiag_sums = prefix[ends] - prefix[starts]

    group_idx = 0
    for row_idx, row_scale_sq in enumerate(scale_sq.tolist()):
        exposure_masses = float(decoder_layer_multiplier * row_scale_sq) * hdiag_sums
        for local_idx in range(len(starts)):
            start = int(starts[local_idx])
            end = int(ends[local_idx])
            exposure_mass = float(exposure_masses[local_idx])
            yield {
                "group_kind": "row_chunk",
                "group_index": int(group_idx),
                "layer_name": info.layer_name,
                "decoder_layer": int(extract_decoder_layer_idx(info.layer_name) or -1),
                "decoder_layer_multiplier": float(decoder_layer_multiplier),
                "row_start": int(row_idx),
                "row_end": int(row_idx),
                "col_start": int(start),
                "col_end_exclusive": int(end),
                "flat_start_weight": int(row_idx * info.cols + start),
                "flat_end_weight_exclusive": int(row_idx * info.cols + end),
                "num_weights": int(end - start),
                "group_exposure_mass": exposure_mass,
                "group_base_mass": exposure_mass,
            }
            group_idx += 1


def iter_flat_chunk_groups(info, group_size_weights, decoder_layer_multiplier):
    scale_sq = info.scale_vec.to(torch.float64).pow(2).numpy()
    hdiag = info.hdiag.to(torch.float64).numpy()
    prefix = np.concatenate(([0.0], np.cumsum(hdiag)))

    group_idx = 0
    group_start_flat = 0
    group_start_row = 0
    group_start_col = 0
    current_mass = 0.0
    remaining = int(group_size_weights)

    for row_idx, row_scale_sq in enumerate(scale_sq.tolist()):
        mass_scale = float(decoder_layer_multiplier * row_scale_sq)
        col = 0
        while col < info.cols:
            take = min(remaining, info.cols - col)
            next_col = col + take
            current_mass += mass_scale * float(prefix[next_col] - prefix[col])
            remaining -= take
            flat_end = row_idx * info.cols + next_col

            if remaining == 0 or (row_idx == info.rows - 1 and next_col == info.cols):
                exposure_mass = float(current_mass)
                yield {
                    "group_kind": "flat_chunk",
                    "group_index": int(group_idx),
                    "layer_name": info.layer_name,
                    "decoder_layer": int(extract_decoder_layer_idx(info.layer_name) or -1),
                    "decoder_layer_multiplier": float(decoder_layer_multiplier),
                    "row_start": int(group_start_row),
                    "row_end": int(row_idx),
                    "col_start": int(group_start_col),
                    "col_end_exclusive": int(next_col),
                    "flat_start_weight": int(group_start_flat),
                    "flat_end_weight_exclusive": int(flat_end),
                    "num_weights": int(flat_end - group_start_flat),
                    "group_exposure_mass": exposure_mass,
                    "group_base_mass": exposure_mass,
                }
                group_idx += 1
                current_mass = 0.0
                remaining = int(group_size_weights)
                group_start_flat = int(flat_end)
                if next_col == info.cols:
                    group_start_row = int(row_idx + 1)
                    group_start_col = 0
                else:
                    group_start_row = int(row_idx)
                    group_start_col = int(next_col)

            col = next_col


def compute_group_row_metrics(raw_row, modes, bit_exposure_template, mode_column_map):
    row = dict(raw_row)
    base_mass = float(row["group_exposure_mass"])
    exposures = [float(base_mass * value) for value in bit_exposure_template]
    for bit_idx, exposure in enumerate(exposures):
        row[f"bit_exposure_{bit_idx}"] = float(exposure)

    mode_risks = {}
    mode_benefits = {}
    mode_efficiencies = {}
    best_risk_mode = modes[0].name
    best_risk_value = float("inf")

    for mode in modes:
        residual_risk = float(
            sum(exposure * prob for exposure, prob in zip(exposures, mode.residual_bit_error_probs))
        )
        mode_risks[mode.name] = residual_risk
        row[f"risk_{mode_column_map[mode.name]}"] = float(residual_risk)
        if residual_risk < best_risk_value:
            best_risk_value = residual_risk
            best_risk_mode = mode.name

    baseline_risk = float(mode_risks["none"])
    for mode in modes:
        benefit = max(baseline_risk - mode_risks[mode.name], 0.0)
        efficiency = benefit / mode.cost if mode.cost > 0.0 else 0.0
        mode_benefits[mode.name] = float(benefit)
        mode_efficiencies[mode.name] = float(efficiency)
        if mode.name != "none":
            row[f"benefit_{mode_column_map[mode.name]}"] = float(benefit)
            row[f"efficiency_{mode_column_map[mode.name]}"] = float(efficiency)

    best_benefit_mode = max(mode_benefits, key=mode_benefits.get)
    best_efficiency_mode = max(mode_efficiencies, key=mode_efficiencies.get)
    reducible_risk = max(baseline_risk - best_risk_value, 0.0)

    row.update(
        {
            "baseline_risk": float(baseline_risk),
            "group_risk": float(baseline_risk),
            "min_residual_risk": float(best_risk_value),
            "reducible_risk": float(reducible_risk),
            "protection_efficiency": float(mode_efficiencies[best_efficiency_mode]),
            "best_risk_mode": best_risk_mode,
            "best_benefit_mode": best_benefit_mode,
            "best_efficiency_mode": best_efficiency_mode,
        }
    )
    return row


def aggregate_group_candidates(
    layer_infos,
    group_kind,
    group_size_weights,
    decoder_layer_multipliers,
    candidate_topk,
    candidate_metric,
    modes,
    bit_exposure_template,
    mode_column_map,
):
    heap = []
    total_groups = 0
    total_model_baseline_risk = 0.0
    total_model_reducible_risk = 0.0
    total_model_min_risk = 0.0
    total_model_base_mass = 0.0

    for info in layer_infos:
        decoder_layer = extract_decoder_layer_idx(info.layer_name)
        decoder_layer_multiplier = (
            float(decoder_layer_multipliers.get(decoder_layer, 1.0))
            if decoder_layer_multipliers is not None
            else 1.0
        )
        if group_kind == "row_chunk":
            iterator = iter_row_chunk_groups(
                info,
                group_size_weights,
                decoder_layer_multiplier,
            )
        elif group_kind == "flat_chunk":
            iterator = iter_flat_chunk_groups(
                info,
                group_size_weights,
                decoder_layer_multiplier,
            )
        else:
            raise ValueError(f"Unsupported group_kind: {group_kind}")

        for raw_row in iterator:
            row = compute_group_row_metrics(
                raw_row,
                modes=modes,
                bit_exposure_template=bit_exposure_template,
                mode_column_map=mode_column_map,
            )
            total_groups += 1
            total_model_base_mass += float(row["group_exposure_mass"])
            total_model_baseline_risk += float(row["baseline_risk"])
            total_model_reducible_risk += float(row["reducible_risk"])
            total_model_min_risk += float(row["min_residual_risk"])
            maybe_push_candidate(heap, candidate_topk, row, candidate_metric)

    rows = finalize_heap_rows(heap, candidate_metric)
    candidate_baseline_risk = float(sum(row["baseline_risk"] for row in rows))
    candidate_reducible_risk = float(sum(row["reducible_risk"] for row in rows))
    candidate_min_risk = float(sum(row["min_residual_risk"] for row in rows))
    candidate_base_mass = float(sum(row["group_exposure_mass"] for row in rows))
    return rows, {
        "group_kind": group_kind,
        "group_size_weights": int(group_size_weights),
        "total_groups": int(total_groups),
        "total_model_risk": float(total_model_baseline_risk),
        "total_model_baseline_risk": float(total_model_baseline_risk),
        "total_model_reducible_risk": float(total_model_reducible_risk),
        "total_model_min_risk": float(total_model_min_risk),
        "total_model_base_mass": float(total_model_base_mass),
        "candidate_topk": int(candidate_topk),
        "candidate_metric": candidate_metric,
        "candidate_total_risk": candidate_baseline_risk,
        "candidate_total_baseline_risk": candidate_baseline_risk,
        "candidate_total_reducible_risk": candidate_reducible_risk,
        "candidate_total_min_risk": candidate_min_risk,
        "candidate_total_base_mass": candidate_base_mass,
        "candidate_risk_fraction": (
            float(candidate_baseline_risk / total_model_baseline_risk)
            if total_model_baseline_risk > 0.0
            else 0.0
        ),
        "candidate_reducible_risk_fraction": (
            float(candidate_reducible_risk / total_model_reducible_risk)
            if total_model_reducible_risk > 0.0
            else 0.0
        ),
    }


def classify_sensitive_blocks(rows, critical_coverage, important_coverage):
    total_reducible = float(sum(row["reducible_risk"] for row in rows))
    if total_reducible <= 0.0:
        for idx, row in enumerate(rows, start=1):
            row["classification_rank"] = idx
            row["cumulative_reducible_risk"] = 0.0
            row["cumulative_reducible_fraction"] = 0.0
            row["sensitive_class"] = "B"
        return {
            "total_reducible_risk": 0.0,
            "critical_coverage": float(critical_coverage),
            "important_coverage": float(important_coverage),
            "class_counts": [{"class": "B", "count": int(len(rows)), "reducible_risk": 0.0}],
        }

    ordered = sorted(
        rows,
        key=lambda item: (
            float(item["protection_efficiency"]),
            float(item["reducible_risk"]),
            float(item["baseline_risk"]),
        ),
        reverse=True,
    )
    s_cutoff = float(critical_coverage * total_reducible)
    a_cutoff = float((critical_coverage + important_coverage) * total_reducible)

    cumulative = 0.0
    for idx, row in enumerate(ordered, start=1):
        prior = cumulative
        cumulative += float(row["reducible_risk"])
        if prior < s_cutoff and row["reducible_risk"] > 0.0:
            sensitive_class = "S"
        elif prior < a_cutoff and row["reducible_risk"] > 0.0:
            sensitive_class = "A"
        else:
            sensitive_class = "B"
        row["classification_rank"] = idx
        row["cumulative_reducible_risk"] = float(cumulative)
        row["cumulative_reducible_fraction"] = float(cumulative / total_reducible)
        row["sensitive_class"] = sensitive_class

    class_summary = {}
    for row in ordered:
        bucket = class_summary.setdefault(
            row["sensitive_class"],
            {"count": 0, "baseline_risk": 0.0, "reducible_risk": 0.0},
        )
        bucket["count"] += 1
        bucket["baseline_risk"] += float(row["baseline_risk"])
        bucket["reducible_risk"] += float(row["reducible_risk"])

    class_rows = [
        {
            "class": label,
            "count": int(class_summary.get(label, {}).get("count", 0)),
            "baseline_risk": float(class_summary.get(label, {}).get("baseline_risk", 0.0)),
            "reducible_risk": float(class_summary.get(label, {}).get("reducible_risk", 0.0)),
        }
        for label in ("S", "A", "B")
    ]
    return {
        "total_reducible_risk": float(total_reducible),
        "critical_coverage": float(critical_coverage),
        "important_coverage": float(important_coverage),
        "class_counts": class_rows,
    }


def build_block_options(row, modes, mode_column_map):
    options = [
        {
            "mode": "none",
            "cost": 0.0,
            "residual_risk": float(row["baseline_risk"]),
            "benefit": 0.0,
            "benefit_per_cost": 0.0,
        }
    ]

    for mode in modes:
        if mode.name == "none":
            continue
        residual_risk = float(row[f"risk_{mode_column_map[mode.name]}"])
        benefit = max(float(row["baseline_risk"]) - residual_risk, 0.0)
        options.append(
            {
                "mode": mode.name,
                "cost": float(mode.cost),
                "residual_risk": residual_risk,
                "benefit": float(benefit),
                "benefit_per_cost": float(benefit / mode.cost) if mode.cost > 0.0 else 0.0,
            }
        )

    best_by_cost = {}
    for option in options:
        existing = best_by_cost.get(option["cost"])
        if existing is None or option["benefit"] > existing["benefit"]:
            best_by_cost[option["cost"]] = option

    pruned = []
    best_benefit = -1.0
    for option in sorted(best_by_cost.values(), key=lambda item: (item["cost"], item["residual_risk"])):
        if option["benefit"] > best_benefit + 1e-12:
            pruned.append(option)
            best_benefit = option["benefit"]
    return pruned


def try_integer_cost(value, tol=1e-9):
    rounded = int(round(float(value)))
    if abs(float(value) - rounded) <= tol:
        return rounded
    return None


def materialize_allocation_rows(
    rows,
    selected_options,
    budget,
    remaining_budget,
    method,
    total_model_baseline_risk,
    full_candidate_baseline_risk,
    full_candidate_reducible_risk,
):
    allocation_rows = []
    total_benefit = 0.0
    by_mode = {}

    for row_idx, option in selected_options.items():
        if option["mode"] == "none":
            continue
        total_benefit += float(option["benefit"])
        bucket = by_mode.setdefault(option["mode"], {"count": 0, "cost": 0.0, "benefit": 0.0})
        bucket["count"] += 1
        bucket["cost"] += float(option["cost"])
        bucket["benefit"] += float(option["benefit"])

        output_row = dict(rows[row_idx])
        output_row.update(
            {
                "selected_mode": option["mode"],
                "mode_cost": float(option["cost"]),
                "mode_residual_risk": float(option["residual_risk"]),
                "residual_group_risk": float(option["residual_risk"]),
                "benefit": float(option["benefit"]),
                "benefit_per_cost": float(option["benefit_per_cost"]),
                "allocation_method": method,
            }
        )
        allocation_rows.append(output_row)

    allocation_rows.sort(key=lambda item: item["benefit"], reverse=True)
    for idx, row in enumerate(allocation_rows, start=1):
        row["allocation_rank"] = idx

    eligible_candidate_baseline_risk = float(sum(row["baseline_risk"] for row in rows))
    eligible_candidate_reducible_risk = float(sum(row["reducible_risk"] for row in rows))
    candidate_residual_risk = float(full_candidate_baseline_risk - total_benefit)
    mode_rows = [
        {
            "mode": mode_name,
            "count": int(bucket["count"]),
            "cost": float(bucket["cost"]),
            "benefit": float(bucket["benefit"]),
        }
        for mode_name, bucket in sorted(by_mode.items())
    ]
    return allocation_rows, {
        "allocation_method": method,
        "budget": float(budget),
        "budget_used": float(budget - remaining_budget),
        "remaining_budget": float(remaining_budget),
        "eligible_groups": int(sum(1 for row in rows if row["reducible_risk"] > 0.0)),
        "selected_groups": int(len(allocation_rows)),
        "total_benefit": float(total_benefit),
        "candidate_baseline_risk": float(full_candidate_baseline_risk),
        "candidate_reducible_risk": float(full_candidate_reducible_risk),
        "eligible_candidate_baseline_risk": float(eligible_candidate_baseline_risk),
        "eligible_candidate_reducible_risk": float(eligible_candidate_reducible_risk),
        "candidate_residual_risk": float(candidate_residual_risk),
        "estimated_model_residual_risk": float(total_model_baseline_risk - total_benefit),
        "mode_counts": mode_rows,
    }


def allocate_budget_exact(
    rows,
    option_tables,
    budget_int,
    budget_float,
    total_model_baseline_risk,
    full_candidate_baseline_risk,
    full_candidate_reducible_risk,
):
    prev = np.full(budget_int + 1, -np.inf, dtype=np.float64)
    prev[0] = 0.0
    choice_tables = []
    parent_tables = []

    for options in option_tables:
        curr = np.full(budget_int + 1, -np.inf, dtype=np.float64)
        choice = np.full(budget_int + 1, -1, dtype=np.int32)
        parent = np.full(budget_int + 1, -1, dtype=np.int32)
        for spent in range(budget_int + 1):
            base_value = float(prev[spent])
            if not np.isfinite(base_value):
                continue
            for option_idx, option in enumerate(options):
                next_spent = spent + option["cost_int"]
                if next_spent > budget_int:
                    continue
                value = base_value + float(option["benefit"])
                if value > curr[next_spent]:
                    curr[next_spent] = value
                    choice[next_spent] = option_idx
                    parent[next_spent] = spent
        prev = curr
        choice_tables.append(choice)
        parent_tables.append(parent)

    best_spent = int(np.argmax(prev))
    selected_options = {}
    spent = best_spent
    for row_idx in range(len(rows) - 1, -1, -1):
        option_idx = int(choice_tables[row_idx][spent])
        option = dict(option_tables[row_idx][option_idx])
        selected_options[row_idx] = option
        spent = int(parent_tables[row_idx][spent])

    return materialize_allocation_rows(
        rows=rows,
        selected_options=selected_options,
        budget=budget_float,
        remaining_budget=float(budget_int - best_spent),
        method="exact_mckp_dp",
        total_model_baseline_risk=total_model_baseline_risk,
        full_candidate_baseline_risk=full_candidate_baseline_risk,
        full_candidate_reducible_risk=full_candidate_reducible_risk,
    )


def push_upgrade(heap, row_idx, option_tables, current_indices, versions):
    current_idx = current_indices[row_idx]
    next_idx = current_idx + 1
    if next_idx >= len(option_tables[row_idx]):
        return

    current = option_tables[row_idx][current_idx]
    nxt = option_tables[row_idx][next_idx]
    delta_cost = float(nxt["cost"] - current["cost"])
    delta_benefit = float(nxt["benefit"] - current["benefit"])
    if delta_cost <= 0.0 or delta_benefit <= 0.0:
        return

    heapq.heappush(
        heap,
        (
            -float(delta_benefit / delta_cost),
            -float(delta_benefit),
            float(delta_cost),
            int(row_idx),
            int(next_idx),
            int(versions[row_idx]),
        ),
    )


def allocate_budget_greedy(
    rows,
    option_tables,
    budget,
    total_model_baseline_risk,
    full_candidate_baseline_risk,
    full_candidate_reducible_risk,
):
    current_indices = [0] * len(rows)
    versions = [0] * len(rows)
    heap = []
    remaining_budget = float(budget)

    for row_idx in range(len(rows)):
        push_upgrade(heap, row_idx, option_tables, current_indices, versions)

    while heap:
        neg_efficiency, neg_benefit, delta_cost, row_idx, next_idx, version = heapq.heappop(heap)
        if version != versions[row_idx]:
            continue
        if next_idx != current_indices[row_idx] + 1:
            continue
        if delta_cost > remaining_budget + 1e-12:
            continue

        current_indices[row_idx] = next_idx
        versions[row_idx] += 1
        remaining_budget -= delta_cost
        push_upgrade(heap, row_idx, option_tables, current_indices, versions)

    selected_options = {
        row_idx: dict(option_tables[row_idx][option_idx])
        for row_idx, option_idx in enumerate(current_indices)
    }
    return materialize_allocation_rows(
        rows=rows,
        selected_options=selected_options,
        budget=budget,
        remaining_budget=remaining_budget,
        method="greedy_marginal_upgrade",
        total_model_baseline_risk=total_model_baseline_risk,
        full_candidate_baseline_risk=full_candidate_baseline_risk,
        full_candidate_reducible_risk=full_candidate_reducible_risk,
    )


def allocate_budget(candidate_rows, modes, budget, mode_column_map, total_model_baseline_risk, max_exact_dp_states):
    if budget <= 0.0 or len(modes) <= 1:
        return [], None

    full_candidate_baseline_risk = float(sum(row["baseline_risk"] for row in candidate_rows))
    full_candidate_reducible_risk = float(sum(row["reducible_risk"] for row in candidate_rows))
    rows = [row for row in candidate_rows if row["reducible_risk"] > 0.0]
    if not rows:
        return [], {
            "allocation_method": "none",
            "budget": float(budget),
            "budget_used": 0.0,
            "remaining_budget": float(budget),
            "eligible_groups": 0,
            "selected_groups": 0,
            "total_benefit": 0.0,
            "candidate_baseline_risk": float(full_candidate_baseline_risk),
            "candidate_reducible_risk": float(full_candidate_reducible_risk),
            "eligible_candidate_baseline_risk": 0.0,
            "eligible_candidate_reducible_risk": 0.0,
            "candidate_residual_risk": float(full_candidate_baseline_risk),
            "estimated_model_residual_risk": float(total_model_baseline_risk),
            "mode_counts": [],
        }

    option_tables = [build_block_options(row, modes, mode_column_map) for row in rows]
    budget_int = try_integer_cost(budget)
    exact_possible = budget_int is not None
    if exact_possible:
        for options in option_tables:
            for option in options:
                cost_int = try_integer_cost(option["cost"])
                if cost_int is None:
                    exact_possible = False
                    break
                option["cost_int"] = cost_int
            if not exact_possible:
                break

    if exact_possible:
        state_count = len(rows) * (budget_int + 1)
        if state_count <= int(max_exact_dp_states):
            return allocate_budget_exact(
                rows=rows,
                option_tables=option_tables,
                budget_int=budget_int,
                budget_float=float(budget),
                total_model_baseline_risk=float(total_model_baseline_risk),
                full_candidate_baseline_risk=full_candidate_baseline_risk,
                full_candidate_reducible_risk=full_candidate_reducible_risk,
            )

    return allocate_budget_greedy(
        rows=rows,
        option_tables=option_tables,
        budget=float(budget),
        total_model_baseline_risk=float(total_model_baseline_risk),
        full_candidate_baseline_risk=full_candidate_baseline_risk,
        full_candidate_reducible_risk=full_candidate_reducible_risk,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./cache/rtn_int8_group_budget",
    )
    parser.add_argument("--logfile", type=str, default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_cuda_graph", type=str2bool, default=False)

    parser.add_argument("--gptq_dataset", type=str, default="c4")
    parser.add_argument("--gptq_nsamples", type=int, default=128)
    parser.add_argument("--gptq_seqlen", type=int, default=2048)
    parser.add_argument("--hessian_on", type=str, choices=["fp", "quant"], default="quant")

    parser.add_argument("--bit_error_prob", type=float, default=1e-6)
    parser.add_argument("--bit_error_prob_by_bit", type=str, default="none")
    parser.add_argument("--grouping", type=str, choices=["row_chunk", "flat_chunk"], default="flat_chunk")
    parser.add_argument("--group_size_weights", type=int, default=0)
    parser.add_argument("--codeword_bits", type=int, default=0)
    parser.add_argument("--candidate_topk", type=int, default=10000)
    parser.add_argument(
        "--candidate_metric",
        type=str,
        choices=["baseline_risk", "reducible_risk", "protection_efficiency"],
        default="baseline_risk",
    )
    parser.add_argument("--budget", type=float, default=0.0)
    parser.add_argument("--mode", type=str, action="append", default=[])
    parser.add_argument("--ecc_mode", type=str, action="append", default=[])
    parser.add_argument("--critical_coverage", type=float, default=0.80)
    parser.add_argument("--important_coverage", type=float, default=0.15)
    parser.add_argument("--max_exact_dp_states", type=int, default=5000000)
    parser.add_argument("--decoder_layer_multiplier_csv", type=str, default="none")
    parser.add_argument("--multiplier_key", type=str, default="relative_multiplier")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this workflow.")
    if args.candidate_topk <= 0:
        raise ValueError(f"candidate_topk must be positive, got {args.candidate_topk}.")
    if args.critical_coverage < 0.0 or args.important_coverage < 0.0:
        raise ValueError("Coverage values must be non-negative.")
    if args.critical_coverage + args.important_coverage > 1.0 + 1e-12:
        raise ValueError("critical_coverage + important_coverage must be <= 1.0.")

    setup_logging(args.logfile)
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    configure_rtn_args(args)
    group_size_weights = resolve_group_size_weights(args)
    base_bit_error_probs = resolve_bit_error_probs(
        bits=args.bits_w,
        bit_error_prob=args.bit_error_prob,
        bit_error_prob_by_bit=parse_bit_error_prob_by_bit(args.bit_error_prob_by_bit),
    )
    base_bit_error_probs = [float(x) for x in base_bit_error_probs.tolist()]
    modes = resolve_protection_modes(base_bit_error_probs, args.mode, args.ecc_mode)
    bit_template = [float(x) for x in bit_exposure_multipliers(args.bits_w).tolist()]
    mode_column_map = {mode.name: sanitize_name(mode.name) for mode in modes}

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

    logging.info(
        f"Aggregating {args.grouping} groups with group_size_weights={group_size_weights} "
        f"from {len(layer_infos)} layers."
    )
    candidate_rows, group_summary = aggregate_group_candidates(
        layer_infos=layer_infos,
        group_kind=args.grouping,
        group_size_weights=group_size_weights,
        decoder_layer_multipliers=decoder_layer_multipliers,
        candidate_topk=args.candidate_topk,
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
    top_group_distribution = summarize_top_rows(candidate_rows[:50])

    allocation_rows = []
    allocation_summary = None
    if args.budget > 0.0 and len(modes) > 1:
        allocation_rows, allocation_summary = allocate_budget(
            candidate_rows=candidate_rows,
            modes=modes,
            budget=args.budget,
            mode_column_map=mode_column_map,
            total_model_baseline_risk=group_summary["total_model_baseline_risk"],
            max_exact_dp_states=args.max_exact_dp_states,
        )

    timestamp = build_timestamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_csv = out_dir / f"{timestamp}_group_candidates.csv"
    allocation_csv = None
    if allocation_rows:
        allocation_csv = out_dir / f"{timestamp}_allocation.csv"
        save_csv(allocation_rows, allocation_csv)
    save_csv(candidate_rows, candidate_csv)

    summary = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "seed": args.seed,
        "hessian_on": args.hessian_on,
        "bits_w": args.bits_w,
        "sym_w": bool(args.sym_w),
        "groupsize_w": args.groupsize_w,
        "bit_error_prob": float(args.bit_error_prob),
        "bit_error_prob_by_bit": list(base_bit_error_probs),
        "grouping": args.grouping,
        "group_size_weights": int(group_size_weights),
        "codeword_bits": int(args.codeword_bits),
        "candidate_topk": int(args.candidate_topk),
        "candidate_metric": args.candidate_metric,
        "budget": float(args.budget),
        "mode_specs": list(args.mode),
        "ecc_mode_specs": list(args.ecc_mode),
        "resolved_modes": [
            {
                "name": mode.name,
                "cost": float(mode.cost),
                "residual_bit_error_probs": [float(x) for x in mode.residual_bit_error_probs],
                "residual_risk_per_unit_mass": float(mode.residual_risk_per_unit_mass),
                "mode_source": mode.mode_source,
                "cost_source": mode.cost_source,
                "ecc_block_size": int(mode.ecc_block_size) if mode.ecc_block_size is not None else None,
                "ecc_correction_capability": (
                    int(mode.ecc_correction_capability)
                    if mode.ecc_correction_capability is not None
                    else None
                ),
                "uncorrectable_probabilities": (
                    [float(x) for x in mode.uncorrectable_probabilities]
                    if mode.uncorrectable_probabilities is not None
                    else None
                ),
            }
            for mode in modes
        ],
        "mode_columns": mode_column_map,
        "critical_coverage": float(args.critical_coverage),
        "important_coverage": float(args.important_coverage),
        "max_exact_dp_states": int(args.max_exact_dp_states),
        "n_layers": int(len(layer_infos)),
        "total_numel": int(total_numel),
        "group_summary": group_summary,
        "classification_summary": classification_summary,
        "top50_distribution": top_group_distribution,
        "group_candidates_csv": str(candidate_csv),
        "allocation_csv": str(allocation_csv) if allocation_csv is not None else None,
        "top_group": candidate_rows[0] if candidate_rows else None,
        "allocation_summary": allocation_summary,
    }
    if args.decoder_layer_multiplier_csv != "none":
        summary["decoder_layer_multiplier_csv"] = args.decoder_layer_multiplier_csv
        summary["multiplier_key"] = args.multiplier_key

    summary_path = out_dir / f"{timestamp}_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    logging.info(
        f"Saved group candidates to {candidate_csv} "
        f"(candidate_topk={len(candidate_rows)}, total_groups={group_summary['total_groups']})."
    )
    if allocation_rows and allocation_summary is not None:
        logging.info(
            f"Saved allocation to {allocation_csv} "
            f"(selected_groups={len(allocation_rows)}, budget_used={allocation_summary['budget_used']:.6f}, "
            f"method={allocation_summary['allocation_method']})."
        )
    logging.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
