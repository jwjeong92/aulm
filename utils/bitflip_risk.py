import heapq
import logging
import math
import re
from dataclasses import dataclass

import torch

from lib.quantization.quantizer import quantize_to_int
from lib.quantization.weight_quant import iter_quantized_linears


@dataclass
class RTNInt8PerChannelLayerInfo:
    layer_name: str
    module: torch.nn.Module
    rows: int
    cols: int
    hdiag: torch.Tensor
    qint: torch.Tensor
    scale_vec: torch.Tensor
    zero_vec: torch.Tensor


def validate_rtn_int8_perchannel_asym(args):
    if int(args.bits_w) != 8:
        raise ValueError(f"Expected bits_w=8 for this workflow, got {args.bits_w}.")
    if bool(args.sym_w):
        raise ValueError("Expected asymmetric quantization (sym_w=False).")
    if int(args.groupsize_w) > 0:
        raise ValueError(
            f"Expected per-channel quantization (groupsize_w=-1), got {args.groupsize_w}."
        )
    if bool(getattr(args, "gptq_act_order", False)):
        raise ValueError("This workflow expects plain RTN without act_order.")


def resolve_bit_error_probs(bits, bit_error_prob=1.0, bit_error_prob_by_bit=None):
    if bit_error_prob_by_bit is None:
        return torch.full((bits,), float(bit_error_prob), dtype=torch.float64)

    probs = torch.as_tensor(bit_error_prob_by_bit, dtype=torch.float64)
    if probs.numel() != bits:
        raise ValueError(
            f"Expected {bits} bit-error probabilities, got {probs.numel()}."
        )
    return probs.reshape(bits)


def build_rtn_int8_perchannel_layer_infos(model, args, hessian_diagonal, quantizer_states):
    validate_rtn_int8_perchannel_asym(args)

    infos = []
    total_numel = 0
    missing_hdiag = 0
    missing_quantizer = 0

    for layer_name, linear in iter_quantized_linears(model, args):
        layer_hdiag = hessian_diagonal.get(layer_name, None)
        if layer_hdiag is None:
            missing_hdiag += 1
            continue

        quantizer_state = quantizer_states.get(layer_name, None)
        if quantizer_state is None:
            missing_quantizer += 1
            continue

        if quantizer_state.get("perm") is not None or quantizer_state.get("invperm") is not None:
            raise ValueError(
                f"Quantizer state for {layer_name} includes act_order permutations; "
                "this utility expects plain RTN."
            )

        weight = linear.weight.data.detach()
        if weight.dim() != 2:
            logging.warning(
                f"Skip non-2D weight for bitflip risk: {layer_name}, shape={tuple(weight.shape)}"
            )
            continue

        rows, cols = weight.shape
        if layer_hdiag.numel() != cols:
            logging.warning(
                f"Skip Hessian mismatch for {layer_name}: "
                f"hdiag={layer_hdiag.numel()}, cols={cols}"
            )
            continue

        scale_vec = quantizer_state["scale"].detach().cpu().to(torch.float32).view(-1)
        zero_vec = quantizer_state["zero"].detach().cpu().to(torch.float32).view(-1)
        if scale_vec.numel() != rows or zero_vec.numel() != rows:
            logging.warning(
                f"Skip quantizer mismatch for {layer_name}: "
                f"scale={scale_vec.numel()}, zero={zero_vec.numel()}, rows={rows}"
            )
            continue

        scale = scale_vec.view(rows, 1).to(weight.device)
        zero = zero_vec.view(rows, 1).to(weight.device)
        maxq = quantizer_state["maxq"].to(weight.device)
        qint = quantize_to_int(weight, scale, zero, maxq).reshape(rows, cols)

        infos.append(
            RTNInt8PerChannelLayerInfo(
                layer_name=layer_name,
                module=linear,
                rows=rows,
                cols=cols,
                hdiag=layer_hdiag.detach().cpu().to(torch.float32).view(cols),
                qint=qint.detach().cpu().to(torch.int32),
                scale_vec=scale_vec,
                zero_vec=zero_vec,
            )
        )
        total_numel += rows * cols

    logging.info(
        f"Prepared {len(infos)} RTN int8 asymmetric per-channel layers "
        f"(missing_hdiag={missing_hdiag}, missing_quantizer={missing_quantizer}, total_numel={total_numel})."
    )
    return infos, total_numel


def bit_risk_multipliers(bit_error_probs):
    bit_error_probs = torch.as_tensor(bit_error_probs, dtype=torch.float64)
    bit_positions = torch.arange(bit_error_probs.numel(), dtype=torch.float64)
    return 0.5 * bit_error_probs * torch.pow(4.0, bit_positions)


def bit_exposure_multipliers(bits):
    bit_positions = torch.arange(int(bits), dtype=torch.float64)
    return 0.5 * torch.pow(4.0, bit_positions)


def summarize_layer_risks(layer_infos, bit_error_probs, decoder_layer_multipliers=None):
    multipliers = bit_risk_multipliers(bit_error_probs)
    weight_multiplier = float(multipliers.sum().item())
    rows = []

    for info in layer_infos:
        scale_sq = info.scale_vec.to(torch.float64).pow(2)
        hdiag = info.hdiag.to(torch.float64)
        base_sum = float(scale_sq.sum().item() * hdiag.sum().item())
        decoder_layer = extract_decoder_layer_idx(info.layer_name)
        decoder_layer_multiplier = (
            float(decoder_layer_multipliers.get(decoder_layer, 1.0))
            if decoder_layer_multipliers is not None
            else 1.0
        )
        total_risk = decoder_layer_multiplier * weight_multiplier * base_sum
        row = {
            "layer_name": info.layer_name,
            "decoder_layer": int(decoder_layer) if decoder_layer is not None else -1,
            "decoder_layer_multiplier": float(decoder_layer_multiplier),
            "rows": int(info.rows),
            "cols": int(info.cols),
            "numel": int(info.rows * info.cols),
            "sum_scale_sq": float(scale_sq.sum().item()),
            "sum_hessian_diag": float(hdiag.sum().item()),
            "weight_multiplier": float(weight_multiplier),
            "total_risk": float(total_risk),
            "mean_weight_risk": float(total_risk / max(info.rows * info.cols, 1)),
            "max_weight_risk_upper_bound": float(
                weight_multiplier * float(scale_sq.max().item()) * float(hdiag.max().item())
            ),
            "max_scale": float(info.scale_vec.max().item()),
            "min_scale": float(info.scale_vec.min().item()),
            "max_hessian_diag": float(info.hdiag.max().item()),
            "min_hessian_diag": float(info.hdiag.min().item()),
        }
        rows.append(row)

    rows.sort(key=lambda item: item["total_risk"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def summarize_model_bit_positions(layer_infos, bit_error_probs):
    multipliers = bit_risk_multipliers(bit_error_probs)
    base_total = 0.0
    for info in layer_infos:
        scale_sq = info.scale_vec.to(torch.float64).pow(2)
        hdiag = info.hdiag.to(torch.float64)
        base_total += float(scale_sq.sum().item() * hdiag.sum().item())

    rows = []
    total_risk = 0.0
    for bit, multiplier in enumerate(multipliers.tolist()):
        bit_risk = base_total * float(multiplier)
        total_risk += bit_risk
        rows.append(
            {
                "bit": int(bit),
                "bit_error_prob": float(bit_error_probs[bit]),
                "bit_multiplier": float(multiplier),
                "total_risk": float(bit_risk),
            }
        )

    for row in rows:
        row["risk_fraction"] = float(row["total_risk"] / total_risk) if total_risk > 0 else 0.0
    rows.sort(key=lambda item: item["bit"], reverse=True)
    return rows


def _sort_desc(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    order = torch.argsort(values, descending=True)
    return values[order], order


def _topk_pair_products(left_values, right_values, topk):
    if topk <= 0:
        return []

    left_sorted, left_order = _sort_desc(left_values)
    right_sorted, right_order = _sort_desc(right_values)
    if left_sorted.numel() == 0 or right_sorted.numel() == 0:
        return []

    heap = [(-float(left_sorted[0].item() * right_sorted[0].item()), 0, 0)]
    visited = {(0, 0)}
    results = []

    while heap and len(results) < topk:
        neg_prod, i, j = heapq.heappop(heap)
        results.append(
            (
                int(left_order[i].item()),
                int(right_order[j].item()),
                float(-neg_prod),
            )
        )

        if i + 1 < left_sorted.numel() and (i + 1, j) not in visited:
            visited.add((i + 1, j))
            heapq.heappush(
                heap,
                (-float(left_sorted[i + 1].item() * right_sorted[j].item()), i + 1, j),
            )
        if j + 1 < right_sorted.numel() and (i, j + 1) not in visited:
            visited.add((i, j + 1))
            heapq.heappush(
                heap,
                (-float(left_sorted[i].item() * right_sorted[j + 1].item()), i, j + 1),
            )

    return results


def _topk_triplet_products(left_values, middle_values, right_values, topk):
    if topk <= 0:
        return []

    left_sorted, left_order = _sort_desc(left_values)
    middle_sorted, middle_order = _sort_desc(middle_values)
    right_sorted, right_order = _sort_desc(right_values)
    if left_sorted.numel() == 0 or middle_sorted.numel() == 0 or right_sorted.numel() == 0:
        return []

    start = float(left_sorted[0].item() * middle_sorted[0].item() * right_sorted[0].item())
    heap = [(-start, 0, 0, 0)]
    visited = {(0, 0, 0)}
    results = []

    while heap and len(results) < topk:
        neg_prod, i, j, k = heapq.heappop(heap)
        results.append(
            (
                int(left_order[i].item()),
                int(middle_order[j].item()),
                int(right_order[k].item()),
                float(-neg_prod),
            )
        )

        neighbors = (
            (i + 1, j, k),
            (i, j + 1, k),
            (i, j, k + 1),
        )
        for ni, nj, nk in neighbors:
            if ni >= left_sorted.numel():
                continue
            if nj >= middle_sorted.numel():
                continue
            if nk >= right_sorted.numel():
                continue
            state = (ni, nj, nk)
            if state in visited:
                continue
            visited.add(state)
            prod = float(
                left_sorted[ni].item()
                * middle_sorted[nj].item()
                * right_sorted[nk].item()
            )
            heapq.heappush(heap, (-prod, ni, nj, nk))

    return results


def select_top_weight_risk_records(
    layer_infos,
    bit_error_probs,
    topk,
    decoder_layer_multipliers=None,
):
    if topk <= 0:
        return []

    multipliers = bit_risk_multipliers(bit_error_probs)
    weight_multiplier = float(multipliers.sum().item())
    rows = []

    for info in layer_infos:
        scale_sq = info.scale_vec.to(torch.float64).pow(2)
        hdiag = info.hdiag.to(torch.float64)
        decoder_layer = extract_decoder_layer_idx(info.layer_name)
        decoder_layer_multiplier = (
            float(decoder_layer_multipliers.get(decoder_layer, 1.0))
            if decoder_layer_multipliers is not None
            else 1.0
        )
        local = _topk_pair_products(scale_sq, hdiag, topk)
        for row_idx, col_idx, base_product in local:
            qint = int(info.qint[row_idx, col_idx].item())
            scale = float(info.scale_vec[row_idx].item())
            zero = float(info.zero_vec[row_idx].item())
            base_weight_risk = float(weight_multiplier * base_product)
            rows.append(
                {
                    "layer_name": info.layer_name,
                    "decoder_layer": int(decoder_layer) if decoder_layer is not None else -1,
                    "decoder_layer_multiplier": float(decoder_layer_multiplier),
                    "row": int(row_idx),
                    "col": int(col_idx),
                    "qint": int(qint),
                    "scale": float(scale),
                    "zero": float(zero),
                    "hessian_diag": float(info.hdiag[col_idx].item()),
                    "weight_value": float(scale * (qint - zero)),
                    "base_weight_risk": float(base_weight_risk),
                    "weight_risk": float(decoder_layer_multiplier * base_weight_risk),
                }
            )

    rows.sort(key=lambda item: item["weight_risk"], reverse=True)
    rows = rows[:topk]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def select_top_bit_risk_records(
    layer_infos,
    bit_error_probs,
    topk,
    decoder_layer_multipliers=None,
):
    if topk <= 0:
        return []

    bit_multipliers = bit_risk_multipliers(bit_error_probs)
    rows = []

    for info in layer_infos:
        scale_sq = info.scale_vec.to(torch.float64).pow(2)
        hdiag = info.hdiag.to(torch.float64)
        decoder_layer = extract_decoder_layer_idx(info.layer_name)
        decoder_layer_multiplier = (
            float(decoder_layer_multipliers.get(decoder_layer, 1.0))
            if decoder_layer_multipliers is not None
            else 1.0
        )
        local = _topk_triplet_products(scale_sq, hdiag, bit_multipliers, topk)
        for row_idx, col_idx, bit, risk in local:
            q_old = int(info.qint[row_idx, col_idx].item())
            mask = 1 << int(bit)
            q_new = q_old ^ mask
            delta_q = q_new - q_old
            scale = float(info.scale_vec[row_idx].item())
            zero = float(info.zero_vec[row_idx].item())
            delta_w = scale * float(delta_q)
            rows.append(
                {
                    "layer_name": info.layer_name,
                    "decoder_layer": int(decoder_layer) if decoder_layer is not None else -1,
                    "decoder_layer_multiplier": float(decoder_layer_multiplier),
                    "row": int(row_idx),
                    "col": int(col_idx),
                    "bit": int(bit),
                    "mask": int(mask),
                    "q_old": int(q_old),
                    "q_new": int(q_new),
                    "delta_q": int(delta_q),
                    "scale": float(scale),
                    "zero": float(zero),
                    "hessian_diag": float(info.hdiag[col_idx].item()),
                    "delta_w": float(delta_w),
                    "bit_error_prob": float(bit_error_probs[bit]),
                    "base_bit_risk": float(risk),
                    "bit_risk": float(decoder_layer_multiplier * risk),
                }
            )

    rows.sort(key=lambda item: item["bit_risk"], reverse=True)
    rows = rows[:topk]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def extract_decoder_layer_idx(layer_name):
    for pattern in (
        r"\bmodel\.decoder\.layers\.(\d+)\.",
        r"\bmodel\.layers\.(\d+)\.",
    ):
        match = re.search(pattern, layer_name)
        if match is not None:
            return int(match.group(1))
    return None


def aggregate_decoder_layer_risks(layer_rows):
    by_decoder = {}
    skipped = 0

    for row in layer_rows:
        layer_name = row["layer_name"]
        decoder_layer = extract_decoder_layer_idx(layer_name)
        if decoder_layer is None:
            skipped += 1
            continue

        bucket = by_decoder.setdefault(
            decoder_layer,
            {
                "decoder_layer": int(decoder_layer),
                "num_modules": 0,
                "total_risk": 0.0,
                "total_numel": 0,
                "module_names": [],
            },
        )
        bucket["num_modules"] += 1
        bucket["total_risk"] += float(row["total_risk"])
        bucket["total_numel"] += int(row.get("numel", 0))
        bucket["module_names"].append(layer_name)

    rows = list(by_decoder.values())
    rows.sort(key=lambda item: item["decoder_layer"])

    total_risk = sum(row["total_risk"] for row in rows)
    for idx, row in enumerate(sorted(rows, key=lambda item: item["total_risk"], reverse=True), start=1):
        row["risk_rank"] = idx
    for row in rows:
        row["risk_share"] = (
            float(row["total_risk"] / total_risk) if total_risk > 0.0 else float("nan")
        )

    return rows, {"total_risk": float(total_risk), "skipped_rows": int(skipped)}


def _rankdata(values):
    values = list(values)
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = 0.5 * (pos + end - 1)
        for idx in order[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def _pearson_corr(x, y):
    if not x or not y or len(x) != len(y):
        return float("nan")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return float(num / (den_x * den_y))


def _spearman_corr(x, y):
    return _pearson_corr(_rankdata(x), _rankdata(y))


def estimate_decoder_layer_multipliers(
    decoder_risk_rows,
    damage_rows,
    damage_key="delta_ppl",
    epsilon=1e-12,
):
    risk_by_layer = {int(row["decoder_layer"]): row for row in decoder_risk_rows}
    damage_by_layer = {int(row["decoder_layer"]): row for row in damage_rows}
    common_layers = sorted(set(risk_by_layer) & set(damage_by_layer))

    matched_rows = []
    total_risk = 0.0
    total_damage = 0.0
    for layer_idx in common_layers:
        risk_value = float(risk_by_layer[layer_idx]["total_risk"])
        damage_value = float(damage_by_layer[layer_idx][damage_key])
        total_risk += risk_value
        total_damage += damage_value
        matched_rows.append(
            {
                "decoder_layer": int(layer_idx),
                "total_risk": risk_value,
                "damage": damage_value,
                "damage_key": damage_key,
                "num_modules": int(risk_by_layer[layer_idx]["num_modules"]),
                "total_numel": int(risk_by_layer[layer_idx]["total_numel"]),
                "module_names": list(risk_by_layer[layer_idx]["module_names"]),
            }
        )

    for row in matched_rows:
        risk_value = row["total_risk"]
        damage_value = row["damage"]
        risk_share = risk_value / total_risk if total_risk > 0.0 else float("nan")
        damage_share = damage_value / total_damage if total_damage != 0.0 else float("nan")
        row["risk_share"] = float(risk_share)
        row["damage_share"] = float(damage_share)
        row["raw_multiplier"] = float(damage_value / (risk_value + epsilon))
        if total_risk > 0.0 and total_damage != 0.0 and risk_share > 0.0:
            row["relative_multiplier"] = float(damage_share / risk_share)
        else:
            row["relative_multiplier"] = float("nan")
        row["log_raw_multiplier"] = float(
            math.log(max(damage_value, epsilon)) - math.log(max(risk_value, epsilon))
        )

    risk_sorted = sorted(matched_rows, key=lambda item: item["total_risk"], reverse=True)
    damage_sorted = sorted(matched_rows, key=lambda item: item["damage"], reverse=True)
    for idx, row in enumerate(risk_sorted, start=1):
        row["risk_rank"] = idx
    for idx, row in enumerate(damage_sorted, start=1):
        row["damage_rank"] = idx
    for row in matched_rows:
        row["rank_gap"] = int(row["damage_rank"] - row["risk_rank"])

    matched_rows.sort(key=lambda item: item["decoder_layer"])

    risk_values = [row["total_risk"] for row in matched_rows]
    damage_values = [row["damage"] for row in matched_rows]
    relative_values = [row["relative_multiplier"] for row in matched_rows if not math.isnan(row["relative_multiplier"])]

    summary = {
        "damage_key": damage_key,
        "num_layers": int(len(matched_rows)),
        "total_risk": float(total_risk),
        "total_damage": float(total_damage),
        "pearson_risk_vs_damage": _pearson_corr(risk_values, damage_values),
        "spearman_risk_vs_damage": _spearman_corr(risk_values, damage_values),
        "mean_relative_multiplier": (
            float(sum(relative_values) / len(relative_values)) if relative_values else float("nan")
        ),
        "max_relative_multiplier_layer": None,
        "min_relative_multiplier_layer": None,
    }

    valid_rel = [row for row in matched_rows if not math.isnan(row["relative_multiplier"])]
    if valid_rel:
        max_row = max(valid_rel, key=lambda item: item["relative_multiplier"])
        min_row = min(valid_rel, key=lambda item: item["relative_multiplier"])
        summary["max_relative_multiplier_layer"] = {
            "decoder_layer": int(max_row["decoder_layer"]),
            "relative_multiplier": float(max_row["relative_multiplier"]),
        }
        summary["min_relative_multiplier_layer"] = {
            "decoder_layer": int(min_row["decoder_layer"]),
            "relative_multiplier": float(min_row["relative_multiplier"]),
        }

    return matched_rows, summary
