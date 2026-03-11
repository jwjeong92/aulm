#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.bitflip_risk import (
    aggregate_decoder_layer_risks,
    estimate_decoder_layer_multipliers,
)


def build_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_csv_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk_csv", type=str, required=True)
    parser.add_argument("--damage_csv", type=str, required=True)
    parser.add_argument("--damage_key", type=str, choices=["delta_ppl", "delta_loss"], default="delta_ppl")
    parser.add_argument(
        "--damage_transform",
        type=str,
        choices=["abs", "identity", "relu"],
        default="abs",
    )
    parser.add_argument("--out_dir", type=str, default="./cache/decoder_layer_multiplier")
    parser.add_argument("--risk_label", type=str, default="local_risk")
    parser.add_argument("--damage_label", type=str, default="layer_damage")
    parser.add_argument("--note", type=str, default="")
    return parser.parse_args()


def transform_damage_value(value, transform):
    if transform == "identity":
        return float(value)
    if transform == "abs":
        return float(abs(value))
    if transform == "relu":
        return float(max(value, 0.0))
    raise ValueError(f"Unsupported damage_transform: {transform}")


def normalize_damage_rows(rows, damage_key, damage_transform):
    normalized = []
    for row in rows:
        if "target_decoder_layer" not in row:
            raise ValueError("damage_csv must contain target_decoder_layer.")
        raw_damage = float(row[damage_key])
        normalized.append(
            {
                "decoder_layer": int(row["target_decoder_layer"]),
                "delta_ppl": float(row.get("delta_ppl", "nan")),
                "delta_loss": float(row.get("delta_loss", "nan")),
                "mean_decoder_nmse": float(row.get("mean_decoder_nmse", "nan")),
                "mean_decoder_mse": float(row.get("mean_decoder_mse", "nan")),
                "damage_raw": raw_damage,
                "damage_transform": damage_transform,
                damage_key: transform_damage_value(raw_damage, damage_transform),
            }
        )
    return normalized


def make_markdown(summary, matched_rows, aggregate_meta, args, risk_agg_csv, multiplier_csv):
    top_rows = sorted(
        matched_rows,
        key=lambda item: item["relative_multiplier"],
        reverse=True,
    )
    bottom_rows = sorted(
        matched_rows,
        key=lambda item: item["relative_multiplier"],
    )

    lines = []
    lines.append("# Decoder-Layer Multiplier Estimate")
    lines.append("")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- local risk csv: `{args.risk_csv}`")
    lines.append(f"- damage csv: `{args.damage_csv}`")
    lines.append(f"- damage metric: `{args.damage_key}`")
    lines.append(f"- damage transform: `{args.damage_transform}`")
    lines.append(f"- risk label: `{args.risk_label}`")
    lines.append(f"- damage label: `{args.damage_label}`")
    if args.note:
        lines.append(f"- note: `{args.note}`")
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append("For decoder layer `ell`, let:")
    lines.append("")
    lines.append("- `R_ell`: aggregated local decoder-layer risk")
    lines.append(
        f"- `D_ell`: observed decoder-layer damage from `{args.damage_transform}({args.damage_key})`"
    )
    lines.append("")
    lines.append("This script reports two multipliers:")
    lines.append("")
    lines.append("- raw multiplier")
    lines.append("  `alpha_raw(ell) = D_ell / R_ell`")
    lines.append("- relative multiplier")
    lines.append("  `alpha_rel(ell) = (D_ell / sum_j D_j) / (R_ell / sum_j R_j)`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- `alpha_rel(ell) > 1`: this decoder layer causes more task damage than its local risk share would suggest")
    lines.append("- `alpha_rel(ell) < 1`: this decoder layer causes less task damage than its local risk share would suggest")
    lines.append("- `alpha_rel(ell)` is the better provisional multiplier for rescaling local bit scores because it is dimensionless")
    if args.damage_transform == "abs":
        lines.append("- because `abs` is used, the multiplier is a damage-magnitude correction, not a signed effect predictor")
    lines.append("")
    lines.append("## Aggregate Summary")
    lines.append("")
    lines.append(f"- matched decoder layers: `{summary['num_layers']}`")
    lines.append(f"- total local risk: `{summary['total_risk']}`")
    lines.append(f"- total observed damage: `{summary['total_damage']}`")
    lines.append(f"- pearson(local risk, damage): `{summary['pearson_risk_vs_damage']}`")
    lines.append(f"- spearman(local risk, damage): `{summary['spearman_risk_vs_damage']}`")
    lines.append(f"- skipped module rows during risk aggregation: `{aggregate_meta['skipped_rows']}`")
    lines.append(f"- aggregated decoder-layer risk csv: `{risk_agg_csv}`")
    lines.append(f"- multiplier csv: `{multiplier_csv}`")
    lines.append("")
    if summary["max_relative_multiplier_layer"] is not None:
        lines.append("## Strongest Underestimated Layers by Local Risk")
        lines.append("")
        for row in top_rows[:5]:
            lines.append(
                f"- layer {row['decoder_layer']}: rel={row['relative_multiplier']:.6f}, "
                f"raw={row['raw_multiplier']:.6f}, risk={row['total_risk']:.6f}, damage={row['damage']:.6f}, "
                f"risk_rank={row['risk_rank']}, damage_rank={row['damage_rank']}"
            )
        lines.append("")
        lines.append("## Strongest Overestimated Layers by Local Risk")
        lines.append("")
        for row in bottom_rows[:5]:
            lines.append(
                f"- layer {row['decoder_layer']}: rel={row['relative_multiplier']:.6f}, "
                f"raw={row['raw_multiplier']:.6f}, risk={row['total_risk']:.6f}, damage={row['damage']:.6f}, "
                f"risk_rank={row['risk_rank']}, damage_rank={row['damage_rank']}"
            )
        lines.append("")
    lines.append("## Provisional Use")
    lines.append("")
    lines.append("A provisional layer-aware bit score would be:")
    lines.append("")
    lines.append("`score_final(r, c, k) = alpha_rel(layer(r,c)) * score_local(r, c, k)`")
    lines.append("")
    lines.append("This is still provisional. If the two source experiments are mismatched, these multipliers should be treated as hypothesis-generating, not deployment-ready calibration.")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    risk_rows = load_csv_rows(args.risk_csv)
    damage_rows = normalize_damage_rows(
        load_csv_rows(args.damage_csv),
        args.damage_key,
        args.damage_transform,
    )

    decoder_risk_rows, aggregate_meta = aggregate_decoder_layer_risks(risk_rows)
    multiplier_rows, summary = estimate_decoder_layer_multipliers(
        decoder_risk_rows=decoder_risk_rows,
        damage_rows=damage_rows,
        damage_key=args.damage_key,
    )

    timestamp = build_timestamp()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    risk_agg_csv = out_dir / f"{timestamp}_decoder_layer_risk.csv"
    multiplier_csv = out_dir / f"{timestamp}_decoder_layer_multiplier.csv"
    summary_json = out_dir / f"{timestamp}_summary.json"
    summary_md = out_dir / f"{timestamp}_summary.md"

    save_csv(decoder_risk_rows, risk_agg_csv)
    save_csv(multiplier_rows, multiplier_csv)

    payload = {
        "timestamp": timestamp,
        "risk_csv": args.risk_csv,
        "damage_csv": args.damage_csv,
        "risk_label": args.risk_label,
        "damage_label": args.damage_label,
        "damage_key": args.damage_key,
        "damage_transform": args.damage_transform,
        "note": args.note,
        "aggregate_meta": aggregate_meta,
        "summary": summary,
        "top_relative_multiplier_rows": sorted(
            multiplier_rows, key=lambda item: item["relative_multiplier"], reverse=True
        )[:5],
        "bottom_relative_multiplier_rows": sorted(
            multiplier_rows, key=lambda item: item["relative_multiplier"]
        )[:5],
        "decoder_layer_risk_csv": str(risk_agg_csv),
        "decoder_layer_multiplier_csv": str(multiplier_csv),
    }

    with open(summary_json, "w") as handle:
        json.dump(payload, handle, indent=2)

    with open(summary_md, "w") as handle:
        handle.write(
            make_markdown(
                summary=summary,
                matched_rows=multiplier_rows,
                aggregate_meta=aggregate_meta,
                args=args,
                risk_agg_csv=risk_agg_csv,
                multiplier_csv=multiplier_csv,
            )
        )

    print(f"decoder_layer_risk_csv={risk_agg_csv}")
    print(f"decoder_layer_multiplier_csv={multiplier_csv}")
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()
