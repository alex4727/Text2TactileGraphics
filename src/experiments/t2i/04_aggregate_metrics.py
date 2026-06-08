"""Step 4: Aggregate T2I benchmark metrics for the paper table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.common import load_json, mean_std, write_json
from experiments.t2i.config import (
    OUTPUT_DIR,
    PLATE_CONDITIONS,
    T2I_BASELINES,
)


def make_key(record: dict) -> tuple:
    return (
        record["baseline"],
        record["plate_condition"],
        int(record["prompt_idx"]),
        int(record["seed"]),
    )


def load_success(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [r for r in load_json(path) if r.get("status") == "success"]


def aggregate(run_id: str) -> None:
    run_dir = OUTPUT_DIR / run_id
    geometry = load_success(run_dir / "geometry_results.json")
    clip = load_success(run_dir / "clip_results.json")
    if not geometry and not clip:
        raise FileNotFoundError(f"No T2I metric files found under {run_dir}")

    merged: dict[tuple, dict] = {}
    for record in geometry:
        metrics = record.get("metrics", {})
        merged[make_key(record)] = {
            "baseline": record["baseline"],
            "plate_condition": record["plate_condition"],
            "prompt_idx": record["prompt_idx"],
            "prompt": record["prompt"],
            "seed": record["seed"],
            "image_path": record["image_path"],
            "plate_detected": metrics.get("plate_detected"),
            "fg_area_ratio": metrics.get("fg_area_ratio"),
            "plate_area_ratio": metrics.get("plate_area_ratio"),
            "plate_depth_mean": metrics.get("plate_depth_mean"),
            "plate_depth_std": metrics.get("plate_depth_std"),
            "has_geometry": True,
        }
    for record in clip:
        item = merged.setdefault(
            make_key(record),
            {
                "baseline": record["baseline"],
                "plate_condition": record["plate_condition"],
                "prompt_idx": record["prompt_idx"],
                "prompt": record["prompt"],
                "seed": record["seed"],
                "image_path": record["image_path"],
            },
        )
        clip_score = record.get("clip_score")
        item["clip_score"] = clip_score
        item["has_clip"] = clip_score is not None

    rows = list(merged.values())
    summary: dict[str, dict] = {}
    incomplete: list[str] = []
    for baseline in T2I_BASELINES:
        for plate_condition in PLATE_CONDITIONS:
            subset = [
                r
                for r in rows
                if r["baseline"] == baseline and r["plate_condition"] == plate_condition
            ]
            complete_subset = [
                r for r in subset if r.get("has_geometry") and r.get("has_clip")
            ]
            if len(complete_subset) != len(subset):
                incomplete.append(
                    f"{baseline}_{plate_condition}: complete={len(complete_subset)} total={len(subset)}"
                )
            if not subset:
                incomplete.append(f"{baseline}_{plate_condition}: no samples")
            plate_subset = [r for r in complete_subset if r.get("plate_detected")]
            plate_stats = plate_height_stats(plate_subset)
            key = f"{baseline}_{plate_condition}"
            summary[key] = {
                "baseline": baseline,
                "display_name": T2I_BASELINES[baseline]["display_name"],
                "plate_condition": plate_condition,
                "plate_display": PLATE_CONDITIONS[plate_condition]["display_name"],
                "n": len(complete_subset),
                "n_total": len(subset),
                "n_clip": sum(1 for r in subset if r.get("has_clip")),
                "n_geometry": sum(1 for r in subset if r.get("has_geometry")),
                "n_plate_detected": len(plate_subset),
                "plate_detect_rate": len(plate_subset) / len(subset)
                if subset
                else None,
                "fg_area_ratio": mean_std(
                    r.get("fg_area_ratio") for r in complete_subset
                ),
                "clip": mean_std(r.get("clip_score") for r in complete_subset),
                "plate_height_avg": mean_std(
                    r.get("plate_depth_mean") for r in plate_subset
                ),
                "plate_height_within": plate_stats["within"],
                "plate_height_between": plate_stats["between"],
                "plate_height_var": plate_stats["total"],
            }
    if incomplete:
        raise RuntimeError("Incomplete T2I metrics: " + "; ".join(incomplete))

    write_json(run_dir / "all_metrics.json", rows)
    write_json(
        run_dir / "metrics_summary.json",
        {
            "run_id": run_id,
            "plate_height_definition": "avg=mean(mean_i), within=sqrt(mean(std_i^2)), between=sample_std(mean_i), total=sqrt(mean(std_i^2)+sample_var(mean_i)).",
            "by_baseline_and_plate": summary,
        },
    )
    write_report(run_dir / "REPORT.md", run_id, summary)
    print(f"Saved {run_dir / 'metrics_summary.json'} and {run_dir / 'REPORT.md'}")


def fmt(stat: dict, digits: int = 3) -> str:
    if stat["mean"] is None:
        return "-"
    return f"{stat['mean']:.{digits}f}"


def plate_height_stats(
    records: list[dict],
) -> dict[str, dict[str, float | int | None]]:
    means = np.array(
        [
            r["plate_depth_mean"]
            for r in records
            if r.get("plate_depth_mean") is not None
        ],
        dtype=np.float64,
    )
    stds = np.array(
        [r["plate_depth_std"] for r in records if r.get("plate_depth_std") is not None],
        dtype=np.float64,
    )
    if len(means) == 0 or len(stds) == 0:
        empty = {"mean": None, "std": None, "n": 0}
        return {"within": empty, "between": empty, "total": empty}

    within_variance = float(np.mean(stds**2))
    between_variance = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
    n = int(min(len(means), len(stds)))
    return {
        "within": {
            "mean": float(np.sqrt(within_variance)),
            "std": None,
            "n": int(len(stds)),
        },
        "between": {
            "mean": float(np.sqrt(between_variance)),
            "std": None,
            "n": int(len(means)),
        },
        "total": {
            "mean": float(np.sqrt(within_variance + between_variance)),
            "std": None,
            "n": n,
            "within_variance": within_variance,
            "between_variance": between_variance,
        },
    }


def write_report(path: Path, run_id: str, summary: dict[str, dict]) -> None:
    lines = [
        f"# T2I Benchmark: {run_id}",
        "",
        "## Main Table",
        "",
        "| Method | Plate Cond | CLIP ↑ | Plate Height Avg | Plate Height Var ↓ | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        summary.values(),
        key=lambda r: (
            list(T2I_BASELINES).index(r["baseline"]),
            list(PLATE_CONDITIONS).index(r["plate_condition"]),
        ),
    )
    for row in ordered:
        lines.append(
            f"| {row['display_name']} | {row['plate_display']} | {fmt(row['clip'], 2)} | {fmt(row['plate_height_avg'], 3)} | {fmt(row['plate_height_var'], 3)} | {row['n']} |"
        )
    lines.extend(
        [
            "",
            "## Table A1",
            "",
            "| Method | Plate Cond | FG ↓ | Within ↓ | Between ↓ | Total ↓ | N |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ordered:
        lines.append(
            f"| {row['display_name']} | {row['plate_display']} | {fmt(row['fg_area_ratio'], 3)} | {fmt(row['plate_height_within'], 3)} | {fmt(row['plate_height_between'], 3)} | {fmt(row['plate_height_var'], 3)} | {row['n']} |"
        )
    lines.extend(
        [
            "",
            "Plate Height Total follows sqrt(mean(std_i^2) + sample_var(mean_i)). Within is sqrt(mean(std_i^2)); Between is sample_std(mean_i).",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    aggregate(args.run_id)


if __name__ == "__main__":
    main()
