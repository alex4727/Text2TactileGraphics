"""Step 6: Aggregate text-to-texture metrics for the paper table."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import load_json, mean_std, write_json
from experiments.text2texture.config import (
    DEFAULT_FREQ_THRESHOLD,
    OUTPUT_DIR,
    TEXT2TEXTURE_BASELINES,
)


def make_key(record: dict) -> tuple:
    return (record["baseline"], int(record["prompt_idx"]), int(record["seed"]))


def load_success(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [r for r in load_json(path) if r.get("status") == "success"]


def aggregate(run_id: str, freq_threshold: int = DEFAULT_FREQ_THRESHOLD) -> None:
    run_dir = OUTPUT_DIR / run_id
    clip = load_success(run_dir / "clip_scores.json")
    patch = load_success(run_dir / "patch_self_similarity_scores.json")
    hf = load_success(run_dir / f"hf_ratio_scores_ft{freq_threshold}.json")
    if not hf:
        hf = load_success(run_dir / "hf_ratio_scores.json")
    if not clip and not patch and not hf:
        raise FileNotFoundError(f"No text2texture metric files found under {run_dir}")

    merged: dict[tuple, dict] = {}
    for record in clip:
        merged[make_key(record)] = {
            "baseline": record["baseline"],
            "prompt_idx": record["prompt_idx"],
            "prompt": record["prompt"],
            "seed": record["seed"],
            "image_path": record["image_path"],
            "clip_score": record.get("clip_score"),
        }
    for record in patch:
        item = merged.setdefault(
            make_key(record),
            {
                "baseline": record["baseline"],
                "prompt_idx": record["prompt_idx"],
                "prompt": record["prompt"],
                "seed": record["seed"],
            },
        )
        item["normal_path"] = record.get("normal_path")
        item["patch_lpips"] = record.get("patch_lpips")
    for record in hf:
        item = merged.setdefault(
            make_key(record),
            {
                "baseline": record["baseline"],
                "prompt_idx": record["prompt_idx"],
                "prompt": record["prompt"],
                "seed": record["seed"],
            },
        )
        item["normal_path"] = item.get("normal_path") or record.get("normal_path")
        item["hf_ratio"] = record.get("hf_ratio")

    rows = list(merged.values())
    summary: dict[str, dict] = {}
    incomplete: list[str] = []
    for baseline in TEXT2TEXTURE_BASELINES:
        subset = [r for r in rows if r["baseline"] == baseline]
        complete_subset = [
            r
            for r in subset
            if r.get("clip_score") is not None
            and r.get("patch_lpips") is not None
            and r.get("hf_ratio") is not None
        ]
        if len(complete_subset) != len(subset):
            incomplete.append(
                f"{baseline}: complete={len(complete_subset)} total={len(subset)}"
            )
        if not subset:
            incomplete.append(f"{baseline}: no samples")
        summary[baseline] = {
            "baseline": baseline,
            "display_name": TEXT2TEXTURE_BASELINES[baseline]["display_name"],
            "description": TEXT2TEXTURE_BASELINES[baseline]["description"],
            "n": len(complete_subset),
            "n_total": len(subset),
            "n_clip": sum(1 for r in subset if r.get("clip_score") is not None),
            "n_patch": sum(1 for r in subset if r.get("patch_lpips") is not None),
            "n_hf": sum(1 for r in subset if r.get("hf_ratio") is not None),
            "clip": mean_std(r.get("clip_score") for r in complete_subset),
            "patch_self_sim": mean_std(r.get("patch_lpips") for r in complete_subset),
            "hf_ratio": mean_std(r.get("hf_ratio") for r in complete_subset),
        }
    if incomplete:
        raise RuntimeError("Incomplete text2texture metrics: " + "; ".join(incomplete))

    write_json(run_dir / "all_metrics.json", rows)
    write_json(
        run_dir / "metrics_summary.json",
        {
            "run_id": run_id,
            "patch_self_sim": "patch_lpips",
            "freq_threshold": freq_threshold,
            "by_baseline": summary,
        },
    )
    write_report(run_dir / "REPORT.md", run_id, summary)
    print(f"Saved {run_dir / 'metrics_summary.json'} and {run_dir / 'REPORT.md'}")


def fmt(stat: dict, digits: int = 3) -> str:
    if stat["mean"] is None:
        return "-"
    return f"{stat['mean']:.{digits}f}"


def write_report(path: Path, run_id: str, summary: dict[str, dict]) -> None:
    lines = [
        f"# Text2Texture Benchmark: {run_id}",
        "",
        "| Method | CLIP ↑ | Patch Self-Sim ↓ | HF Ratio ↑ | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for baseline in TEXT2TEXTURE_BASELINES:
        row = summary[baseline]
        lines.append(
            f"| {row['display_name']} | {fmt(row['clip'], 2)} | {fmt(row['patch_self_sim'], 3)} | {fmt(row['hf_ratio'], 3)} | {row['n']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--freq-threshold", type=int, default=DEFAULT_FREQ_THRESHOLD)
    args = parser.parse_args()
    aggregate(args.run_id, freq_threshold=args.freq_threshold)


if __name__ == "__main__":
    main()
