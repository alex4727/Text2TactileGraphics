"""Step 3: Aggregate tiling metrics for the paper table."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import load_json, mean_std, write_json
from experiments.tiling.config import MAIN_METRICS, OUTPUT_DIR, TILING_BASELINES


def aggregate(run_id: str) -> None:
    run_dir = OUTPUT_DIR / run_id
    scores_path = run_dir / "tileability_scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found: {scores_path}")
    scores = load_json(scores_path)
    timing_path = run_dir / "timing_results.json"
    timings = load_json(timing_path) if timing_path.exists() else {}

    summary: dict[str, dict] = {}
    incomplete: list[str] = []
    for baseline in TILING_BASELINES:
        results = [r for r in scores.get(baseline, []) if r.get("status") == "success"]
        if not results:
            incomplete.append(f"{baseline}: no tileability samples")
        summary[baseline] = {
            "baseline": baseline,
            "display_name": TILING_BASELINES[baseline]["display_name"],
            "stage": TILING_BASELINES[baseline]["stage"],
            "description": TILING_BASELINES[baseline]["description"],
            "n": len(results),
            "metrics": {
                metric: mean_std(r.get(metric) for r in results)
                for metric, _, _ in MAIN_METRICS
            },
            "time_s": timings.get(baseline, {}).get("time_s"),
        }
        for metric, _, _ in MAIN_METRICS:
            if summary[baseline]["metrics"][metric]["n"] != len(results):
                incomplete.append(f"{baseline}: missing {metric}")
    if incomplete:
        raise RuntimeError("Incomplete tiling metrics: " + "; ".join(incomplete))

    write_json(
        run_dir / "metrics_summary.json",
        {
            "run_id": run_id,
            "metrics": [
                {"key": k, "name": n, "direction": d} for k, n, d in MAIN_METRICS
            ],
            "by_baseline": summary,
            "timing_results": timings,
        },
    )
    write_report(run_dir / "REPORT.md", run_id, summary)
    print(f"Saved {run_dir / 'metrics_summary.json'} and {run_dir / 'REPORT.md'}")


def fmt(stat: dict, digits: int = 2) -> str:
    if stat["mean"] is None:
        return "-"
    return f"{stat['mean']:.{digits}f}"


def write_report(path: Path, run_id: str, summary: dict[str, dict]) -> None:
    lines = [
        f"# Tiling Benchmark: {run_id}",
        "",
        "| Method | Border Cont ↓ | Grad Cont ↓ | Time(s) | N |",
        "|---|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        summary.values(),
        key=lambda r: (r["stage"], list(TILING_BASELINES).index(r["baseline"])),
    )
    for row in ordered:
        time_s = "-" if row["time_s"] is None else f"{row['time_s']:.2f}"
        lines.append(
            f"| {row['display_name']} | {fmt(row['metrics']['border_avg'])} | {fmt(row['metrics']['grad_cont_avg'])} | {time_s} | {row['n']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    aggregate(args.run_id)


if __name__ == "__main__":
    main()
