"""Step 2: Evaluate paper tileability metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.common import (
    image_filename,
    load_json,
    load_prompts,
    prompt_slice,
    write_json,
)
from experiments.tiling.config import (
    DEFAULT_SEED,
    OUTPUT_DIR,
    PROMPTS_PATH,
    TILING_BASELINES,
)


def compute_border_discontinuity(img: np.ndarray, direction: str) -> float:
    img = img.astype(np.float32)
    if direction == "x":
        edge_1 = img[:, -1]
        edge_2 = img[:, 0]
    elif direction == "y":
        edge_1 = img[-1, :]
        edge_2 = img[0, :]
    else:
        raise ValueError("direction must be 'x' or 'y'")
    return float(np.mean(np.abs(edge_1 - edge_2)))


def compute_gradient_continuity_at_border(img: np.ndarray, direction: str) -> float:
    img = img.astype(np.float32)
    if direction == "x":
        grad_at_right = img[:, -1] - img[:, -2]
        grad_at_left = img[:, 1] - img[:, 0]
        grad_across_seam = img[:, 0] - img[:, -1]
        discontinuity = np.abs(grad_across_seam - grad_at_right) + np.abs(
            grad_at_left - grad_across_seam
        )
    elif direction == "y":
        grad_at_bottom = img[-1, :] - img[-2, :]
        grad_at_top = img[1, :] - img[0, :]
        grad_across_seam = img[0, :] - img[-1, :]
        discontinuity = np.abs(grad_across_seam - grad_at_bottom) + np.abs(
            grad_at_top - grad_across_seam
        )
    else:
        raise ValueError("direction must be 'x' or 'y'")
    return float(np.mean(discontinuity))


def compute_paper_tileability_metrics(image: Image.Image) -> dict[str, float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    border_x = compute_border_discontinuity(arr, "x")
    border_y = compute_border_discontinuity(arr, "y")
    grad_x = compute_gradient_continuity_at_border(arr, "x")
    grad_y = compute_gradient_continuity_at_border(arr, "y")
    return {
        "border_x": border_x,
        "border_y": border_y,
        "border_avg": float((border_x + border_y) / 2.0),
        "grad_cont_x": grad_x,
        "grad_cont_y": grad_y,
        "grad_cont_avg": float((grad_x + grad_y) / 2.0),
    }


def evaluate_tileability(
    run_id: str,
    baselines: list[str] | None = None,
    prompt_range: tuple[int, int] | None = None,
    seed: int = DEFAULT_SEED,
) -> None:
    selected_baselines = baselines or list(TILING_BASELINES)
    prompts, _ = load_prompts(PROMPTS_PATH)
    run_dir = OUTPUT_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    all_results: dict[str, list[dict]] = {
        baseline: [] for baseline in selected_baselines
    }
    processed = 0
    missing = 0

    print(f"Run: {run_id}")
    print(f"Baselines: {selected_baselines}")

    for baseline_name in selected_baselines:
        img_dir = run_dir / baseline_name
        if not img_dir.exists():
            print(f"Missing directory: {img_dir}")
            continue

        for prompt_idx in prompt_slice(prompts, prompt_range):
            prompt = prompts[prompt_idx]
            img_path = img_dir / image_filename(prompt_idx, prompt, seed)
            if not img_path.exists():
                missing += 1
                continue
            image = Image.open(img_path).convert("RGB")
            all_results[baseline_name].append(
                {
                    "baseline": baseline_name,
                    "stage": TILING_BASELINES[baseline_name]["stage"],
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "seed": seed,
                    "image_path": str(img_path.relative_to(run_dir)),
                    **compute_paper_tileability_metrics(image),
                    "status": "success",
                }
            )
            processed += 1

    scores_path = run_dir / "tileability_scores.json"
    write_json(scores_path, merge_tileability_scores(scores_path, all_results))
    print(f"Saved {run_dir / 'tileability_scores.json'}")
    print(f"Processed={processed}, missing={missing}")


def merge_tileability_scores(
    path: Path, new_results: dict[str, list[dict]]
) -> dict[str, list[dict]]:
    if not path.exists():
        return new_results
    merged = load_json(path)
    for baseline, records in new_results.items():
        by_key = {
            (int(r["prompt_idx"]), int(r["seed"])): r for r in merged.get(baseline, [])
        }
        for record in records:
            by_key[(int(record["prompt_idx"]), int(record["seed"]))] = record
        merged[baseline] = list(by_key.values())
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--baselines", nargs="+", choices=list(TILING_BASELINES), default=None
    )
    parser.add_argument(
        "--prompts", nargs=2, type=int, metavar=("START", "END"), default=None
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    evaluate_tileability(
        run_id=args.run_id,
        baselines=args.baselines,
        prompt_range=tuple(args.prompts) if args.prompts else None,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
