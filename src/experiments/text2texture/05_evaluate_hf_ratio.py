"""Step 5: Evaluate HF ratio on normal maps."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.common import load_prompts, write_json
from experiments.text2texture.config import (
    DEFAULT_FREQ_THRESHOLD,
    OUTPUT_DIR,
    PROMPTS_PATH,
    TEXT2TEXTURE_BASELINES,
)
from tactilegen.geometry.filtering import apply_high_pass_to_normal_map


def scan_normal_maps(run_dir: Path) -> list[dict]:
    prompts, _ = load_prompts(PROMPTS_PATH)
    images: list[dict] = []
    for baseline in TEXT2TEXTURE_BASELINES:
        folder = run_dir / baseline
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*_normal.png")):
            match = re.match(r"(\d+)_(.+)_seed(\d+)_normal$", path.stem)
            if not match:
                continue
            prompt_idx = int(match.group(1))
            if prompt_idx >= len(prompts):
                continue
            images.append(
                {
                    "baseline": baseline,
                    "prompt_idx": prompt_idx,
                    "prompt": prompts[prompt_idx],
                    "seed": int(match.group(3)),
                    "normal_path": path,
                }
            )
    return images


def compute_signal_energy(image: np.ndarray) -> float:
    image = image.astype(np.float32)
    if image.min() >= 0 and image.max() <= 1:
        image = image * 2.0 - 1.0
    total_energy = 0.0
    for channel_idx in range(image.shape[2]):
        channel = image[:, :, channel_idx] - image[:, :, channel_idx].mean()
        energy = np.abs(np.fft.fft2(channel)) ** 2
        energy[0, 0] = 0.0
        total_energy += float(np.sum(energy))
    return total_energy


def compute_hf_ratio(normal_rgb: np.ndarray, freq_threshold: int) -> dict:
    normal_pil = Image.fromarray(normal_rgb)
    normal_01 = normal_rgb.astype(np.float32) / 255.0
    energy_before = compute_signal_energy(normal_01)
    if energy_before <= 0:
        return {"hf_ratio": 0.0, "energy_before": 0.0, "energy_after": 0.0}

    filtered = apply_high_pass_to_normal_map(
        normal_pil,
        freq_threshold=freq_threshold,
        method="per_channel",
    )
    filtered_01 = np.asarray(filtered, dtype=np.float32) / 255.0
    energy_after = compute_signal_energy(filtered_01)
    return {
        "hf_ratio": float(np.clip(energy_after / energy_before, 0.0, 1.0)),
        "energy_before": float(energy_before),
        "energy_after": float(energy_after),
    }


def evaluate_hf_ratio(
    run_id: str, freq_threshold: int = DEFAULT_FREQ_THRESHOLD
) -> None:
    run_dir = OUTPUT_DIR / run_id
    images = scan_normal_maps(run_dir)
    if not images:
        raise FileNotFoundError(
            f"No normal maps found under {run_dir}; run 02_estimate_normals.py first"
        )

    results: list[dict] = []
    print(f"Run: {run_id}")
    print(f"Normal maps: {len(images)}")
    print(f"freq_threshold={freq_threshold}")

    for idx, info in enumerate(images, start=1):
        normal_path = info["normal_path"]
        print(f"[{idx}/{len(images)}] {normal_path.relative_to(run_dir)}")
        record = {
            **{k: v for k, v in info.items() if k != "normal_path"},
            "normal_path": str(normal_path.relative_to(run_dir)),
        }
        try:
            normal_rgb = np.asarray(Image.open(normal_path).convert("RGB"))
            metrics = compute_hf_ratio(normal_rgb, freq_threshold)
            results.append(
                {
                    **record,
                    "hf_ratio": metrics["hf_ratio"],
                    "freq_threshold": freq_threshold,
                    "energy_before": metrics["energy_before"],
                    "energy_after": metrics["energy_after"],
                    "status": "success",
                }
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({**record, "status": "error", "error": str(exc)})

    write_json(run_dir / f"hf_ratio_scores_ft{freq_threshold}.json", results)
    if freq_threshold == DEFAULT_FREQ_THRESHOLD:
        write_json(run_dir / "hf_ratio_scores.json", results)
    print(f"Saved {run_dir / f'hf_ratio_scores_ft{freq_threshold}.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--freq-threshold", type=int, default=DEFAULT_FREQ_THRESHOLD)
    args = parser.parse_args()
    evaluate_hf_ratio(args.run_id, freq_threshold=args.freq_threshold)


if __name__ == "__main__":
    main()
