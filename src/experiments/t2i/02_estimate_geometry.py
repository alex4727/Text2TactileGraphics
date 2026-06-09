"""Step 2: Estimate depth and plate masks for generated T2I images."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.common import (
    load_prompts,
    write_json,
)
from experiments.t2i.config import (
    OUTPUT_DIR,
    PLATE_CONDITIONS,
    PLATE_SEGMENTATION_CONFIDENCE,
    PLATE_SEGMENTATION_PROMPT,
    PROMPTS_PATH,
    T2I_BASELINES,
)
from text2tactilegraphics.config import Config
from text2tactilegraphics.generation.models import ModelManager
from text2tactilegraphics.generation.segmentation import SegmentationEngine
from text2tactilegraphics.generation.texture_generation import GeometryEstimator
from text2tactilegraphics.generation.utils import depth_to_image, mask_to_image


def scan_images(run_dir: Path) -> list[dict]:
    prompts, _ = load_prompts(PROMPTS_PATH)
    images: list[dict] = []
    for baseline in T2I_BASELINES:
        for plate_condition in PLATE_CONDITIONS:
            folder = run_dir / f"{baseline}_{plate_condition}"
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.png")):
                if any(token in path.stem for token in ("_depth", "_mask")):
                    continue
                match = re.match(r"(\d+)_(.+)_seed(\d+)$", path.stem)
                if not match:
                    continue
                prompt_idx = int(match.group(1))
                if prompt_idx >= len(prompts):
                    continue
                images.append(
                    {
                        "baseline": baseline,
                        "plate_condition": plate_condition,
                        "prompt_idx": prompt_idx,
                        "prompt": prompts[prompt_idx],
                        "seed": int(match.group(3)),
                        "image_path": path,
                    }
                )
    return images


def segment_plate(seg_engine, image: Image.Image) -> tuple[np.ndarray, float]:
    try:
        segments = seg_engine.segment_with_text(
            image,
            PLATE_SEGMENTATION_PROMPT,
            confidence_threshold=PLATE_SEGMENTATION_CONFIDENCE,
        )
    except Exception as exc:
        print(f"  Plate segmentation failed: {exc}")
        return np.zeros((image.height, image.width), dtype=np.float32), 0.0
    if not segments:
        return np.zeros((image.height, image.width), dtype=np.float32), 0.0
    mask, score = max(segments, key=lambda x: x[1])
    return mask.astype(np.float32), float(score)


def derive_foreground_mask(
    depth_map: np.ndarray, plate_mask: np.ndarray
) -> tuple[np.ndarray, dict]:
    h, w = depth_map.shape
    plate_coords = np.where(plate_mask > 0.5)
    if len(plate_coords[0]) == 0:
        fg_mask = (depth_map > 0.1).astype(np.float32)
        return fg_mask, {
            "plate_detected": False,
            "fg_area_ratio": float(fg_mask.sum() / (h * w)),
            "plate_area_ratio": 0.0,
            "plate_depth_mean": None,
            "plate_depth_std": None,
        }

    margin = 20
    y_min = max(int(plate_coords[0].min()) + margin, 0)
    y_max = min(int(plate_coords[0].max()) - margin, h)
    x_min = max(int(plate_coords[1].min()) + margin, 0)
    x_max = min(int(plate_coords[1].max()) - margin, w)

    inside_bbox = np.zeros((h, w), dtype=bool)
    if y_min < y_max and x_min < x_max:
        inside_bbox[y_min:y_max, x_min:x_max] = True

    plate_pixels = depth_map[plate_mask > 0.5]
    threshold = float(np.percentile(plate_pixels, 75)) if len(plate_pixels) else 0.5
    fg_mask = (inside_bbox & (plate_mask < 0.5) & (depth_map > threshold)).astype(
        np.float32
    )
    return fg_mask, {
        "plate_detected": True,
        "fg_area_ratio": float(fg_mask.sum() / (h * w)),
        "plate_area_ratio": float((plate_mask > 0.5).sum() / (h * w)),
        "plate_depth_mean": float(plate_pixels.mean()) if len(plate_pixels) else None,
        "plate_depth_std": float(plate_pixels.std()) if len(plate_pixels) else None,
    }


def estimate_geometry(run_id: str) -> None:
    run_dir = OUTPUT_DIR / run_id
    images = scan_images(run_dir)
    if not images:
        raise FileNotFoundError(f"No generated T2I images found under {run_dir}")

    config = Config()
    mm = ModelManager(config=config)
    estimator = GeometryEstimator(model_manager=mm)
    seg_engine = SegmentationEngine(model_manager=mm)
    results: list[dict] = []

    print(f"Run: {run_id}")
    print(f"Images: {len(images)}")

    for idx, info in enumerate(images, start=1):
        image_path = info["image_path"]
        print(f"[{idx}/{len(images)}] {image_path.relative_to(run_dir)}")
        record = {
            **{k: v for k, v in info.items() if k != "image_path"},
            "image_path": str(image_path.relative_to(run_dir)),
        }
        try:
            image = Image.open(image_path).convert("RGB")
            depth = estimator.compute_depth(image, normalize=True, crop=False)
            plate_mask, plate_score = segment_plate(seg_engine, image)
            fg_mask, metrics = derive_foreground_mask(depth, plate_mask)

            base = image_path.with_suffix("")
            depth_to_image(depth).save(f"{base}_depth.png")
            np.save(f"{base}_depth.npy", depth)
            mask_to_image(plate_mask).save(f"{base}_plate_mask.png")
            mask_to_image(fg_mask).save(f"{base}_fg_mask.png")

            results.append(
                {
                    **record,
                    "plate_score": plate_score,
                    "metrics": metrics,
                    "status": "success",
                }
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({**record, "status": "error", "error": str(exc)})

    write_json(run_dir / "geometry_results.json", results)
    print(f"Saved {run_dir / 'geometry_results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    estimate_geometry(args.run_id)


if __name__ == "__main__":
    main()
