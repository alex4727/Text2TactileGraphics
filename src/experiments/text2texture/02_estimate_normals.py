"""Step 2: Estimate OpenGL normal maps for generated textures."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

from experiments.common import (
    load_prompts,
    write_json,
)
from experiments.text2texture.config import (
    OUTPUT_DIR,
    PROMPTS_PATH,
    TEXT2TEXTURE_BASELINES,
)
from tactilegen.config import Config
from tactilegen.generation.models import ModelManager
from tactilegen.generation.texture_generation import GeometryEstimator
from tactilegen.generation.utils import normal_to_image


def scan_texture_images(run_dir: Path, baselines: list[str]) -> list[dict]:
    prompts, _ = load_prompts(PROMPTS_PATH)
    images: list[dict] = []
    for baseline in baselines:
        folder = run_dir / baseline
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.png")):
            if path.stem.endswith("_normal"):
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
                    "prompt_idx": prompt_idx,
                    "prompt": prompts[prompt_idx],
                    "seed": int(match.group(3)),
                    "image_path": path,
                }
            )
    return images


def estimate_normals(run_id: str, baselines: list[str] | None = None) -> None:
    selected_baselines = baselines or list(TEXT2TEXTURE_BASELINES)
    run_dir = OUTPUT_DIR / run_id
    images = scan_texture_images(run_dir, selected_baselines)
    if not images:
        raise FileNotFoundError(f"No generated texture images found under {run_dir}")

    mm = ModelManager(config=Config())
    estimator = GeometryEstimator(model_manager=mm)
    results: list[dict] = []

    print(f"Run: {run_id}")
    print(f"Images: {len(images)}")

    for idx, info in enumerate(images, start=1):
        image_path = info["image_path"]
        out_path = image_path.with_name(f"{image_path.stem}_normal.png")
        print(f"[{idx}/{len(images)}] {image_path.relative_to(run_dir)}")
        record = {
            **{k: v for k, v in info.items() if k != "image_path"},
            "image_path": str(image_path.relative_to(run_dir)),
            "normal_path": str(out_path.relative_to(run_dir)),
        }
        if out_path.exists():
            results.append({**record, "status": "skipped"})
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            normal = estimator.compute_normal(image, crop=False)
            normal_to_image(normal).save(out_path)
            results.append({**record, "status": "success", "normal_format": "opengl"})
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({**record, "status": "error", "error": str(exc)})

    write_json(run_dir / "normal_estimation_results.json", results)
    print(f"Saved {run_dir / 'normal_estimation_results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--baselines", nargs="+", choices=list(TEXT2TEXTURE_BASELINES), default=None
    )
    args = parser.parse_args()
    estimate_normals(args.run_id, baselines=args.baselines)


if __name__ == "__main__":
    main()
