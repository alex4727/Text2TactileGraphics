"""Step 3: Evaluate CLIP alignment for generated texture images."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

from experiments.common import ClipScorer, load_prompts, write_json
from experiments.text2texture.config import (
    OUTPUT_DIR,
    PROMPTS_PATH,
    TEXT2TEXTURE_BASELINES,
    TEXTURE_PROMPT_TEMPLATE,
)


def scan_images(run_dir: Path) -> list[dict]:
    prompts, _ = load_prompts(PROMPTS_PATH)
    images: list[dict] = []
    for baseline in TEXT2TEXTURE_BASELINES:
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


def evaluate_clip(run_id: str, device: str = "cuda:0") -> None:
    run_dir = OUTPUT_DIR / run_id
    images = scan_images(run_dir)
    if not images:
        raise FileNotFoundError(f"No generated texture images found under {run_dir}")

    scorer = ClipScorer(device=device)
    results: list[dict] = []

    print(f"Run: {run_id}")
    print(f"Images: {len(images)}")

    for idx, info in enumerate(images, start=1):
        image_path = info["image_path"]
        prompt = info["prompt"]
        print(f"[{idx}/{len(images)}] {image_path.relative_to(run_dir)}")
        record = {
            **{k: v for k, v in info.items() if k != "image_path"},
            "image_path": str(image_path.relative_to(run_dir)),
        }
        try:
            image = Image.open(image_path).convert("RGB")
            clip_prompt = TEXTURE_PROMPT_TEMPLATE.format(prompt.strip())
            results.append(
                {
                    **record,
                    "clip_score": scorer.score(image, clip_prompt),
                    "status": "success",
                }
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({**record, "status": "error", "error": str(exc)})

    write_json(run_dir / "clip_scores.json", results)
    print(f"Saved {run_dir / 'clip_scores.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate_clip(args.run_id, device=args.device)


if __name__ == "__main__":
    main()
