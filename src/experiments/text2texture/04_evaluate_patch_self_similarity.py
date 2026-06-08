"""Step 4: Evaluate patch self-similarity with LPIPS."""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image

from experiments.common import load_prompts, write_json
from experiments.text2texture.config import (
    DEFAULT_N_PATCHES,
    DEFAULT_PATCH_SIZE,
    OUTPUT_DIR,
    PROMPTS_PATH,
    TEXT2TEXTURE_BASELINES,
)


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


def sample_random_patches(
    img: np.ndarray, n_patches: int, patch_size: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image {w}x{h} is smaller than patch_size={patch_size}")
    patches = []
    for _ in range(n_patches):
        y = int(rng.integers(0, h - patch_size + 1))
        x = int(rng.integers(0, w - patch_size + 1))
        patches.append(img[y : y + patch_size, x : x + patch_size])
    return patches


def compute_pairwise_lpips(
    patches: list[np.ndarray], lpips_model, device: str
) -> float:
    scores: list[float] = []
    for p1, p2 in combinations(patches, 2):
        t1 = (
            torch.from_numpy(p1.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        t2 = (
            torch.from_numpy(p2.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            scores.append(float(lpips_model(t1, t2, normalize=True).item()))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_patch_self_similarity(
    run_id: str,
    device: str = "cuda:0",
    n_patches: int = DEFAULT_N_PATCHES,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> None:
    run_dir = OUTPUT_DIR / run_id
    images = scan_normal_maps(run_dir)
    if not images:
        raise FileNotFoundError(
            f"No normal maps found under {run_dir}; run 02_estimate_normals.py first"
        )

    lpips_model = lpips.LPIPS(net="vgg").to(device)
    lpips_model.eval()
    results: list[dict] = []

    print(f"Run: {run_id}")
    print(f"Normal maps: {len(images)}")

    for idx, info in enumerate(images, start=1):
        normal_path = info["normal_path"]
        print(f"[{idx}/{len(images)}] {normal_path.relative_to(run_dir)}")
        record = {
            **{k: v for k, v in info.items() if k != "normal_path"},
            "normal_path": str(normal_path.relative_to(run_dir)),
        }
        try:
            normal_rgb = np.asarray(Image.open(normal_path).convert("RGB"))
            patches = sample_random_patches(
                normal_rgb, n_patches, patch_size, seed=info["seed"]
            )
            results.append(
                {
                    **record,
                    "patch_lpips": compute_pairwise_lpips(patches, lpips_model, device),
                    "n_patches": n_patches,
                    "patch_size": patch_size,
                    "status": "success",
                }
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({**record, "status": "error", "error": str(exc)})

    write_json(run_dir / "patch_self_similarity_scores.json", results)
    print(f"Saved {run_dir / 'patch_self_similarity_scores.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-patches", type=int, default=DEFAULT_N_PATCHES)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    args = parser.parse_args()
    evaluate_patch_self_similarity(
        args.run_id,
        device=args.device,
        n_patches=args.n_patches,
        patch_size=args.patch_size,
    )


if __name__ == "__main__":
    main()
