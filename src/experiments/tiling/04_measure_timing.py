"""Step 4: Measure inference time for one tiling baseline."""

from __future__ import annotations

import argparse
import time

import torch
from PIL import Image

from experiments.common import (
    image_filename,
    load_json,
    load_prompts,
    write_json,
)
from experiments.tiling.config import (
    DEFAULT_HIGHPASS_FREQ,
    DEFAULT_SEED,
    OUTPUT_DIR,
    PROMPTS_PATH,
    SOURCE_BASELINE,
    SOURCE_RUN_ID,
    TEXT2TEXTURE_OUTPUT_DIR,
    TILING_BASELINES,
)
from text2tactilegraphics.config import Config
from text2tactilegraphics.generation.models import ModelManager
from text2tactilegraphics.generation.tileable_patch_generation import (
    InterTilePatchGenerator,
    IntraTilePatchGenerator,
    TiledDiffusion,
)
from text2tactilegraphics.generation.utils import center_crop as tactile_center_crop
from text2tactilegraphics.geometry.filtering import apply_high_pass_to_normal_map


def center_crop(image: Image.Image) -> Image.Image:
    return tactile_center_crop(image, 512)


def highpass(image: Image.Image) -> Image.Image:
    return apply_high_pass_to_normal_map(
        image, freq_threshold=DEFAULT_HIGHPASS_FREQ, method="per_channel"
    )


def load_sample_normal(
    t2t_run_id: str, source_baseline: str, sample_index: int, seed: int
) -> Image.Image:
    prompts, _ = load_prompts(PROMPTS_PATH)
    prompt = prompts[sample_index]
    path = (
        TEXT2TEXTURE_OUTPUT_DIR
        / t2t_run_id
        / source_baseline
        / image_filename(sample_index, prompt, seed, suffix="_normal")
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def build_gpu_generator(method: str):
    mm = ModelManager(config=Config())
    if method == "tiled_diffusion_sdxl":
        return TiledDiffusion(model_manager=mm), mm
    if method == "ours_inter":
        return InterTilePatchGenerator(model_manager=mm), mm
    if method == "ours_intra":
        return IntraTilePatchGenerator(model_manager=mm), mm
    raise ValueError(f"{method} is not a GPU tiling baseline")


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_cpu_method(method: str, normal: Image.Image):
    baseline = TILING_BASELINES[method]
    if baseline["method"] == "resize":
        return normal.resize((512, 512), Image.Resampling.LANCZOS)
    if baseline["method"] == "center_crop":
        return center_crop(normal)
    if baseline["method"] == "highpass":
        return highpass(center_crop(normal))
    raise ValueError(f"{method} is not a CPU tiling baseline")


def measure_timing(
    run_id: str,
    method: str,
    t2t_run_id: str = SOURCE_RUN_ID,
    source_baseline: str = SOURCE_BASELINE,
    sample_index: int = 0,
    seed: int = DEFAULT_SEED,
    warmup: int = 1,
    repeats: int = 1,
) -> None:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    baseline = TILING_BASELINES[method]

    normal = load_sample_normal(t2t_run_id, source_baseline, sample_index, seed)
    times: list[float] = []
    if baseline["requires_gpu"]:
        prepared = highpass(center_crop(normal))
        generator, mm = build_gpu_generator(baseline["method"])
        try:
            for _ in range(warmup):
                _ = generator.make_tileable(
                    prepared, seed=seed, **baseline.get("params", {})
                )
            for _ in range(repeats):
                synchronize_cuda()
                start = time.perf_counter()
                _ = generator.make_tileable(
                    prepared, seed=seed, **baseline.get("params", {})
                )
                synchronize_cuda()
                times.append(time.perf_counter() - start)
        finally:
            mm.unload_all_models()
    else:
        for _ in range(warmup):
            _ = run_cpu_method(method, normal)
        for _ in range(repeats):
            start = time.perf_counter()
            _ = run_cpu_method(method, normal)
            times.append(time.perf_counter() - start)

    elapsed = sum(times) / len(times)

    run_dir = OUTPUT_DIR / run_id
    timing_path = run_dir / "timing_results.json"
    timings = load_json(timing_path) if timing_path.exists() else {}
    timings[method] = {
        "time_s": float(elapsed),
        "sample_index": sample_index,
        "seed": seed,
        "t2t_run_id": t2t_run_id,
        "source_baseline": source_baseline,
        "warmup": warmup,
        "repeats": repeats,
        "times_s": [float(t) for t in times],
    }
    write_json(timing_path, timings)
    print(f"{method}: {elapsed:.2f}s")
    print(f"Saved {timing_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True, choices=list(TILING_BASELINES))
    parser.add_argument("--t2t-run-id", default=SOURCE_RUN_ID)
    parser.add_argument("--source-baseline", default=SOURCE_BASELINE)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    measure_timing(
        run_id=args.run_id,
        method=args.method,
        t2t_run_id=args.t2t_run_id,
        source_baseline=args.source_baseline,
        sample_index=args.sample_index,
        seed=args.seed,
        warmup=args.warmup,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
