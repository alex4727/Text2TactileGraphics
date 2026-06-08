"""Step 1: Generate tiled normal-map variants for the paper tiling benchmark."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from PIL import Image

from experiments.common import (
    image_filename,
    load_prompts,
    prompt_slice,
    write_merged_result_manifest,
)
from experiments.tiling.config import (
    DEFAULT_HIGHPASS_FREQ,
    DEFAULT_SEED,
    OUTPUT_DIR,
    PROMPTS_PATH,
    SOURCE_BASELINE,
    SOURCE_RUN_ID,
    TARGET_SIZE,
    TEXT2TEXTURE_OUTPUT_DIR,
    TILING_BASELINES,
)
from tactilegen.config import Config
from tactilegen.generation.models import ModelManager
from tactilegen.generation.tileable_patch_generation import (
    InterTilePatchGenerator,
    IntraTilePatchGenerator,
    TiledDiffusion,
)
from tactilegen.generation.utils import center_crop, tile_image
from tactilegen.geometry.filtering import apply_high_pass_to_normal_map


def apply_resize(image: Image.Image) -> Image.Image:
    return image.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)


def apply_center_crop(image: Image.Image) -> Image.Image:
    return center_crop(image, TARGET_SIZE)


def apply_highpass(
    image: Image.Image, freq_threshold: int = DEFAULT_HIGHPASS_FREQ
) -> Image.Image:
    return apply_high_pass_to_normal_map(
        image, freq_threshold=freq_threshold, method="per_channel"
    )


def create_tiled_preview(image: Image.Image, n: int = 3) -> Image.Image:
    return tile_image(image, rows=n, cols=n)


def source_normal_path(
    source_dir: Path, prompt_idx: int, prompt: str, seed: int
) -> Path:
    return source_dir / image_filename(prompt_idx, prompt, seed, suffix="_normal")


def get_or_create_highpass(
    run_dir: Path, source_dir: Path, prompt_idx: int, prompt: str, seed: int
) -> Image.Image | None:
    highpass_dir = run_dir / "highpass"
    highpass_dir.mkdir(parents=True, exist_ok=True)
    out_path = highpass_dir / image_filename(prompt_idx, prompt, seed)
    if out_path.exists():
        return Image.open(out_path).convert("RGB")
    normal_path = source_normal_path(source_dir, prompt_idx, prompt, seed)
    if not normal_path.exists():
        return None
    normal = Image.open(normal_path).convert("RGB")
    result = apply_highpass(apply_center_crop(normal))
    result.save(out_path)
    create_tiled_preview(result).save(
        highpass_dir / image_filename(prompt_idx, prompt, seed, suffix="_tiled")
    )
    return result


def build_generators(selected_baselines: list[str]):
    mm = ModelManager(config=Config())
    generators = {}
    for baseline in selected_baselines:
        method = TILING_BASELINES[baseline]["method"]
        if (
            method == "tiled_diffusion_sdxl"
            and "tiled_diffusion_sdxl" not in generators
        ):
            generators["tiled_diffusion_sdxl"] = TiledDiffusion(model_manager=mm)
        elif method == "ours_inter" and "ours_inter" not in generators:
            generators["ours_inter"] = InterTilePatchGenerator(model_manager=mm)
        elif method == "ours_intra" and "ours_intra" not in generators:
            generators["ours_intra"] = IntraTilePatchGenerator(model_manager=mm)
    return generators, mm


def generate_tiled(
    run_id: str,
    t2t_run_id: str = SOURCE_RUN_ID,
    source_baseline: str = SOURCE_BASELINE,
    baselines: list[str] | None = None,
    prompt_range: tuple[int, int] | None = None,
    seed: int = DEFAULT_SEED,
) -> None:
    selected_baselines = baselines or list(TILING_BASELINES)

    prompts, _ = load_prompts(PROMPTS_PATH)
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_dir = TEXT2TEXTURE_OUTPUT_DIR / t2t_run_id / source_baseline
    if not source_dir.exists():
        raise FileNotFoundError(f"Source normal-map directory not found: {source_dir}")

    generators = {}
    mm = None
    if any(TILING_BASELINES[name]["requires_gpu"] for name in selected_baselines):
        generators, mm = build_generators(selected_baselines)

    total = len(selected_baselines) * len(prompt_slice(prompts, prompt_range))
    count = 0
    generated = 0
    skipped = 0
    errors = 0
    records: list[dict] = []

    print(f"Run: {run_id}")
    print(f"Source: {source_dir}")
    print(f"Baselines: {selected_baselines}")

    for baseline_name in selected_baselines:
        baseline = TILING_BASELINES[baseline_name]
        out_dir = run_dir / baseline_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for prompt_idx in prompt_slice(prompts, prompt_range):
            prompt = prompts[prompt_idx]
            count += 1
            out_path = out_dir / image_filename(prompt_idx, prompt, seed)
            preview_path = out_dir / image_filename(
                prompt_idx, prompt, seed, suffix="_tiled"
            )
            record = {
                "baseline": baseline_name,
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "seed": seed,
                "image_path": str(out_path.relative_to(run_dir)),
                "preview_path": str(preview_path.relative_to(run_dir)),
            }
            if out_path.exists() and preview_path.exists():
                skipped += 1
                records.append({**record, "status": "skipped"})
                print(f"[{count}/{total}] SKIP {out_path.relative_to(run_dir)}")
                continue

            print(f"[{count}/{total}] GEN {out_path.relative_to(run_dir)}")
            try:
                method = baseline["method"]
                params = baseline.get("params", {})
                if method == "resize":
                    source = Image.open(
                        source_normal_path(source_dir, prompt_idx, prompt, seed)
                    ).convert("RGB")
                    result = apply_resize(source)
                elif method == "center_crop":
                    source = Image.open(
                        source_normal_path(source_dir, prompt_idx, prompt, seed)
                    ).convert("RGB")
                    result = apply_center_crop(source)
                elif method == "highpass":
                    source = Image.open(
                        source_normal_path(source_dir, prompt_idx, prompt, seed)
                    ).convert("RGB")
                    result = apply_highpass(apply_center_crop(source), **params)
                else:
                    highpass = get_or_create_highpass(
                        run_dir, source_dir, prompt_idx, prompt, seed
                    )
                    if highpass is None:
                        raise FileNotFoundError(
                            source_normal_path(source_dir, prompt_idx, prompt, seed)
                        )
                    result = generators[method].make_tileable(
                        highpass, seed=seed, **params
                    )

                result.save(out_path)
                create_tiled_preview(result).save(preview_path)
                generated += 1
                records.append({**record, "status": "success"})
            except Exception as exc:
                print(f"  ERROR: {exc}")
                errors += 1
                records.append({**record, "status": "error", "error": str(exc)})

        if baseline["requires_gpu"] and mm is not None:
            mm.unload_all_models()
            gc.collect()

    if mm is not None:
        mm.unload_all_models()

    write_merged_result_manifest(
        run_dir / "generation_results.json",
        {
            "run_id": run_id,
            "t2t_run_id": t2t_run_id,
            "source_baseline": source_baseline,
            "baselines": selected_baselines,
            "seed": seed,
            "target_size": TARGET_SIZE,
            "stats": {
                "total": count,
                "generated": generated,
                "skipped": skipped,
                "errors": errors,
            },
            "results": records,
        },
        key_fields=("baseline", "prompt_idx", "seed"),
    )
    print(
        f"Done: generated={generated}, skipped={skipped}, errors={errors}, total={count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--t2t-run-id", default=SOURCE_RUN_ID)
    parser.add_argument("--source-baseline", default=SOURCE_BASELINE)
    parser.add_argument(
        "--baselines", nargs="+", choices=list(TILING_BASELINES), default=None
    )
    parser.add_argument(
        "--prompts", nargs=2, type=int, metavar=("START", "END"), default=None
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    generate_tiled(
        run_id=args.run_id,
        t2t_run_id=args.t2t_run_id,
        source_baseline=args.source_baseline,
        baselines=args.baselines,
        prompt_range=tuple(args.prompts) if args.prompts else None,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
