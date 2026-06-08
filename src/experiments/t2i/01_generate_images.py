"""Step 1: Generate images for the paper T2I benchmark."""

from __future__ import annotations

import argparse

from experiments.common import (
    image_filename,
    load_prompts,
    prompt_slice,
    write_merged_result_manifest,
)
from experiments.t2i.config import (
    OUTPUT_DIR,
    PLATE_CONDITIONS,
    PROMPTS_PATH,
    STYLE_PROMPT_PREFIX,
    T2I_BASELINES,
)
from tactilegen.config import Config
from tactilegen.generation.base_image_generation import BaseImageGenerator
from tactilegen.generation.models import ModelManager


def generate_images(
    run_id: str,
    baselines: list[str] | None = None,
    prompt_range: tuple[int, int] | None = None,
    seeds: list[int] | None = None,
) -> None:
    selected_baselines = baselines or list(T2I_BASELINES)

    prompts, default_seeds = load_prompts(PROMPTS_PATH)
    seeds = seeds or default_seeds
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    mm = ModelManager(config=config)
    generator = BaseImageGenerator(model_manager=mm, config=config)

    results: list[dict] = []
    total = (
        len(selected_baselines)
        * len(PLATE_CONDITIONS)
        * len(prompt_slice(prompts, prompt_range))
        * len(seeds)
    )
    count = 0
    skipped = 0
    generated = 0

    print(f"Run: {run_id}")
    print(f"Baselines: {selected_baselines}")
    print(f"Seeds: {seeds}")

    for baseline_name in selected_baselines:
        baseline = T2I_BASELINES[baseline_name]
        model = baseline["model"]
        steps = baseline["steps"]

        for plate_condition, plate_config in PLATE_CONDITIONS.items():
            config.base_image_style_prefix = STYLE_PROMPT_PREFIX
            config.base_image_style_suffix = plate_config["suffix"]
            config.plate_image = plate_config["plate_image"]

            out_dir = run_dir / f"{baseline_name}_{plate_condition}"
            out_dir.mkdir(parents=True, exist_ok=True)

            for prompt_idx in prompt_slice(prompts, prompt_range):
                prompt = prompts[prompt_idx]
                for seed in seeds:
                    count += 1
                    out_path = out_dir / image_filename(prompt_idx, prompt, seed)
                    record = {
                        "baseline": baseline_name,
                        "plate_condition": plate_condition,
                        "prompt_idx": prompt_idx,
                        "prompt": prompt,
                        "seed": seed,
                        "image_path": str(out_path.relative_to(run_dir)),
                    }

                    if out_path.exists():
                        skipped += 1
                        results.append({**record, "status": "skipped"})
                        print(f"[{count}/{total}] SKIP {out_path.relative_to(run_dir)}")
                        continue

                    print(f"[{count}/{total}] GEN {out_path.relative_to(run_dir)}")
                    try:
                        if model == "qwen_edit":
                            image = generator.generate(
                                prompt=prompt,
                                model=model,
                                steps=steps,
                                seed=seed,
                                height=1024,
                                width=1024,
                            )
                        else:
                            image = generator.generate(prompt=prompt, model=model)
                        image.save(out_path)
                        generated += 1
                        results.append({**record, "status": "success"})
                    except Exception as exc:
                        print(f"  ERROR: {exc}")
                        results.append({**record, "status": "error", "error": str(exc)})

    write_merged_result_manifest(
        run_dir / "generation_results.json",
        {
            "run_id": run_id,
            "baselines": selected_baselines,
            "seeds": seeds,
            "stats": {"total": count, "skipped": skipped, "generated": generated},
            "results": results,
        },
        key_fields=("baseline", "plate_condition", "prompt_idx", "seed"),
    )
    print(f"Done: generated={generated}, skipped={skipped}, total={count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--baselines", nargs="+", choices=list(T2I_BASELINES), default=None
    )
    parser.add_argument(
        "--prompts", nargs=2, type=int, metavar=("START", "END"), default=None
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    generate_images(
        run_id=args.run_id,
        baselines=args.baselines,
        prompt_range=tuple(args.prompts) if args.prompts else None,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
