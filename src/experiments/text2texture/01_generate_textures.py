"""Step 1: Generate texture images for the paper text-to-texture benchmark."""

from __future__ import annotations

import argparse

from google.genai import types
from PIL import Image

from experiments.common import (
    image_filename,
    load_prompts,
    prompt_slice,
    write_merged_result_manifest,
)
from experiments.text2texture.config import (
    OUTPUT_DIR,
    PROMPTS_PATH,
    TEXT2TEXTURE_BASELINES,
    TEXTURE_PROMPT_TEMPLATE,
)
from text2tactilegraphics.config import Config
from text2tactilegraphics.generation.base_image_generation import (
    GEMINI_MODELS,
    _extract_gemini_image,
)
from text2tactilegraphics.generation.models import LoraManager, ModelManager


class BenchmarkTextureGenerator:
    """Qwen texture generator with explicit LoRA control for ablations."""

    def __init__(self, model_manager, config) -> None:
        self.mm = model_manager
        self.config = config
        self.pipeline = self.mm.qwen_texture["pipeline"]
        self.device = self.mm.qwen_texture["device"]
        self.loras = LoraManager(pipeline=self.pipeline, config=self.config)

    def generate(
        self,
        prompt: str,
        *,
        texture_lora: str | None,
        steps: int,
        seed: int,
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        step_config = self.config.get_texture_config(steps)
        self.loras.apply([texture_lora, step_config.get("distill_lora")])
        return self.pipeline(
            prompt=TEXTURE_PROMPT_TEMPLATE.format(prompt.strip()),
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=steps,
            cfg_scale=step_config["cfg_scale"],
            rand_device=self.device,
            progress_bar_cmd=lambda x: x,
        )


def generate_gemini_texture(mm, prompt: str, model: str) -> Image.Image:
    full_prompt = TEXTURE_PROMPT_TEMPLATE.format(prompt.strip())
    kwargs = {"model": GEMINI_MODELS[model], "contents": [full_prompt]}
    if model == "nano_banana_pro":
        kwargs["config"] = types.GenerateContentConfig(
            image_config=types.ImageConfig(aspect_ratio="1:1", image_size="1K")
        )
    return _extract_gemini_image(
        mm.genai["client"].models.generate_content(**kwargs), model
    ).convert("RGB")


def generate_textures(
    run_id: str,
    baselines: list[str] | None = None,
    prompt_range: tuple[int, int] | None = None,
    seeds: list[int] | None = None,
) -> None:
    selected_baselines = baselines or list(TEXT2TEXTURE_BASELINES)

    prompts, default_seeds = load_prompts(PROMPTS_PATH)
    seeds = seeds or default_seeds
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    config.texture_prompt_template = TEXTURE_PROMPT_TEMPLATE
    mm = ModelManager(config=config)
    qwen_generator: BenchmarkTextureGenerator | None = None
    results: list[dict] = []

    total = (
        len(selected_baselines) * len(prompt_slice(prompts, prompt_range)) * len(seeds)
    )
    count = 0
    skipped = 0
    generated = 0

    print(f"Run: {run_id}")
    print(f"Baselines: {selected_baselines}")
    print(f"Seeds: {seeds}")

    for baseline_name in selected_baselines:
        baseline = TEXT2TEXTURE_BASELINES[baseline_name]
        out_dir = run_dir / baseline_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for prompt_idx in prompt_slice(prompts, prompt_range):
            prompt = prompts[prompt_idx]
            for seed in seeds:
                count += 1
                out_path = out_dir / image_filename(prompt_idx, prompt, seed)
                record = {
                    "baseline": baseline_name,
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
                    if baseline["model"] == "qwen":
                        if qwen_generator is None:
                            qwen_generator = BenchmarkTextureGenerator(mm, config)
                        image = qwen_generator.generate(
                            prompt,
                            texture_lora=baseline["texture_lora"],
                            steps=baseline["steps"],
                            seed=seed,
                        )
                    else:
                        image = generate_gemini_texture(mm, prompt, baseline["model"])
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
        key_fields=("baseline", "prompt_idx", "seed"),
    )
    print(f"Done: generated={generated}, skipped={skipped}, total={count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--baselines", nargs="+", choices=list(TEXT2TEXTURE_BASELINES), default=None
    )
    parser.add_argument(
        "--prompts", nargs=2, type=int, metavar=("START", "END"), default=None
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    generate_textures(
        run_id=args.run_id,
        baselines=args.baselines,
        prompt_range=tuple(args.prompts) if args.prompts else None,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
