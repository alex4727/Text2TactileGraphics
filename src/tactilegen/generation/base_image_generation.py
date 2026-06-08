import io
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from tactilegen.config import Config, global_config
from tactilegen.generation.models import LoraManager, ModelManager, global_model_manager
from tactilegen.generation.utils import open_rgb_image

# Shared `Literal` aliases for the base-image API.
GeminiModel = Literal["nano_banana_pro", "nano_banana"]
BaseImageModel = Literal["qwen_edit", "nano_banana_pro", "nano_banana"]
BaseImageSteps = Literal[4, 40]


# Model ID mapping for Gemini base image generation.
GEMINI_MODELS = {
    "nano_banana_pro": "gemini-3-pro-image-preview",  # Gemini 3 Pro
    "nano_banana": "gemini-2.5-flash-image",  # Gemini 2.5 Flash
}


def _img_to_genai_part(image: str | Path | Image.Image):
    """Convert image path or PIL Image to a Google GenAI Part."""
    from google.genai import types

    pil = open_rgb_image(image)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    buffer.seek(0)
    return types.Part.from_bytes(data=buffer.read(), mime_type="image/png")


def _extract_gemini_image(response: Any, model: str) -> Image.Image:
    """Pull the first inline image out of a Gemini generate_content response."""
    parts = (
        response.parts
        if model == "nano_banana_pro"
        else response.candidates[0].content.parts
    )
    for part in parts:
        if part.inline_data is not None:
            return Image.open(io.BytesIO(part.inline_data.data))
    raise ValueError(f"No image in {model} response")


class BaseImageGenerator:
    """Generate base relief images."""

    def __init__(
        self, model_manager: ModelManager | None = None, config: Config | None = None
    ) -> None:
        self.mm = model_manager or global_model_manager()
        self.config = config or global_config()

    @property
    def qwen_pipeline(self):
        return self.mm.qwen_base_edit["pipeline"]

    @cached_property
    def loras(self) -> LoraManager:
        return LoraManager(pipeline=self.qwen_pipeline, config=self.config)

    def generate(
        self,
        prompt: str,
        model: BaseImageModel = "qwen_edit",
        steps: BaseImageSteps = 4,
        seed: int = 42,
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        """Generate a base relief image with the chosen `model`.

        Args:
            prompt: Image prompt.
            model: Generation model to use: ``"qwen_edit"``
                (local Qwen-Image-Edit-2511), ``"nano_banana_pro"``,
                (Gemini 3 Pro API) or ``"nano_banana"`` (Gemini 2.5 Flash API).
            steps: Number of denoising steps for ``qwen_edit``. ``4`` loads
                a 4-step distilled edit LoRA; ``40`` runs full-quality
                inference with no distill LoRA (slow). Ignored by Gemini.
            seed: RNG seed for ``qwen_edit`` reproducibility. Ignored by Gemini.
            height: Output height in pixels for ``qwen_edit``. Ignored by Gemini.
            width: Output width in pixels for ``qwen_edit``. Ignored by Gemini.

        Returns:
            The generated image.
        """
        if model == "qwen_edit":
            return self._generate_qwen_edit(prompt, steps, seed, height, width)
        if model in GEMINI_MODELS:
            return self._generate_gemini(prompt, model)
        raise ValueError(f"Unknown model: {model!r}")

    def _generate_qwen_edit(
        self,
        prompt: str,
        steps: BaseImageSteps,
        seed: int,
        height: int,
        width: int,
    ) -> Image.Image:
        self._setup_for_steps(steps)
        step_config = self.config.get_base_image_config(steps)
        plate_image = self.config.plate_image
        edit_images = [open_rgb_image(plate_image)] if plate_image is not None else None
        return self.qwen_pipeline(
            prompt=self._build_prompt(prompt),
            edit_image=edit_images,
            seed=seed,
            num_inference_steps=steps,
            height=height,
            width=width,
            cfg_scale=step_config["cfg_scale"],
            edit_image_auto_resize=True,
            zero_cond_t=True,  # required by Qwen-Image-Edit-2511
        )

    def _setup_for_steps(self, steps: int) -> None:
        """Configure the LoRA for the requested step count."""
        step_config = self.config.get_base_image_config(steps)
        self.loras.apply([step_config.get("edit_lora")])

    def _build_prompt(self, prompt: str) -> str:
        """Wrap `prompt` with the configured style prefix/suffix."""
        return f"{self.config.base_image_style_prefix} {prompt} {self.config.base_image_style_suffix}"

    def _generate_gemini(
        self,
        prompt: str,
        model: GeminiModel,
    ) -> Image.Image:
        from google.genai import types

        client = self.mm.genai["client"]
        model_id = GEMINI_MODELS[model]
        full_prompt = self._build_prompt(prompt)
        plate_image = self.config.plate_image
        contents = (
            [full_prompt]
            if plate_image is None
            else [full_prompt, _img_to_genai_part(plate_image)]
        )

        kwargs: dict = {"model": model_id, "contents": contents}
        if model == "nano_banana_pro":
            kwargs["config"] = types.GenerateContentConfig(
                image_config=types.ImageConfig(aspect_ratio="1:1", image_size="1K")
            )

        response = client.models.generate_content(**kwargs)
        return _extract_gemini_image(response, model)
