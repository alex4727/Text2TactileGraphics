from functools import cached_property
from typing import Literal

import numpy as np
import torch
from PIL import Image

from tactilegen.config import Config, global_config
from tactilegen.generation.models import LoraManager, ModelManager, global_model_manager
from tactilegen.generation.utils import center_crop_array

# Step counts supported by the Qwen texture presets.
TextureSteps = Literal[4, 40]


class TextureGenerator:
    """Generate texture images via fine-tuned Qwen model."""

    def __init__(
        self, model_manager: ModelManager | None = None, config: Config | None = None
    ) -> None:
        self.mm = model_manager or global_model_manager()
        self.config = config or global_config()

    @property
    def qwen_pipeline(self):
        return self.mm.qwen_texture["pipeline"]

    @property
    def qwen_device(self) -> str:
        return self.mm.qwen_texture["device"]

    @cached_property
    def loras(self) -> LoraManager:
        return LoraManager(pipeline=self.qwen_pipeline, config=self.config)

    def generate(
        self,
        prompt: str,
        steps: TextureSteps = 4,
        seed: int = 42,
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        """Generate a texture image from `prompt`."""
        self._setup_for_steps(steps)
        step_config = self.config.get_texture_config(steps)
        return self.qwen_pipeline(
            prompt=self._build_prompt(prompt),
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=steps,
            cfg_scale=step_config["cfg_scale"],
            rand_device=self.qwen_device,
            progress_bar_cmd=lambda x: x,  # Silence progress bar
        )

    def _setup_for_steps(self, steps: int):
        """Configure LoRAs for the requested step count."""
        step_config = self.config.get_texture_config(steps)
        # Texture LoRA is always loaded; distill LoRA depends on step count.
        self.loras.apply(["texture_lora", step_config.get("distill_lora")])

    def _build_prompt(self, prompt: str) -> str:
        """Wrap a short user description in the configured texture template."""
        return self.config.texture_prompt_template.format(prompt.strip())


class GeometryEstimator:
    """Extract surface geometry (normal/depth maps) from images via MoGe v2."""

    def __init__(self, model_manager: ModelManager | None = None) -> None:
        self.mm = model_manager or global_model_manager()

    @property
    def moge_model(self):
        return self.mm.moge2["model"]

    @property
    def moge_device(self) -> str:
        return self.mm.moge2["device"]

    def compute_normal(
        self, image: Image.Image, *, crop: bool = True, crop_size: int = 512
    ) -> np.ndarray:
        """Extract a unit-length normal map using MoGe.
        **The result is converted to OpenGL format (Y-up).**

        MoGe is always run on the full image so it sees the full context;
        when `crop` is true, the result is center-cropped to `crop_size` on the way out.
        """
        result = self._run_moge(image)
        normal = result["normal"].cpu().numpy()
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)
        # Convert to OpenGL format
        normal[..., 1] *= -1
        normal[..., 2] *= -1
        if crop:
            normal = center_crop_array(normal, crop_size)
        return normal

    def compute_depth(
        self,
        image: Image.Image,
        *,
        normalize: bool = True,
        crop: bool = False,
        crop_size: int = 512,
    ) -> np.ndarray:
        """Extract a depth map from MoGe.
        **Note that if normalize=True, the result is inverted for consistency with
        downstream pipelines (displacement mapping).**

        When `crop` is true, the result is center-cropped to `crop_size` on the way out.
        Cropping is applied *before* normalization so the dynamic range maps to the cropped region.
        """
        result = self._run_moge(image)
        valid_mask: np.ndarray = result["mask"].cpu().numpy()
        depth: np.ndarray = result["depth"].cpu().numpy()
        depth = depth.clip(max=depth[valid_mask].max())  # strip INFs
        if crop:
            depth = center_crop_array(depth, crop_size)
        if normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            return 1 - depth  # closer = higher displacement
        return depth

    def _run_moge(self, image: Image.Image) -> dict:
        """Run MoGe v2 on `image`; return the raw result dict (normal, depth, mask, …)."""
        from torchvision.transforms.functional import to_tensor

        img_tensor = to_tensor(image).to(self.moge_device)
        with torch.no_grad():
            return self.moge_model.infer(img_tensor)
