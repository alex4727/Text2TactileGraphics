"""Lazy model loading with per-model GPU assignment."""

import gc
import logging
import os
from functools import cache, cached_property
from typing import Any

from tactilegen.config import DEFAULT_CKPT_PATHS, Config, global_config
from tactilegen.secrets_ import get_gemini_api_key

logger = logging.getLogger(__name__)


def _vram_limit_gb(device: str) -> float:
    """Total VRAM on `device`, minus 0.5 GB headroom."""
    import torch

    return torch.cuda.mem_get_info(device)[1] / (1024**3) - 0.5


def _reclaim_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ModelManager:
    """Lazily loads and caches all models used by the pipeline."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or global_config()

    # =========================================================================
    # Qwen
    # =========================================================================

    @cached_property
    def qwen_base_edit(self) -> dict[str, Any]:
        """Qwen-Image-Edit-2511 pipeline for base relief image generation."""
        return self._load_qwen_pipeline(
            role="qwen_base_edit",
            dit_model_id="Qwen/Qwen-Image-Edit-2511",
            processor_model_id="Qwen/Qwen-Image-Edit",
        )

    @cached_property
    def qwen_texture(self) -> dict[str, Any]:
        """Qwen-Image T2I pipeline used for texture generation.

        This loads the base weights only. The texture LoRA is loaded later by `TextureGenerator.`
        """
        return self._load_qwen_pipeline(
            role="qwen_texture",
            dit_model_id="Qwen/Qwen-Image",
            tokenizer_model_id="Qwen/Qwen-Image",
        )

    @cached_property
    def qwen_tiling(self) -> dict[str, Any]:
        """Qwen-Image pipeline used for seamless-tiling inpainting."""
        return self._load_qwen_pipeline(
            role="tile_generator",
            dit_model_id="Qwen/Qwen-Image",
            tokenizer_model_id="Qwen/Qwen-Image",
        )

    def _load_qwen_pipeline(
        self,
        *,
        role: str,
        dit_model_id: str,
        base_model_id: str = "Qwen/Qwen-Image",
        tokenizer_model_id: str | None = None,
        processor_model_id: str | None = None,
    ) -> dict:
        """Build a `QwenImagePipeline` for a given Qwen role.

        Exactly one of `tokenizer_model_id` / `processor_model_id` must be set;
        Qwen-Image uses a tokenizer, Qwen-Image-Edit uses a processor.
        """
        if (tokenizer_model_id is None) == (processor_model_id is None):
            raise ValueError(
                "Specify exactly one of tokenizer_model_id / processor_model_id"
            )

        import torch
        from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

        device = self.config.get_device(role)
        logger.info("Loading Qwen pipeline (%s) on %s...", role, device)

        vram_config = self.config.get_qwen_vram_config(device)

        kwargs: dict = {
            "torch_dtype": torch.bfloat16,
            "device": device,
            "model_configs": self._qwen_core_configs(
                dit_model_id, base_model_id, vram_config
            ),
            "vram_limit": _vram_limit_gb(device),
        }
        if tokenizer_model_id:
            kwargs["tokenizer_config"] = ModelConfig(
                model_id=tokenizer_model_id, origin_file_pattern="tokenizer/"
            )
        if processor_model_id:
            kwargs["processor_config"] = ModelConfig(
                model_id=processor_model_id, origin_file_pattern="processor/"
            )

        os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
        return {
            "pipeline": QwenImagePipeline.from_pretrained(**kwargs),
            "device": device,
        }

    @staticmethod
    def _qwen_core_configs(
        dit_model_id: str, base_model_id: str, vram_config: dict
    ) -> list[Any]:
        """Build the three standard Qwen ModelConfigs (transformer + text encoder + VAE).

        `dit_model_id` is the model the transformer weights come from (e.g.
        "Qwen/Qwen-Image-Edit-2511" for the edit pipeline); the text encoder
        and VAE are always pulled from `base_model_id`.
        """
        from diffsynth.pipelines.qwen_image import ModelConfig

        return [
            ModelConfig(
                model_id=dit_model_id,
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id=base_model_id,
                origin_file_pattern="text_encoder/model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id=base_model_id,
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **vram_config,
            ),
        ]

    # =========================================================================
    # SDXL
    # =========================================================================

    @cached_property
    def sdxl_tiling(self) -> dict[str, Any]:
        """SDXL base + refiner pipelines for tileable texture generation."""
        import torch
        from diffusers.models import AutoencoderKL

        from tactilegen.generation.tiled_diffusion.tiled_diffusion_pipeline import (
            StableDiffusionXLDiffImg2ImgPipeline,
        )

        device = self.config.get_device("tile_generator")
        vae_fix = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )

        logger.info("Loading SDXL base model on %s...", device)
        base = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae_fix,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(device)

        logger.info("Loading SDXL refiner model on %s...", device)
        refiner = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(device)

        return {"base": base, "refiner": refiner, "vae_fix": vae_fix, "device": device}

    # =========================================================================
    # GenAI (Gemini)
    # =========================================================================

    @cached_property
    def genai(self) -> dict[str, Any]:
        """Google GenAI client for Gemini-based image generation."""
        from google import genai

        return {"client": genai.Client(api_key=get_gemini_api_key())}

    # =========================================================================
    # SAM3
    # =========================================================================

    @cached_property
    def sam3_text(self) -> dict:
        """SAM3 text model for text-prompt segmentation."""
        from transformers import Sam3Model, Sam3Processor

        return self._load_sam3(Sam3Model, Sam3Processor, label="SAM3 Text")

    @cached_property
    def sam3_tracker(self) -> dict:
        """SAM3 tracker model for click-based segmentation."""
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        return self._load_sam3(
            Sam3TrackerModel, Sam3TrackerProcessor, label="SAM3 Tracker"
        )

    def _load_sam3(
        self, model_cls: Any, processor_cls: Any, *, label: str
    ) -> dict[str, Any]:
        """Load a SAM3 model + matching processor on the SAM3 device."""
        device = self.config.get_device("sam3")
        logger.info("Loading %s on %s...", label, device)
        return {
            "model": model_cls.from_pretrained("facebook/sam3").to(device),
            "processor": processor_cls.from_pretrained("facebook/sam3"),
            "device": device,
        }

    # =========================================================================
    # MoGe
    # =========================================================================

    @cached_property
    def moge2(self) -> dict[str, Any]:
        """MoGe v2 depth/normal estimation model."""
        from moge.model.v2 import MoGeModel

        device = self.config.get_device("moge2")
        logger.info("Loading MoGe v2 on %s...", device)
        model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
        return {"model": model, "device": device}

    # =========================================================================
    # Cleanup
    # =========================================================================

    @classmethod
    def _model_names(cls) -> set[str]:
        """Names of every model attribute on this class.

        **This assumes the models are exactly those attributes defined using** ``cached_property``.
        """
        return {
            name
            for name, value in vars(cls).items()
            if isinstance(value, cached_property)
        }

    def unload_model(self, name: str) -> None:
        """Drop the cached bundle for ``name`` and free its GPU memory.

        If ``name`` was never loaded or has already been unloaded, this is a no-op.
        """
        valid = self._model_names()
        if name not in valid:
            raise ValueError(
                f"{name!r} is not a recognized model. Valid names: {sorted(valid)}"
            )
        if name not in self.__dict__:
            logger.info("unload_model(%r): not currently loaded — no-op", name)
            return
        logger.info("Unloading %s", name)
        del self.__dict__[name]
        _reclaim_memory()

    def unload_all_models(self) -> None:
        """Unload every currently-loaded model and free GPU memory.

        Cheaper than calling ``unload_model`` for each name because
        we only reclaim memory once.
        """
        loaded = self._model_names() & set(self.__dict__)
        if not loaded:
            return
        for name in loaded:
            logger.info("Unloading %s", name)
            del self.__dict__[name]
        _reclaim_memory()


# =============================================================================
# Module-level singleton
# =============================================================================


@cache
def global_model_manager() -> ModelManager:
    return ModelManager()


# =============================================================================
# LoRA management
# =============================================================================


class LoraManager:
    """Track and swap LoRA adapters on a single diffsynth-style pipeline."""

    def __init__(self, pipeline: Any, config: Config | None = None) -> None:
        self.pipeline = pipeline
        self._config = config or global_config()
        self._loaded_paths: tuple[str, ...] = ()

    @property
    def loaded_paths(self) -> tuple[str, ...]:
        """LoRA paths currently loaded into `self.pipeline`."""
        return self._loaded_paths

    def resolve(self, key: str | None) -> str | None:
        """Resolve a checkpoint key to an existing path.

        Returns `None` if `key` is falsy (no checkpoint requested).
        """
        if not key:
            return None

        ckpt_path = self._config.ckpt_paths.get(key) or DEFAULT_CKPT_PATHS.get(key)
        if ckpt_path is None:
            available = sorted(set(self._config.ckpt_paths) | set(DEFAULT_CKPT_PATHS))
            raise KeyError(f"Unknown checkpoint key {key!r}. Known keys: {available}")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {key!r} not found at {ckpt_path!r}")
        return ckpt_path

    def apply(self, desired_keys: list[str | None]) -> None:
        """
        Load the LoRAs identified by `desired_keys` into `self.pipeline`,
        if they differ from what's currently loaded.

        Each entry is a checkpoint key or None.
        The ordering of `desired_keys` matters — it's preserved when reloading.
        """
        desired_paths = tuple(p for p in (self.resolve(k) for k in desired_keys) if p)
        if desired_paths == self._loaded_paths:
            return
        if self._loaded_paths:
            self.pipeline.clear_lora()
            logger.info("Cleared LoRAs: %s", list(self._loaded_paths))
        for path in desired_paths:
            self.pipeline.load_lora(self.pipeline.dit, path)
            logger.info("Loaded LoRA: %s", path)
        self._loaded_paths = desired_paths
