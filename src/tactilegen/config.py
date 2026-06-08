"""Global configuration for TactileGen."""

import atexit
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

# =============================================================================
# Filesystem layout
# =============================================================================

CKPT_DIR = os.environ.get("TACTILEGEN_CKPT_DIR") or str(
    Path.home() / ".cache" / "tactilegen" / "ckpt"
)

DEFAULT_CKPT_PATHS: dict[str, str] = {
    # Distill LoRAs for different step counts
    "distill_4step": os.path.join(CKPT_DIR, "lightning-4step-v2.0-noalpha.safetensors"),
    # Texture LoRA trained for tactile textures
    "texture_lora": os.path.join(CKPT_DIR, "texture-step3000.safetensors"),
    # Qwen-Image-Edit-2511 4-step LoRA for fast base image generation
    "edit_4step_lora": os.path.join(
        CKPT_DIR, "edit-2511-4steps-v1.0-noalpha.safetensors"
    ),
}


# =============================================================================
# GPU assignments
# =============================================================================

# Logical "slots" (0/1/2) — three groups of co-located models. At Config
# construction the slots are remapped to real CUDA device IDs by free memory,
# so role-pairs that share a slot still share a real GPU.
GPU_ASSIGNMENTS: dict[str, int] = {
    "qwen_base_edit": 0,
    "moge2": 0,
    "sam3": 0,
    "qwen_texture": 1,
    "tile_generator": 2,
}


def get_total_gpus() -> int:
    """Number of visible CUDA devices (0 when CUDA is unavailable)."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def debug_enabled() -> bool:
    """True iff the UI should show the Debug settings panel."""
    raw = os.environ.get("TACTILEGEN_DEBUG", "").strip().lower()
    return raw not in ("", "0", "false", "no")


def _query_free_gpus() -> list[tuple[int, float]]:
    """Return [(gpu_id, free_gb), …] sorted by free memory descending."""
    if not torch.cuda.is_available():
        return []
    gpus = [
        (i, torch.cuda.mem_get_info(i)[0] / (1024**3))
        for i in range(torch.cuda.device_count())
    ]
    gpus.sort(key=lambda x: x[1], reverse=True)
    return gpus


def _build_gpu_assignments(default: dict[str, int]) -> dict[str, int]:
    """Remap logical-slot assignments to real GPUs picked by free memory.

    Preserves the structure of `default` — roles sharing a slot still share
    a real GPU. If fewer real GPUs are available than logical slots, slots
    cycle through the available GPUs.
    """
    free = _query_free_gpus()
    if not free:
        return dict(default)

    # Ordered dedup of logical slot ids.
    slots = list(dict.fromkeys(default.values()))
    available = [gid for gid, _ in free]
    slot_to_real = {slot: available[i % len(available)] for i, slot in enumerate(slots)}
    return {role: slot_to_real[slot] for role, slot in default.items()}


# Shared `Literal` aliases used across the codebase.
VRAMMode = Literal["48gb", "80gb"]
GeometryType = Literal["normal", "depth"]


def _detect_vram_mode(gpu_id: int = 0) -> VRAMMode:
    """Pick '80gb' vs '48gb' VRAM config based on the device's total memory."""
    if not torch.cuda.is_available():
        return "80gb"
    try:
        _free, total = torch.cuda.mem_get_info(gpu_id)
    except RuntimeError:
        return "80gb"
    return "80gb" if total / (1024**3) >= 64 else "48gb"


# =============================================================================
# VRAM-mode configurations for the Qwen texture pipeline
# =============================================================================

# Values of "cuda" are placeholders that `Config.get_qwen_vram_config`
# replaces with the caller's actual device string (e.g. "cuda:1").
QWEN_VRAM_CONFIGS: dict[str, dict] = {
    "80gb": {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cuda",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    },
    "48gb": {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    },
}

_VRAM_DEVICE_KEYS = (
    "offload_device",
    "onload_device",
    "preparing_device",
    "computation_device",
)


# =============================================================================
# Inference step presets
# =============================================================================

# Maps step count → preset for the Qwen texture generation pipeline.
INFERENCE_STEPS: dict[int, dict] = {
    4: {"distill_lora": "distill_4step", "cfg_scale": 1.0},
    40: {"distill_lora": None, "cfg_scale": 4.0},  # full quality, no distill
}

# Maps step count → preset for Qwen-Image-Edit base image generation.
BASE_IMAGE_STEPS: dict[int, dict] = {
    4: {"edit_lora": "edit_4step_lora", "cfg_scale": 1.0},  # fast w/ LoRA
    40: {"edit_lora": None, "cfg_scale": 4.0},  # full quality
}


def _lookup_steps(table: dict[int, dict], steps: int) -> dict:
    """Return a shallow copy of `table[steps]` with a friendly error on miss."""
    try:
        return table[steps].copy()
    except KeyError:
        raise ValueError(f"Unsupported step count: {steps}. Supported: {sorted(table)}")


# =============================================================================
# Prompt templates
# =============================================================================

STYLE_PROMPT_PREFIX = "A white marble stone relief sculpture of:"
STYLE_PROMPT_SUFFIX = (
    "Depicted as a detailed relief sculpture carved from white marble, "
    "with well-defined textures and crisp, well-defined edges, "
    "in a monochrome white-to-gray color scheme. "
    "The sculpture is carved and formed directly on top of the provided marble plate, "
    "using the plate as the base material. "
    "The relief is gently but clearly convex, rising higher above the plate surface "
    "to create readable volume and form, "
    "while remaining a restrained relief rather than a fully detached sculpture. "
    "It is never engraved inward, recessed, or cut into the stone. "
    "Simple, centered composition with top-down lighting that emphasizes surface depth and height. "
    "Viewed from directly above, fully confined within the marble plate."
)

TEXTURE_PROMPT_TEMPLATE = (
    "A texture of {}, extreme close-up, plain lighting, straight-on view"
)

TILING_PROMPT_TEMPLATE_TILED_DIFFUSION = "normal map, highly detailed"
TILING_NEGATIVE_PROMPT_TILED_DIFFUSION = "blurry, low quality, distorted"
TILING_PROMPT_INPAINTING = (
    "a highly detailed surface normal map (normal map color scheme) "
    "smooth, continuous, repetitive patterns"
)
TILING_NEGATIVE_PROMPT_INPAINTING = "seams, visible seams, tiled, nxn tiles"


# =============================================================================
# Packaged assets
# =============================================================================
# `importlib.resources.as_file` resolves each entry to a real filesystem path
# (extracting from a zip-installed wheel into a temp dir if needed). The
# ExitStack keeps any temp extractions alive for the lifetime of the process.
_asset_stack = ExitStack()
atexit.register(_asset_stack.close)


def _materialize_asset(name: str) -> Path:
    """Return a real filesystem path for the asset `name` under `assets/`."""
    resource = resources.files("tactilegen.assets") / name
    return Path(_asset_stack.enter_context(resources.as_file(resource)))


DEFAULT_PLATE_IMAGE: str = str(_materialize_asset("white_marble_plate.png"))
DEFAULT_TILED_DIFFUSION_MASK: str = str(_materialize_asset("mask_xy.png"))
SENSOR_TILED_DIR: Path = _materialize_asset("tiled_from_sensor")


# =============================================================================
# Config dataclass
# =============================================================================


@dataclass
class Config:
    """Global pipeline configuration."""

    # VRAM mode — auto-detected from device 0's total memory at construction.
    vram_mode: VRAMMode = field(default_factory=_detect_vram_mode)

    # Whether to use normal or depth estimation
    geometry_type: GeometryType = "normal"

    # Style prompts for base image generation
    base_image_style_prefix: str = STYLE_PROMPT_PREFIX
    base_image_style_suffix: str = STYLE_PROMPT_SUFFIX

    # User prompt is wrapped in this template before being fed to the texture generator.
    # Must contain exactly one `{}` placeholder.
    texture_prompt_template: str = TEXTURE_PROMPT_TEMPLATE

    # Prompts for the tiled-diffusion patch generators
    tiling_prompt_tiled_diffusion: str = TILING_PROMPT_TEMPLATE_TILED_DIFFUSION
    tiling_negative_prompt_tiled_diffusion: str = TILING_NEGATIVE_PROMPT_TILED_DIFFUSION

    # Prompts for the inpainting-based tiling generators
    tiling_prompt_inpainting: str = TILING_PROMPT_INPAINTING
    tiling_negative_prompt_inpainting: str = TILING_NEGATIVE_PROMPT_INPAINTING

    # Baseplate image conditioning the base image generation. `None` runs pure text-to-image.
    plate_image: str | Path | Image.Image | None = DEFAULT_PLATE_IMAGE

    # Paths to local model weights
    ckpt_paths: dict[str, str] = field(
        default_factory=lambda: DEFAULT_CKPT_PATHS.copy()
    )

    # Logical GPU slots for various steps of the pipeline
    gpu_assignments: dict[str, int] = field(
        default_factory=lambda: _build_gpu_assignments(GPU_ASSIGNMENTS)
    )

    def get_device(self, model_name: str) -> str:
        """Return the device string ('cuda:N' or 'cpu') for `model_name`."""
        gpu_id = self.gpu_assignments.get(model_name, 0)
        return f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    def get_qwen_vram_config(self, device: str) -> dict:
        """Return the Qwen VRAM config with 'cuda' placeholders bound to `device`."""
        cfg = QWEN_VRAM_CONFIGS[self.vram_mode].copy()
        for key in _VRAM_DEVICE_KEYS:
            if cfg[key] == "cuda":
                cfg[key] = device
        return cfg

    @staticmethod
    def get_base_image_config(steps: int) -> dict:
        """Return the base image generation preset for `steps`."""
        return _lookup_steps(BASE_IMAGE_STEPS, steps)

    @staticmethod
    def get_texture_config(steps: int) -> dict:
        """Return the Qwen texture inference preset for `steps`."""
        return _lookup_steps(INFERENCE_STEPS, steps)


# =============================================================================
# Module-level singleton
# =============================================================================


@cache
def global_config() -> Config:
    return Config()
