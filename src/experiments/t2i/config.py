"""Configuration for the paper T2I benchmark."""

from __future__ import annotations

from pathlib import Path

from tactilegen.config import (
    DEFAULT_PLATE_IMAGE,
)
from tactilegen.config import (
    STYLE_PROMPT_PREFIX as _STYLE_PROMPT_PREFIX,
)
from tactilegen.config import (
    STYLE_PROMPT_SUFFIX as _STYLE_PROMPT_SUFFIX,
)

STYLE_PROMPT_PREFIX = _STYLE_PROMPT_PREFIX
STYLE_PROMPT_SUFFIX = _STYLE_PROMPT_SUFFIX

T2I_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = T2I_DIR / "output"
PROMPTS_PATH = T2I_DIR / "prompts.json"

STYLE_PROMPT_SUFFIX_NO_PLATE = (
    "Depicted as a detailed relief sculpture carved from white marble, "
    "with well-defined textures and crisp, well-defined edges, "
    "in a monochrome white-to-gray color scheme. "
    "The relief is gently but clearly convex, rising higher above the surface "
    "to create readable volume and form, "
    "while remaining a restrained relief rather than a fully detached sculpture. "
    "It is never engraved inward, recessed, or cut into the stone. "
    "Simple, centered composition with top-down lighting that emphasizes surface depth and height. "
    "Viewed from directly above."
)

T2I_BASELINES = {
    "qwen_edit_4step": {
        "model": "qwen_edit",
        "steps": 4,
        "display_name": "QIE-4",
        "description": "Qwen-Image-Edit-2511 with 4-step edit LoRA",
    },
    "qwen_edit_40step": {
        "model": "qwen_edit",
        "steps": 40,
        "display_name": "QIE-40",
        "description": "Qwen-Image-Edit-2511 full 40-step inference",
    },
    "nano_banana_pro": {
        "model": "nano_banana_pro",
        "steps": None,
        "display_name": "NBP",
        "description": "Gemini 3 Pro image generation",
    },
}

PLATE_CONDITIONS = {
    "without_plate": {
        "display_name": "No",
        "plate_image": None,
        "suffix": STYLE_PROMPT_SUFFIX_NO_PLATE,
    },
    "with_plate": {
        "display_name": "Yes",
        "plate_image": DEFAULT_PLATE_IMAGE,
        "suffix": STYLE_PROMPT_SUFFIX,
    },
}

PLATE_SEGMENTATION_PROMPT = "a white marble plate background"
PLATE_SEGMENTATION_CONFIDENCE = 0.3
