"""Configuration for the paper text-to-texture benchmark."""

from __future__ import annotations

from pathlib import Path

from tactilegen.config import TEXTURE_PROMPT_TEMPLATE as _TEXTURE_PROMPT_TEMPLATE

TEXTURE_PROMPT_TEMPLATE = _TEXTURE_PROMPT_TEMPLATE

TEXT2TEXTURE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TEXT2TEXTURE_DIR / "output"
PROMPTS_PATH = TEXT2TEXTURE_DIR / "prompts.json"

TEXT2TEXTURE_BASELINES = {
    "qwen_no_lora_4step": {
        "model": "qwen",
        "steps": 4,
        "texture_lora": None,
        "display_name": "QI-4",
        "description": "Qwen-Image without texture LoRA, 4-step distilled",
    },
    "qwen_no_lora_40step": {
        "model": "qwen",
        "steps": 40,
        "texture_lora": None,
        "display_name": "QI-40",
        "description": "Qwen-Image without texture LoRA, 40-step full inference",
    },
    "qwen_stage2_4step": {
        "model": "qwen",
        "steps": 4,
        "texture_lora": "texture_lora",
        "display_name": "Ours-4",
        "description": "TactileGen texture LoRA, 4-step distilled",
    },
    "qwen_stage2_40step": {
        "model": "qwen",
        "steps": 40,
        "texture_lora": "texture_lora",
        "display_name": "Ours-40",
        "description": "TactileGen texture LoRA, 40-step full inference",
    },
    "nano_banana_pro": {
        "model": "nano_banana_pro",
        "steps": None,
        "texture_lora": None,
        "display_name": "NBP",
        "description": "Gemini 3 Pro texture generation",
    },
}

DEFAULT_N_PATCHES = 32
DEFAULT_PATCH_SIZE = 128
DEFAULT_FREQ_THRESHOLD = 120
