"""Configuration for the paper tiling benchmark."""

from __future__ import annotations

from pathlib import Path

TILING_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TILING_DIR / "output"
PROMPTS_PATH = TILING_DIR / "prompts.json"
TEXT2TEXTURE_OUTPUT_DIR = TILING_DIR.parent / "text2texture" / "output"

SOURCE_RUN_ID = "paper_t2t"
SOURCE_BASELINE = "qwen_stage2_4step"
TARGET_SIZE = 512
DEFAULT_SEED = 42
DEFAULT_HIGHPASS_FREQ = 120

TILING_BASELINES = {
    "resize": {
        "stage": 1,
        "display_name": "Resize",
        "description": "Plain resize from 1024 px to 512 px",
        "method": "resize",
        "requires_gpu": False,
    },
    "center_crop": {
        "stage": 2,
        "display_name": "Crop",
        "description": "Center crop 512 px from the 1024 px normal map",
        "method": "center_crop",
        "requires_gpu": False,
    },
    "highpass": {
        "stage": 3,
        "display_name": "+HPF",
        "description": f"Center crop plus per-channel Gaussian high-pass filter at freq_threshold={DEFAULT_HIGHPASS_FREQ}",
        "method": "highpass",
        "requires_gpu": False,
        "params": {"freq_threshold": DEFAULT_HIGHPASS_FREQ},
    },
    "tiled_diffusion_sdxl": {
        "stage": 4,
        "display_name": "++TD",
        "description": "Highpass plus SDXL tiled diffusion",
        "method": "tiled_diffusion_sdxl",
        "requires_gpu": True,
        "params": {
            "use_soft_mask": False,
            "use_periodic_projection": False,
        },
    },
    "ours_inter": {
        "stage": 4,
        "display_name": "++Ours-I",
        "description": "Highpass plus inter-tile Qwen inpainting",
        "method": "ours_inter",
        "requires_gpu": True,
        "params": {
            "n_tiles": 3,
            "mask_portion": 0.05,
            "denoising_strength": 0.9,
            "num_inference_steps": 10,
        },
    },
    "ours_intra": {
        "stage": 4,
        "display_name": "++Ours-A",
        "description": "Highpass plus intra-tile seam-swap Qwen inpainting",
        "method": "ours_intra",
        "requires_gpu": True,
        "params": {
            "direction": "both",
            "mask_width": 64,
            "denoising_strength": 0.9,
            "num_inference_steps": 10,
        },
    },
}

MAIN_METRICS = [
    ("border_avg", "Border Cont", "lower"),
    ("grad_cont_avg", "Grad Cont", "lower"),
]
