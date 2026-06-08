"""Shared helpers for benchmark reproduction scripts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

EXPERIMENTS_DIR = Path(__file__).resolve().parent


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_prompts(path: Path) -> tuple[list[str], list[int]]:
    data = load_json(path)
    if "prompts" not in data and "source" in data:
        source_path = (path.parent / data["source"]).resolve()
        prompts, source_seeds = load_prompts(source_path)
        return prompts, data.get("seeds", source_seeds)
    return data["prompts"], data.get("seeds", [42])


def prompt_slice(items: list[Any], prompt_range: tuple[int, int] | None) -> range:
    start = prompt_range[0] if prompt_range else 0
    end = prompt_range[1] if prompt_range else len(items)
    return range(start, end)


def safe_prompt_slug(prompt: str, max_len: int = 30) -> str:
    slug = prompt.replace(" ", "_").replace("/", "_")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)
    return slug[:max_len].strip("_") or "prompt"


def image_filename(prompt_idx: int, prompt: str, seed: int, suffix: str = "") -> str:
    return f"{prompt_idx:02d}_{safe_prompt_slug(prompt)}_seed{seed}{suffix}.png"


def mean_std(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "n": len(clean),
    }


def write_merged_result_manifest(
    path: Path,
    payload: dict[str, Any],
    *,
    key_fields: tuple[str, ...],
) -> None:
    """Write a run manifest without discarding records from previous shards."""
    existing = load_json(path) if path.exists() else {}
    merged: dict[tuple[Any, ...], dict] = {}

    for record in existing.get("results", []):
        merged[_record_key(record, key_fields)] = record

    for record in payload.get("results", []):
        key = _record_key(record, key_fields)
        previous = merged.get(key)
        if (
            previous
            and previous.get("status") == "success"
            and record.get("status") == "skipped"
        ):
            continue
        merged[key] = record

    payload["results"] = list(merged.values())
    payload["stats"] = _status_counts(payload["results"])
    for field in ("baselines", "seeds"):
        payload[field] = _merge_ordered_lists(existing.get(field), payload.get(field))
    write_json(path, payload)


def _record_key(record: dict, key_fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in key_fields)


def _status_counts(records: list[dict]) -> dict[str, int]:
    counts = {"total": len(records), "success": 0, "skipped": 0, "error": 0}
    for record in records:
        status = record.get("status")
        if status in counts:
            counts[status] += 1
    counts["generated"] = counts["success"]
    counts["errors"] = counts["error"]
    return counts


def _merge_ordered_lists(existing: Any, current: Any) -> Any:
    if not isinstance(existing, list) or not isinstance(current, list):
        return current
    merged = []
    for item in [*existing, *current]:
        if item not in merged:
            merged.append(item)
    return merged


def normal_rgb_to_float(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return arr * 2.0 - 1.0


class ClipScorer:
    """CLIPScore-compatible scorer using the project's existing transformers dependency."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda:0",
    ) -> None:
        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def score(self, image: Image.Image, text: str) -> float:
        inputs = self.processor(
            text=[text],
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            image_features = self._feature_tensor(
                self.model.get_image_features(pixel_values=inputs["pixel_values"])
            )
            text_features = self._feature_tensor(
                self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            score = 100.0 * (image_features * text_features).sum(dim=-1)
        return float(score.item())

    def _feature_tensor(self, features: Any) -> torch.Tensor:
        if isinstance(features, self.torch.Tensor):
            return features
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        if isinstance(features, tuple):
            return features[1]
        raise TypeError(f"Unsupported CLIP feature output: {type(features)!r}")
