"""SAM3-based segmentation for iterative region selection."""

import logging

import numpy as np
import torch
from PIL import Image, ImageDraw

from text2tactilegraphics.generation.models import ModelManager, global_model_manager
from text2tactilegraphics.generation.utils import mask_to_image

logger = logging.getLogger(__name__)

# =============================================================================
# SegmentationEngine
# =============================================================================


class SegmentationEngine:
    """Click- and text-based segmentation via the SAM3 model family."""

    def __init__(self, model_manager: ModelManager | None = None) -> None:
        self.mm = model_manager or global_model_manager()

    def segment_with_points(
        self,
        image: Image.Image,
        points: list[tuple[int, int]],
        labels: list[int] | None = None,
    ) -> np.ndarray:
        """Run SAM3 with the given points and labels.

        If ``labels`` is None, assume all labels are positive.
        """
        if labels is None:
            labels = [1] * len(points)

        sam = self.mm.sam3_tracker
        model, processor, device = sam["model"], sam["processor"], sam["device"]

        # SAM3 expects [image][object][point][coordinates]:
        #   one image, one object, N points, each point is [x, y].
        inputs = processor(
            images=image,
            input_points=[[[list(p) for p in points]]],
            input_labels=[[labels]],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"], binarize=True
        )[0]
        return masks[0, 0].numpy().astype(np.float32)

    def segment_with_text(
        self,
        image: Image.Image,
        text_prompt: str,
        confidence_threshold: float = 0.5,
    ) -> list[tuple[np.ndarray, float]]:
        """Segment `image` using a free-form text prompt.

        Returns a list of `(mask, score)` tuples — one per detected
        instance whose score clears `confidence_threshold`.
        """
        sam = self.mm.sam3_text
        model, processor, device = sam["model"], sam["processor"], sam["device"]

        inputs = processor(
            images=image.convert("RGB"),
            text=text_prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=confidence_threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results["masks"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        return [
            (mask.astype(np.float32), float(score))
            for mask, score in zip(masks, scores)
        ]


# =============================================================================
# Drawing helpers
# =============================================================================


def apply_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    opacity: float = 0.5,
) -> Image.Image:
    """Overlay a colored mask onto ``image``."""
    image = image.convert("RGBA")

    mask_img = mask_to_image(mask)
    if mask_img.size != image.size:
        logger.warning("mask and image are different sizes, resizing…")
        mask_img = mask_img.resize(image.size, resample=Image.Resampling.NEAREST)

    color_fill = Image.new("RGBA", image.size, color + (0,))
    color_fill.putalpha(mask_img.point(lambda v: int(v * opacity)))
    return Image.alpha_composite(image, color_fill).convert("RGB")


def draw_points_on_image(
    image: Image.Image,
    points: list[tuple[int, int]],
    labels: list[int] | None = None,
    radius: float = 8,
) -> Image.Image:
    """Draw red/blue dots for positive/negative click points."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    labels = labels if labels is not None else [1] * len(points)

    for (x, y), label in zip(points, labels, strict=True):
        color = "red" if label == 1 else "dodgerblue"
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline="white",
            width=3,
        )
    return image
