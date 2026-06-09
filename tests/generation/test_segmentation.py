from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pytest_regressions.image_regression import ImageRegressionFixture

from text2tactilegraphics.generation.segmentation import (
    SegmentationEngine,
    apply_mask_overlay,
    draw_points_on_image,
)
from text2tactilegraphics.generation.utils import mask_to_image

_DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Test SegmentationEngine
# =============================================================================


@pytest.fixture
def engine():
    g = SegmentationEngine()
    yield g
    g.mm.unload_all_models()


@pytest.mark.slow
class TestSegmentationEngine:
    def test_segment_with_points(
        self, engine: SegmentationEngine, image_regression: ImageRegressionFixture
    ):
        image = Image.open(_DATA_DIR / "dolphin.png").convert("RGB")
        mask = engine.segment_with_points(image, points=[(512, 512)], labels=[1])
        mask_img = mask_to_image(mask)
        image_regression.check(mask_img)

    def test_segment_with_text(
        self, engine: SegmentationEngine, image_regression: ImageRegressionFixture
    ):
        image = Image.open(_DATA_DIR / "dolphin.png").convert("RGB")
        segments = engine.segment_with_text(
            image, text_prompt="dolphin", confidence_threshold=0.5
        )
        best_mask, _score = max(segments, key=lambda s: s[1])
        mask_img = mask_to_image(best_mask)
        image_regression.check(mask_img)


# =============================================================================
# Test drawing utilities
# =============================================================================


class TestApplyMaskOverlay:
    def test_apply_mask_overlay(self, image_regression: ImageRegressionFixture):
        base = Image.open(_DATA_DIR / "dolphin.png").convert("RGB")
        h, w = base.size[1], base.size[0]
        y, x = np.ogrid[:h, :w]
        mask = ((x - w / 2) / (w / 4)) ** 2 + ((y - h / 2) / (h / 4)) ** 2 < 1
        out = apply_mask_overlay(
            base,
            mask,  # type:ignore
            color=(255, 100, 100),
            opacity=0.5,
        )
        image_regression.check(out)


class TestDrawPointsOnImage:
    def test_empty_points_returns_copy(self):
        base = Image.new("RGB", (32, 32), (10, 20, 30))
        out = draw_points_on_image(base, [])
        np.testing.assert_array_equal(np.array(base), np.array(out))

    def test_does_not_mutate_input_pil(self):
        base = Image.new("RGB", (32, 32), (255, 255, 255))
        before = np.array(base).copy()
        draw_points_on_image(base, [(16, 16)])
        np.testing.assert_array_equal(before, np.array(base))

    def test_draw_points_on_image(self, image_regression: ImageRegressionFixture):
        base = Image.open(_DATA_DIR / "dolphin.png").convert("RGB")
        out = draw_points_on_image(
            base,
            points=[(300, 400), (700, 500), (512, 800)],
            labels=[1, 1, 0],
            radius=24,
        )
        image_regression.check(out)
