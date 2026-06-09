import os
from pathlib import Path

import pytest
from PIL import Image
from pytest_regressions.image_regression import ImageRegressionFixture

from text2tactilegraphics.generation.base_image_generation import (
    BaseImageGenerator,
)

_DOLPHIN_KWARGS: dict = {
    "prompt": "a dolphin with wings",
    "model": "qwen_edit",
    "steps": 4,
    "seed": 42,
    "height": 1024,
    "width": 1024,
}


@pytest.fixture
def gen():
    g = BaseImageGenerator()
    yield g
    g.mm.unload_all_models()


@pytest.mark.slow
class TestBaseImageGenerator:
    def test_qwen_edit_e2e(
        self,
        gen: BaseImageGenerator,
        image_regression: ImageRegressionFixture,
    ):
        actual = gen.generate(**_DOLPHIN_KWARGS)
        image_regression.check(actual, diff_threshold=5.0)

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GENAI_API_KEY"),
        reason="Gemini API key not set",
    )
    def test_nano_banana_api_e2e(self, gen: BaseImageGenerator, tmp_path: Path):
        img = gen.generate(prompt="a tactile cat", model="nano_banana")
        assert isinstance(img, Image.Image)
        img.save(tmp_path / "cat.png")
