from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from pytest_regressions.image_regression import ImageRegressionFixture

from text2tactilegraphics.generation.tileable_patch_generation import (
    InterTilePatchGenerator,
    IntraTilePatchGenerator,
    TiledDiffusion,
)

_DATA_DIR = Path(__file__).parent / "data"

_RED = (255, 0, 0)
_BLUE = (0, 0, 255)

_GEOMETRY_IMG = Image.open(_DATA_DIR / "geometry.png").convert("RGB")
_INTRA_TILE_KWARGS: dict = {
    "image": _GEOMETRY_IMG,
    "direction": "both",
    "mask_width": 64,
    "denoising_strength": 0.9,
    "cfg_scale": 4.0,
    "num_inference_steps": 10,
    "inpaint_blur_size": 4,
    "inpaint_blur_sigma": 1.0,
    "seed": 42,
    "restore_orientation": True,
}
_INTER_TILE_KWARGS: dict = {
    "image": _GEOMETRY_IMG,
    "n_tiles": 3,
    "mask_portion": 0.05,
    "denoising_strength": 0.9,
    "cfg_scale": 4.0,
    "num_inference_steps": 10,
    "inpaint_blur_size": 4,
    "inpaint_blur_sigma": 1.0,
    "seed": 42,
}
_TILED_DIFFUSION_KWARGS: dict = {
    "image": _GEOMETRY_IMG,
    "strength": 1.0,
    "guidance_scale": 17.5,
    "num_inference_steps": 100,
    "denoising_end": 0.8,
    "denoising_start": 0.8,
    "max_blend_size": 32,
    "use_soft_mask": True,
    "soft_mask_temp": 0.03,
    "use_periodic_projection": True,
    "proj_cutoff_ratio": 0.6,
    "band_size": 2,
    "seed": 42,
}


class TestIntraTilePatchGenerator:
    @pytest.fixture
    def gen(self):
        g = IntraTilePatchGenerator()
        yield g
        g.mm.unload_all_models()

    @pytest.fixture
    def gen_patched(self) -> IntraTilePatchGenerator:
        fake_pipe = Mock(side_effect=lambda **kw: kw["input_image"])
        mm = Mock()
        mm.qwen_tiling = {"pipeline": fake_pipe}
        g = IntraTilePatchGenerator()
        g.mm = mm
        return g

    def test_restore_orientation_undoes_the_swap(
        self, gen_patched: IntraTilePatchGenerator
    ):
        arr = np.zeros((4, 8, 3), dtype=np.uint8)
        arr[:, :4] = _RED
        arr[:, 4:] = _BLUE

        out = gen_patched.make_tileable(
            Image.fromarray(arr), direction="horizontal", restore_orientation=True
        )

        np.testing.assert_array_equal(np.array(out), arr)

    def test_no_restore_orientation_returns_swapped(
        self, gen_patched: IntraTilePatchGenerator
    ):
        arr = np.zeros((4, 8, 3), dtype=np.uint8)
        arr[:, :4] = _RED
        arr[:, 4:] = _BLUE

        out = gen_patched.make_tileable(
            Image.fromarray(arr), direction="horizontal", restore_orientation=False
        )

        expected = np.concatenate([arr[:, 4:], arr[:, :4]], axis=1)
        np.testing.assert_array_equal(np.array(out), expected)

    @pytest.mark.slow
    def test_make_tileable_intra_tile_e2e(
        self,
        gen: IntraTilePatchGenerator,
        image_regression: ImageRegressionFixture,
    ):
        actual = gen.make_tileable(**_INTRA_TILE_KWARGS)
        image_regression.check(actual, diff_threshold=5.0)


class TestInterTilePatchGenerator:
    @pytest.fixture
    def gen(self):
        g = InterTilePatchGenerator()
        yield g
        g.mm.unload_model("qwen_tiling")

    @pytest.mark.slow
    def test_make_tileable_inter_tile_e2e(
        self,
        gen: InterTilePatchGenerator,
        image_regression: ImageRegressionFixture,
    ):
        actual = gen.make_tileable(**_INTER_TILE_KWARGS)
        image_regression.check(actual, diff_threshold=5.0)


class TestTiledDiffusion:
    @pytest.fixture
    def gen(self):
        g = TiledDiffusion()
        yield g
        g.mm.unload_model("sdxl_tiling")

    @pytest.mark.slow
    def test_make_tileable_tiled_diffusion_e2e(
        self,
        gen: TiledDiffusion,
        image_regression: ImageRegressionFixture,
    ):
        actual = gen.make_tileable(**_TILED_DIFFUSION_KWARGS)
        image_regression.check(actual, diff_threshold=5.0)
