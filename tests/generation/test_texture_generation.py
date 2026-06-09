from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from pytest_regressions.image_regression import ImageRegressionFixture
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from text2tactilegraphics import GeometryEstimator, TextureGenerator
from text2tactilegraphics.generation.utils import depth_to_image, normal_to_image

_DATA_DIR = Path(__file__).parent / "data"

_AVOCADO_IMG = Image.open(_DATA_DIR / "avocado.png").convert("RGB")
_AVOCADO_GENERATE_KWARGS: dict = {
    "prompt": "an avocado skin",
    "steps": 4,
    "seed": 42,
    "height": 1024,
    "width": 1024,
}
_AVOCADO_NORMAL_KWARGS: dict = {
    "image": _AVOCADO_IMG,
    "crop": True,
    "crop_size": 512,
}
_AVOCADO_DEPTH_KWARGS: dict = {
    "image": _AVOCADO_IMG,
    "normalize": True,
    "crop": True,
    "crop_size": 512,
}


@pytest.fixture
def gen():
    g = TextureGenerator()
    yield g
    g.mm.unload_all_models()


@pytest.fixture
def geom():
    g = GeometryEstimator()
    yield g
    g.mm.unload_all_models()


def _patch_moge(
    geom: GeometryEstimator,
    *,
    normal: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> None:
    result: dict[str, torch.Tensor] = {}
    if normal is not None:
        result["normal"] = torch.as_tensor(normal, dtype=torch.float32)
    if depth is not None:
        result["depth"] = torch.as_tensor(depth, dtype=torch.float32)
    if mask is not None:
        result["mask"] = torch.as_tensor(mask, dtype=torch.bool)
    geom._run_moge = Mock(return_value=result)


_DUMMY_IMAGE = Image.new("RGB", (8, 8))


class TestComputeNormal:
    def test_returns_opengl_normal(self, geom: GeometryEstimator):
        raw = np.array(
            [[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32
        )
        _patch_moge(geom, normal=raw)

        out = geom.compute_normal(_DUMMY_IMAGE, crop=False)

        # Normalized to unit length, then Y and Z negated.
        expected = np.array(
            [[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]], dtype=np.float32
        )
        np.testing.assert_allclose(out, expected)

    def test_zero_vector_stays_zero(self, geom: GeometryEstimator):
        raw = np.zeros((1, 1, 3), dtype=np.float32)
        _patch_moge(geom, normal=raw)

        out = geom.compute_normal(_DUMMY_IMAGE, crop=False)

        # The `where=norm != 0` guard keeps the divide from blowing up,
        # and `-0.0` is still 0.
        assert (out == 0).all()

    def test_crops_when_requested(self, geom: GeometryEstimator):
        raw = np.ones((8, 8, 3), dtype=np.float32)
        _patch_moge(geom, normal=raw)

        out = geom.compute_normal(_DUMMY_IMAGE, crop=True, crop_size=4)

        assert out.shape == (4, 4, 3)

    def test_no_crop_keeps_full_size(self, geom: GeometryEstimator):
        raw = np.ones((8, 8, 3), dtype=np.float32)
        _patch_moge(geom, normal=raw)

        out = geom.compute_normal(_DUMMY_IMAGE, crop=False)

        assert out.shape == (8, 8, 3)

    @pytest.mark.slow
    def test_compute_normal_e2e(
        self,
        geom: GeometryEstimator,
        ndarrays_regression: NDArraysRegressionFixture,
        tmp_path: Path,
    ):
        actual = geom.compute_normal(**_AVOCADO_NORMAL_KWARGS)
        img = normal_to_image(actual)
        img.save(tmp_path / "normal.png")
        ndarrays_regression.check({"normal": actual})


class TestComputeDepth:
    def test_strips_infs_using_mask(self, geom: GeometryEstimator):
        depth = np.array([[1.0, 2.0], [np.inf, 3.0]], dtype=np.float32)
        # The inf corner is excluded from the valid mask; depth's max is
        # then clipped to 3.0 (the max of the masked-in values).
        mask = np.array([[True, True], [False, True]], dtype=bool)
        _patch_moge(geom, depth=depth, mask=mask)

        out = geom.compute_depth(_DUMMY_IMAGE, normalize=False, crop=False)

        np.testing.assert_allclose(out, [[1.0, 2.0], [3.0, 3.0]])

    def test_normalize_maps_to_unit_range_and_inverts(self, geom: GeometryEstimator):
        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        mask = np.ones_like(depth, dtype=bool)
        _patch_moge(geom, depth=depth, mask=mask)

        out = geom.compute_depth(_DUMMY_IMAGE, normalize=True, crop=False)

        np.testing.assert_allclose(out, [[1.0, 2.0 / 3.0], [1.0 / 3.0, 0.0]])

    def test_crops_before_normalize(self, geom: GeometryEstimator):
        """Normalization should map the dynamic range of the cropped region,
        not of the whole MoGe output."""
        depth = np.full((4, 4), 100.0, dtype=np.float32)
        depth[1:3, 1:3] = [[1.0, 2.0], [3.0, 4.0]]
        mask = np.ones_like(depth, dtype=bool)
        _patch_moge(geom, depth=depth, mask=mask)

        out = geom.compute_depth(_DUMMY_IMAGE, normalize=True, crop=True, crop_size=2)

        np.testing.assert_allclose(out, [[1.0, 2.0 / 3.0], [1.0 / 3.0, 0.0]])

    def test_no_crop_keeps_full_size(self, geom: GeometryEstimator):
        depth = np.zeros((8, 8), dtype=np.float32)
        mask = np.ones_like(depth, dtype=bool)
        _patch_moge(geom, depth=depth, mask=mask)

        out = geom.compute_depth(_DUMMY_IMAGE, normalize=False, crop=False)

        assert out.shape == (8, 8)

    @pytest.mark.slow
    def test_compute_depth_e2e(
        self,
        geom: GeometryEstimator,
        ndarrays_regression: NDArraysRegressionFixture,
        tmp_path: Path,
    ):
        actual = geom.compute_depth(**_AVOCADO_DEPTH_KWARGS)
        img = depth_to_image(actual)
        img.save(tmp_path / "depth.png")
        ndarrays_regression.check({"depth": actual})


class TestGenerate:
    @pytest.mark.slow
    def test_generate_e2e(
        self,
        gen: TextureGenerator,
        image_regression: ImageRegressionFixture,
    ):
        actual = gen.generate(**_AVOCADO_GENERATE_KWARGS)
        image_regression.check(actual, diff_threshold=5.0)
