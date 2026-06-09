from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pytest_regressions.image_regression import ImageRegressionFixture
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from text2tactilegraphics.geometry.filtering import (
    HighpassMethod,
    apply_high_pass_to_normal_map,
    height_to_normal,
    highpass_filter_fft,
    normal_to_height,
    normal_to_rgb,
    normalize_vectors,
    rgb_to_normal,
)

_DATA_DIR = Path(__file__).parent / "data"


def _make_flat_normal(h: int = 64, w: int = 64) -> np.ndarray:
    """Flat normal map (all pointing +Z)."""
    n: np.ndarray = np.zeros((h, w, 3), dtype=np.float32)
    n[..., 2] = 1.0  # (0, 0, 1) in [-1, 1]
    return normal_to_rgb(n)


class TestApplyHighPassToNormalMap:
    def test_uint8_input_normalized(self):
        n = (_make_flat_normal(32, 32) * 255).astype(np.uint8)
        result = apply_high_pass_to_normal_map(n, freq_threshold=8)
        assert isinstance(result, Image.Image)

    def test_accepts_pil_image_input(self):
        n = (_make_flat_normal(32, 32) * 255).astype(np.uint8)
        img = Image.fromarray(n)
        result = apply_high_pass_to_normal_map(img, freq_threshold=8)
        assert isinstance(result, Image.Image)

    def test_flat_normal_stays_flat(self):
        # A perfectly flat normal map has no high-frequency content,
        # so HP filtering should produce a near-flat normal map (still pointing +Z).
        n = _make_flat_normal(32, 32)
        result_img = apply_high_pass_to_normal_map(n, freq_threshold=8)
        result: np.ndarray = np.array(result_img).astype(np.float32)
        result = rgb_to_normal(result / 255.0)
        # Z channel should still be the dominant ~1 (encoded as 255)
        z = result[..., 2]
        assert z.mean() > 0.9

    @pytest.mark.parametrize("method", ["per_channel", "height_integration"])
    def test_high_pass_normal_regression(
        self, method: HighpassMethod, image_regression: ImageRegressionFixture
    ):
        normal = Image.open(_DATA_DIR / "normal.png").convert("RGB")
        result = apply_high_pass_to_normal_map(
            normal, freq_threshold=120, method=method
        )
        image_regression.check(result, basename=f"highpass_{method}")


class TestHighpassFilterFFT:
    def test_zeros_in_zeros_out(self):
        data = np.zeros((32, 32), dtype=np.float32)
        out = highpass_filter_fft(data, cutoff_ratio=0.1)
        np.testing.assert_allclose(out, 0)

    def test_dc_signal_removed(self):
        data = np.full((32, 32), 5.0, dtype=np.float32)
        out = highpass_filter_fft(data, cutoff_ratio=0.1)
        # DC (constant) should be removed
        assert abs(out.mean()) < 1e-4

    def test_shape_preserved(self):
        data = np.random.RandomState(0).rand(16, 24).astype(np.float32)
        out = highpass_filter_fft(data, cutoff_ratio=0.1)
        assert out.shape == data.shape

    def test_multichannel(self):
        data = np.random.RandomState(0).rand(16, 16, 3).astype(np.float32)
        out = highpass_filter_fft(data, cutoff_ratio=0.1)
        assert out.shape == data.shape


class TestHeightNormal:
    def test_height_to_normal_shape(self):
        h = np.random.RandomState(0).rand(16, 16).astype(np.float32)
        n = height_to_normal(h)
        assert n.shape == (16, 16, 3)

    def test_height_to_normal_unit_length(self):
        h: np.ndarray = np.random.RandomState(0).rand(16, 16).astype(np.float32)
        h *= 0.01
        n = height_to_normal(h)
        norms = np.linalg.norm(n, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_flat_height_gives_z_normal(self):
        h = np.zeros((16, 16), dtype=np.float32)
        n = height_to_normal(h)
        np.testing.assert_allclose(n[..., 0], 0, atol=1e-6)
        np.testing.assert_allclose(n[..., 1], 0, atol=1e-6)
        np.testing.assert_allclose(n[..., 2], 1, atol=1e-6)

    def test_integrate_flat_normal_yields_flat_height(self):
        # Flat normal (0,0,1) integrates to a constant (zero-mean) height
        n = np.zeros((16, 16, 3), dtype=np.float32)
        n[..., 2] = 1.0
        h = normal_to_height(n)
        np.testing.assert_allclose(h, 0, atol=1e-5)

    def test_normal_to_height(self, ndarrays_regression: NDArraysRegressionFixture):
        normal = np.load(_DATA_DIR / "normal.npy")
        height = normal_to_height(normal)
        ndarrays_regression.check({"height": height})

    def test_height_to_normal(self, ndarrays_regression: NDArraysRegressionFixture):
        height = np.load(_DATA_DIR / "height.npy")
        normal = height_to_normal(height)
        ndarrays_regression.check({"normal": normal})


class TestNormalizeVectors:
    def test_unit_length(self):
        v = np.array([[3.0, 0.0, 4.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        out = normalize_vectors(v)
        norms = np.linalg.norm(out, axis=-1)
        np.testing.assert_allclose(norms, 1.0)

    def test_zero_vector_stable(self):
        v = np.zeros((1, 3))
        out = normalize_vectors(v)
        # Should not nan out
        assert np.all(np.isfinite(out))
