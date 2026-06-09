from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from text2tactilegraphics.generation.utils import (
    as_pil,
    center_crop,
    center_crop_array,
    center_seam_mask,
    depth_to_image,
    displacement_to_image,
    mask_to_image,
    merge_images,
    normal_to_image,
    open_rgb_image,
    split_image,
    swap_to_center_seam,
    tile_image,
    tiled_seam_mask,
)

# =============================================================================
# Test image helpers
# =============================================================================


class TestOpenRGBImage:
    def test_pil_rgb_passthrough(self):
        img = Image.new("RGB", (8, 8), (10, 20, 30))
        assert open_rgb_image(img) is img

    def test_pil_rgba_converted_to_rgb(self):
        img = Image.new("RGBA", (8, 8), (10, 20, 30, 200))
        out = open_rgb_image(img)
        assert out.mode == "RGB"

    def test_path_input(self, tmp_path: Path):
        p = tmp_path / "in.png"
        Image.new("RGB", (4, 4), (50, 60, 70)).save(p)
        out = open_rgb_image(str(p))
        assert out.mode == "RGB"
        assert out.size == (4, 4)

    def test_pathlib_input(self, tmp_path: Path):
        p = tmp_path / "in.png"
        Image.new("RGB", (4, 4), (50, 60, 70)).save(p)
        out = open_rgb_image(p)
        assert out.mode == "RGB"


class TestAsPIL:
    def test_pil_passthrough(self):
        img = Image.new("RGB", (16, 16), (10, 20, 30))
        assert as_pil(img) is img

    def test_uint8_array(self):
        arr = np.full((8, 8, 3), 127, dtype=np.uint8)
        out = as_pil(arr)
        assert isinstance(out, Image.Image)
        assert out.size == (8, 8)
        assert (np.array(out) == 127).all()

    def test_float_in_0_1_range(self):
        arr = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = np.array(as_pil(arr))
        # 0.5 * 255 → ~127
        assert 120 <= out[0, 0, 0] <= 130

    def test_float_in_minus1_to_1_range(self):
        # All zeros in normal-space → 0.5 in RGB-space → ~127 uint8.
        arr = np.zeros((4, 4, 3), dtype=np.float32)
        # Force the "min < 0" branch by including a negative.
        arr[0, 0, 0] = -1.0
        out = np.array(as_pil(arr))
        # The (-1)-pixel becomes 0; the 0-pixels become 127.
        assert out[0, 0, 0] == 0
        assert 120 <= out[1, 1, 0] <= 130

    def test_max_above_1_treated_as_0_255_range(self):
        # If max > 1 the array is assumed to already be in [0, 255] —
        # values pass through with a clip-and-cast (not rescaled).
        arr = np.full((4, 4, 3), 50.0, dtype=np.float32)
        out = np.array(as_pil(arr))
        assert (out == 50).all()

    def test_clamps_above_255(self):
        # Values exceeding 255 in the "[0, 255] floats" branch get clipped.
        arr = np.full((4, 4, 3), 1000.0, dtype=np.float32)
        out = np.array(as_pil(arr))
        assert (out == 255).all()


class TestMaskToImage:
    def test_binary_mask_scales_to_uint8(self):
        mask = np.array([[True, False], [False, True]])
        out = np.array(mask_to_image(mask))
        assert out.max() == 255
        assert out.min() == 0


class TestNormalToImage:
    def test_returns_rgb_image(self):
        normal = np.zeros((4, 4, 3), dtype=np.float32)
        normal[..., 2] = 1.0
        out = normal_to_image(normal)
        assert out.mode == "RGB"


class TestDepthToImage:
    def test_returns_grayscale_image(self):
        depth = np.full((4, 4), 0.5, dtype=np.float32)
        out = depth_to_image(depth)
        assert out.mode == "L"


class TestDisplacementToImage:
    def test_normalizes_to_uint8(self):
        arr = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)
        img = displacement_to_image(arr)
        assert isinstance(img, Image.Image)
        # min → 0, max → 255 after normalization.
        pix = np.asarray(img)
        assert pix.min() == 0
        assert pix.max() == 255

    def test_constant_array_does_not_divide_by_zero(self):
        # All values equal — `(arr - min) / (max - min + 1e-8)` should not raise.
        arr = np.full((4, 4), 0.7, dtype=np.float32)
        img = displacement_to_image(arr)
        assert isinstance(img, Image.Image)
        assert np.asarray(img).shape == (4, 4)

    def test_clip_when_normalize_false(self):
        arr = np.array([[-1.0, 0.2], [1.2, 2.0]], dtype=np.float32)
        img = displacement_to_image(arr, normalize=False)
        pix = np.asarray(img)
        assert pix.min() == 0
        assert pix.max() == 255


# =============================================================================
# Test cropping functions
# =============================================================================


class TestCenterCrop:
    def test_crops_to_requested_size(self):
        img = Image.new("RGB", (1024, 768), (255, 0, 0))
        out = center_crop(img, size=512)
        assert out.size == (512, 512)

    def test_crop_is_centered(self):
        # Make an image with a recognizable center color
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[40:60, 40:60] = 255  # white square at center
        img = Image.fromarray(arr)
        out = center_crop(img, size=20)
        # Out should be entirely white (the center square)
        assert (np.array(out) == 255).all()


class TestCenterCropArray:
    def test_crops_to_requested_size(self):
        arr = np.zeros((768, 1024, 3), dtype=np.uint8)
        out = center_crop_array(arr, size=512)
        assert out.shape == (512, 512, 3)

    def test_2d_array(self):
        arr = np.arange(100 * 100).reshape(100, 100)
        out = center_crop_array(arr, size=20)
        assert out.shape == (20, 20)


# =============================================================================
# Test tiling functions
# =============================================================================


class TestTileImage:
    def test_3x3_default(self):
        patch = Image.new("RGB", (32, 32), (123, 200, 50))
        out = tile_image(patch)
        assert out.size == (96, 96)

    def test_mxn_tiling(self):
        patch = Image.new("RGB", (16, 32), (50, 50, 50))
        out = tile_image(patch, rows=2, cols=4)
        # m=rows (y), n=cols (x). Output size = (patch_w * n, patch_h * m)
        assert out.size == (16 * 4, 32 * 2)

    def test_uniform_patch_tiles_uniformly(self):
        patch = Image.new("RGB", (8, 8), (10, 20, 30))
        out = tile_image(patch, rows=2, cols=2)
        arr = np.array(out)
        assert (arr[..., 0] == 10).all()
        assert (arr[..., 1] == 20).all()
        assert (arr[..., 2] == 30).all()


class TestTiledSeamMask:
    def test_tiled_seam_mask(self):
        img = Image.new("RGB", (64, 64), (100, 100, 100))
        mask = tiled_seam_mask(img, n_tiles=3, mask_portion=0.1)
        arr = np.array(mask)
        # Some pixels are white (255 = inpaint region), some are black
        assert (arr == 255).any()
        assert (arr == 0).any()


# =============================================================================
# Test seam-swapping helper functions
# =============================================================================

_RED = (255, 0, 0)
_GREEN = (0, 255, 0)
_BLUE = (0, 0, 255)
_YELLOW = (255, 255, 0)


class TestSplitMergeImages:
    @pytest.mark.parametrize("width", [8, 9])
    def test_split_horizontal(self, width: int):
        mid = width // 2
        input_arr = np.zeros((4, width, 3), dtype=np.uint8)
        input_arr[:, :mid] = _RED
        input_arr[:, mid:] = _BLUE

        left, right = split_image(Image.fromarray(input_arr), "horizontal")

        np.testing.assert_array_equal(np.array(left), input_arr[:, :mid])
        np.testing.assert_array_equal(np.array(right), input_arr[:, mid:])

    @pytest.mark.parametrize("height", [8, 9])
    def test_split_vertical(self, height: int):
        mid = height // 2
        input_arr = np.zeros((height, 4, 3), dtype=np.uint8)
        input_arr[:mid, :] = _RED
        input_arr[mid:, :] = _BLUE

        top, bottom = split_image(Image.fromarray(input_arr), "vertical")

        np.testing.assert_array_equal(np.array(top), input_arr[:mid, :])
        np.testing.assert_array_equal(np.array(bottom), input_arr[mid:, :])

    @pytest.mark.parametrize("width", [4, 5])
    def test_merge_horizontal(self, width: int):
        left_arr = np.full((4, width, 3), _RED, dtype=np.uint8)
        right_arr = np.full((4, width, 3), _GREEN, dtype=np.uint8)
        expected_arr = np.concatenate([left_arr, right_arr], axis=1)

        merged = merge_images(
            Image.fromarray(left_arr), Image.fromarray(right_arr), "horizontal"
        )

        np.testing.assert_array_equal(np.array(merged), expected_arr)

    @pytest.mark.parametrize("height", [4, 5])
    def test_merge_vertical(self, height: int):
        left_arr = np.full((height, 4, 3), _RED, dtype=np.uint8)
        right_arr = np.full((height, 4, 3), _GREEN, dtype=np.uint8)
        expected_arr = np.concatenate([left_arr, right_arr], axis=0)

        merged = merge_images(
            Image.fromarray(left_arr), Image.fromarray(right_arr), "vertical"
        )

        np.testing.assert_array_equal(np.array(merged), expected_arr)


class TestSwapToCenterSeam:
    def test_horizontal_swaps_halves(self):
        # Input:  [ red | blue ]      Expected:  [ blue | red ]
        input_arr = np.zeros((4, 8, 3), dtype=np.uint8)
        input_arr[:, :4] = _RED
        input_arr[:, 4:] = _BLUE

        expected_arr = np.zeros_like(input_arr)
        expected_arr[:, :4] = _BLUE
        expected_arr[:, 4:] = _RED

        actual = swap_to_center_seam(Image.fromarray(input_arr), "horizontal")
        np.testing.assert_array_equal(np.array(actual), expected_arr)

    def test_vertical_swaps_halves(self):
        # Input:     red               Expected:    blue
        #            blue                            red
        input_arr = np.zeros((8, 4, 3), dtype=np.uint8)
        input_arr[:4, :] = _RED
        input_arr[4:, :] = _BLUE

        expected_arr = np.zeros_like(input_arr)
        expected_arr[:4, :] = _BLUE
        expected_arr[4:, :] = _RED

        actual = swap_to_center_seam(Image.fromarray(input_arr), "vertical")
        np.testing.assert_array_equal(np.array(actual), expected_arr)

    def test_both_swaps_quadrants(self):
        # `direction="both"` applies a horizontal swap then a vertical one:
        #     Input               Expected
        #     ┌────┬────┐         ┌────┬────┐
        #     │ R  │ G  │         │ Y  │ B  │
        #     ├────┼────┤   →     ├────┼────┤
        #     │ B  │ Y  │         │ G  │ R  │
        #     └────┴────┘         └────┴────┘
        input_arr = np.zeros((8, 8, 3), dtype=np.uint8)
        input_arr[:4, :4] = _RED  # TL
        input_arr[:4, 4:] = _GREEN  # TR
        input_arr[4:, :4] = _BLUE  # BL
        input_arr[4:, 4:] = _YELLOW  # BR

        expected_arr = np.zeros_like(input_arr)
        expected_arr[:4, :4] = _YELLOW
        expected_arr[:4, 4:] = _BLUE
        expected_arr[4:, :4] = _GREEN
        expected_arr[4:, 4:] = _RED

        actual = swap_to_center_seam(Image.fromarray(input_arr), "both")
        np.testing.assert_array_equal(np.array(actual), expected_arr)


class TestCenterSeamMask:
    @pytest.mark.parametrize("width", [16, 17])
    def test_horizontal_line_in_center(self, width):
        height, line_width = 8, 4
        x0 = (width - line_width) // 2

        expected = np.zeros((height, width), dtype=np.uint8)
        expected[:, x0 : x0 + line_width] = 255

        mask = center_seam_mask(
            width, height, line_width=line_width, direction="horizontal"
        )
        np.testing.assert_array_equal(np.array(mask), expected)

    @pytest.mark.parametrize("height", [16, 17])
    def test_vertical_line_in_center(self, height):
        width, line_width = 8, 4
        y0 = (height - line_width) // 2

        expected = np.zeros((height, width), dtype=np.uint8)
        expected[y0 : y0 + line_width, :] = 255

        mask = center_seam_mask(
            width, height, line_width=line_width, direction="vertical"
        )
        np.testing.assert_array_equal(np.array(mask), expected)

    @pytest.mark.parametrize("side", [16, 17])
    def test_both_directions_form_cross(self, side):
        line_width = 4
        x0 = (side - line_width) // 2
        y0 = x0

        # Cross-shaped mask: union of the vertical and horizontal bands.
        expected = np.zeros((side, side), dtype=np.uint8)
        expected[:, x0 : x0 + line_width] = 255
        expected[y0 : y0 + line_width, :] = 255

        mask = center_seam_mask(side, side, line_width=line_width, direction="both")
        np.testing.assert_array_equal(np.array(mask), expected)
