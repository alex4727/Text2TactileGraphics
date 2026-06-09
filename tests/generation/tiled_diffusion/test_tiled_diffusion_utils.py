import torch

from text2tactilegraphics.generation.tiled_diffusion.utils import (
    pad_tensor_x,
    pad_tensor_xy,
    pad_tensor_y,
    project_periodic_boundary,
    wrap_edges_x,
    wrap_edges_xy,
    wrap_edges_y,
)


class TestPadTensors:
    def test_pad_x_shape(self):
        t = torch.ones(1, 3, 8, 16)
        out = pad_tensor_x(t, max_width=4)
        assert out.shape == (1, 3, 8, 16 + 8)

    def test_pad_x_preserves_center(self):
        t = torch.ones(1, 3, 8, 16)
        out = pad_tensor_x(t, max_width=4)
        assert torch.all(out[:, :, :, 4:20] == 1)
        assert torch.all(out[:, :, :, :4] == 0)
        assert torch.all(out[:, :, :, 20:] == 0)

    def test_pad_y_shape(self):
        t = torch.ones(1, 3, 8, 16)
        out = pad_tensor_y(t, max_height=2)
        assert out.shape == (1, 3, 12, 16)

    def test_pad_xy_combines(self):
        t = torch.ones(1, 3, 8, 16)
        out = pad_tensor_xy(t, max_width=4, max_height=2)
        assert out.shape == (1, 3, 12, 24)


class TestWrapEdges:
    def test_wrap_edges_x_preserves_shape(self):
        t = torch.rand(1, 3, 16, 32)
        out = wrap_edges_x(t, max_width=4)
        assert out.shape == t.shape

    def test_wrap_edges_x_copies_correct_band(self):
        t = torch.rand(1, 3, 16, 32)
        out = wrap_edges_x(t, max_width=4)
        # Left band gets src[..., 32-8 : 32-4] = src[..., 24:28]
        assert torch.equal(out[:, :, :, :4], t[:, :, :, 24:28])
        # Right band gets src[..., 4:8]
        assert torch.equal(out[:, :, :, 28:], t[:, :, :, 4:8])

    def test_wrap_edges_y_copies_correct_band(self):
        t = torch.rand(1, 3, 32, 16)
        out = wrap_edges_y(t, max_height=4)
        assert torch.equal(out[:, :, :4, :], t[:, :, 24:28, :])
        assert torch.equal(out[:, :, 28:, :], t[:, :, 4:8, :])

    def test_wrap_edges_xy_shape_preserved(self):
        t = torch.rand(1, 3, 32, 32)
        out = wrap_edges_xy(t, max_width=4, max_height=4)
        assert out.shape == t.shape


class TestProjectPeriodicBoundary:
    def test_xy_makes_edges_match(self):
        # Center tile occupies the full tensor (max_w = max_h = 0 would be degenerate)
        t = torch.rand(1, 1, 32, 32)
        # Pad so we have an inner "tile" region
        padded = pad_tensor_xy(t, max_width=8, max_height=8)
        out = project_periodic_boundary(
            padded,
            max_width=8,
            max_height=8,
            direction="xy",
            band_width=4,
            band_height=4,
        )
        # Bounds of center tile in padded space
        t_b, b_b = 8, 32 + 8
        l_b, r_b = 8, 32 + 8
        # Left and right bands should match after averaging (excluding corners)
        cs = 4  # corner size = min(bw, bh)
        bw = 4
        bh = 4
        left = out[:, :, t_b + cs : b_b - cs, l_b : l_b + bw]
        right = out[:, :, t_b + cs : b_b - cs, r_b - bw : r_b]
        torch.testing.assert_close(left, right)
        top = out[:, :, t_b : t_b + bh, l_b + cs : r_b - cs]
        bot = out[:, :, b_b - bh : b_b, l_b + cs : r_b - cs]
        torch.testing.assert_close(top, bot)

    def test_x_direction_only(self):
        padded = pad_tensor_xy(torch.rand(1, 1, 16, 16), max_width=4, max_height=4)
        out = project_periodic_boundary(
            padded,
            max_width=4,
            max_height=4,
            direction="x",
            band_width=2,
            band_height=2,
        )
        t_b, b_b, l_b, r_b = 4, 20, 4, 20
        left = out[:, :, t_b:b_b, l_b : l_b + 2]
        right = out[:, :, t_b:b_b, r_b - 2 : r_b]
        torch.testing.assert_close(left, right)
