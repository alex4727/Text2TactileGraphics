from typing import Literal

import torch
from torch.nn import functional as F

# =============================================================================
# Zero padding on (B, C, H, W) tensors
# =============================================================================


def pad_tensor_x(tensor: torch.Tensor, max_width: int) -> torch.Tensor:
    """Zero-pad `tensor` by `max_width` on the left and right."""
    return F.pad(tensor, (max_width, max_width, 0, 0))


def pad_tensor_y(tensor: torch.Tensor, max_height: int) -> torch.Tensor:
    """Zero-pad `tensor` by `max_height` on the top and bottom."""
    return F.pad(tensor, (0, 0, max_height, max_height))


def pad_tensor_xy(
    tensor: torch.Tensor, max_width: int, max_height: int
) -> torch.Tensor:
    """Zero-pad `tensor` on all four sides."""
    return F.pad(tensor, (max_width, max_width, max_height, max_height))


# =============================================================================
# Edge wrapping on already-padded (B, C, H, W) tensors
# =============================================================================
#
# `wrap_edges_*` overwrite the outer band on a side with the inner band on the
# opposite side. Required spatial dim: strictly greater than `2 * pad` so the
# source and destination bands are disjoint.


def wrap_edges_x(tensor: torch.Tensor, max_width: int) -> torch.Tensor:
    """Wrap horizontal edges of an already-padded tensor."""
    return _wrap_edges_along(tensor, max_width, dim=3)


def wrap_edges_y(tensor: torch.Tensor, max_height: int) -> torch.Tensor:
    """Wrap vertical edges of an already-padded tensor."""
    return _wrap_edges_along(tensor, max_height, dim=2)


def wrap_edges_xy(
    tensor: torch.Tensor, max_width: int, max_height: int
) -> torch.Tensor:
    """Wrap both axes of an already-padded tensor."""
    return wrap_edges_y(wrap_edges_x(tensor, max_width), max_height)


def _wrap_edges_along(tensor: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    """Wrap the outer `pad`-wide bands along `dim` with content from the
    opposite inner band (used by both x- and y-axis wrapping)."""
    n = tensor.size(dim)
    result = tensor.clone()
    # Outer-left/top band ← inner band from the right/bottom of the interior.
    result.narrow(dim, 0, pad).copy_(tensor.narrow(dim, n - 2 * pad, pad))
    # Outer-right/bottom band ← inner band from the left/top of the interior.
    result.narrow(dim, n - pad, pad).copy_(tensor.narrow(dim, pad, pad))
    return result


# =============================================================================
# Periodic-boundary projection
# =============================================================================


def project_periodic_boundary(
    tensor: torch.Tensor,
    max_width: int,
    max_height: int,
    direction: Literal["x", "y", "xy"] = "xy",
    band_width: int = 4,
    band_height: int = 4,
) -> torch.Tensor:
    """Enforce periodic boundaries by averaging opposite boundary bands.

    Makes left=right and/or top=bottom edges match.
    """
    result = tensor.clone()
    h, w = tensor.shape[2], tensor.shape[3]

    # Center-tile bounds (inside the padded tensor) and capped band sizes.
    top, bot = max_height, h - max_height
    lft, rgt = max_width, w - max_width
    bw = min(band_width, max_width // 2)
    bh = min(band_height, max_height // 2)
    full = (slice(None), slice(None))  # (batch, channels)

    if direction == "xy":
        # 1. Average all four corners together (4-way mean).
        cs = min(bw, bh)
        _average_bands(
            result,
            [
                (*full, slice(top, top + cs), slice(lft, lft + cs)),  # TL
                (*full, slice(top, top + cs), slice(rgt - cs, rgt)),  # TR
                (*full, slice(bot - cs, bot), slice(lft, lft + cs)),  # BL
                (*full, slice(bot - cs, bot), slice(rgt - cs, rgt)),  # BR
            ],
        )
        # 2. Left/right edge bands (excluding corners).
        _average_bands(
            result,
            [
                (*full, slice(top + cs, bot - cs), slice(lft, lft + bw)),
                (*full, slice(top + cs, bot - cs), slice(rgt - bw, rgt)),
            ],
        )
        # 3. Top/bottom edge bands (excluding corners).
        _average_bands(
            result,
            [
                (*full, slice(top, top + bh), slice(lft + cs, rgt - cs)),
                (*full, slice(bot - bh, bot), slice(lft + cs, rgt - cs)),
            ],
        )
    elif direction == "x":
        _average_bands(
            result,
            [
                (*full, slice(top, bot), slice(lft, lft + bw)),
                (*full, slice(top, bot), slice(rgt - bw, rgt)),
            ],
        )
    elif direction == "y":
        _average_bands(
            result,
            [
                (*full, slice(top, top + bh), slice(lft, rgt)),
                (*full, slice(bot - bh, bot), slice(lft, rgt)),
            ],
        )

    return result


def _average_bands(result: torch.Tensor, slices: list[tuple[slice, ...]]) -> None:
    """Replace every band in `slices` with the elementwise mean of all bands.

    All slices must select tensors of the same shape. Mutates `result`.
    """
    avg = sum(result[s] for s in slices) / len(slices)
    for s in slices:
        result[s] = avg
