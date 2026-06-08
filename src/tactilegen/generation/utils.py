from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

# =============================================================================
# Image helpers
# =============================================================================


def open_rgb_image(image: str | Path | Image.Image) -> Image.Image:
    """Open a path-like or pass through an existing PIL Image, converted to RGB."""
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB") if image.mode != "RGB" else image


def as_pil(image: Image.Image | np.ndarray) -> Image.Image:
    """Coerce numpy arrays in [-1, 1], [0, 1], or [0, 255] to a PIL Image.

    PIL inputs are passed through unchanged.
    """
    if isinstance(image, Image.Image):
        return image
    arr = np.asarray(image)
    if arr.dtype.kind == "f":
        if arr.min() < 0:  # assume [-1, 1] (e.g. tangent-space normals)
            arr = (arr + 1) / 2 * 255
        elif arr.max() <= 1.0:  # assume [0, 1]
            arr = arr * 255
        # else assume [0, 255] floats; np.clip + cast below handles it
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    """Convert a binary mask to a uint8 grayscale `Image`."""
    arr = np.asarray(mask)
    if arr.dtype.kind in {"b", "i", "u"} and arr.max() <= 1:
        arr = arr.astype(np.float32)
    return as_pil(arr).convert("L")


def normal_to_image(normal: np.ndarray) -> Image.Image:
    """Convert a [-1, 1] normal map (HxWx3) to an 8-bit RGB image."""
    return as_pil(normal).convert("RGB")


def depth_to_image(depth: np.ndarray) -> Image.Image:
    """Convert a [0, 1] depth map to a grayscale `L` image."""
    return as_pil(depth).convert("L")


def displacement_to_image(
    displacement: np.ndarray, *, normalize: bool = True
) -> Image.Image:
    """Render a displacement map as a uint8 grayscale image.

    If `normalize` is True, rescale values to [0, 1] via min/max; otherwise
    clip to [0, 1].
    """
    arr = np.asarray(displacement, dtype=np.float32)
    if normalize:
        lo = float(arr.min())
        hi = float(arr.max())
        denom = hi - lo
        if denom < 1e-8:
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            arr = (arr - lo) / denom
    else:
        arr = np.clip(arr, 0.0, 1.0)
    return as_pil(arr).convert("L")


# =============================================================================
# Center crop
# =============================================================================


def center_crop(image: Image.Image, size: int = 512) -> Image.Image:
    """Center-crop a PIL image to `size × size`."""
    w, h = image.size
    left, top = _centered_offset(w, size), _centered_offset(h, size)
    return image.crop((left, top, left + size, top + size))


def center_crop_array(arr: np.ndarray, size: int = 512) -> np.ndarray:
    """Center-crop a numpy array (H, W, …) to `size × size`."""
    h, w = arr.shape[:2]
    top, left = _centered_offset(h, size), _centered_offset(w, size)
    return arr[top : top + size, left : left + size]


def _centered_offset(outer: int, inner: int) -> int:
    """Offset that centers an `inner`-sized window inside an `outer` extent."""
    return (outer - inner) // 2


# =============================================================================
# Tiling
# =============================================================================


def tile_image(image: Image.Image, rows: int = 3, cols: int = 3) -> Image.Image:
    """Tile `image` into a `rows × cols` grid."""
    w, h = image.size
    out = Image.new(image.mode, (w * cols, h * rows))
    for r in range(rows):
        for c in range(cols):
            out.paste(image, (c * w, r * h))
    return out


def tiled_seam_mask(
    image: Image.Image,
    n_tiles: int = 3,
    mask_portion: float = 0.1,
) -> Image.Image:
    """Return the seam mask for an `n_tiles × n_tiles` tiling.

    The mask is white (255) inside a band of half-width
    `mask_portion * min(w, h)` around each interior seam, black elsewhere.
    """
    w, h = image.size

    mask = np.zeros((h * n_tiles, w * n_tiles), dtype=np.uint8)
    border = int(min(w, h) * mask_portion)
    for i in range(1, n_tiles):
        mask[:, i * w - border : i * w + border] = 255  # vertical seams
        mask[i * h - border : i * h + border, :] = 255  # horizontal seams
    mask_img = Image.fromarray(mask).convert("RGB")

    return mask_img


# =============================================================================
# Seam-swapping helper functions
# =============================================================================

AXIS_DIRECTION = Literal["horizontal", "vertical"]
SWAP_DIRECTION = Literal["horizontal", "vertical", "both"]


def swap_to_center_seam(image: Image.Image, direction: SWAP_DIRECTION) -> Image.Image:
    """Swap halves so original edges move to the center."""
    if direction == "both":
        image = swap_to_center_seam(image, "horizontal")
        return swap_to_center_seam(image, "vertical")

    first, second = split_image(image, direction)
    return merge_images(second, first, direction)


def split_image(
    image: Image.Image, direction: AXIS_DIRECTION
) -> tuple[Image.Image, Image.Image]:
    """Split `image` in half. The halves (not the split) are along `direction`."""
    w, h = image.size
    if direction == "horizontal":
        mid = w // 2
        return image.crop((0, 0, mid, h)), image.crop((mid, 0, w, h))
    mid = h // 2
    return image.crop((0, 0, w, mid)), image.crop((0, mid, w, h))


def merge_images(
    a: Image.Image, b: Image.Image, direction: AXIS_DIRECTION
) -> Image.Image:
    """Concatenate two images together along `direction`."""
    w, h = a.size
    if direction == "horizontal":
        merged = Image.new("RGB", (w * 2, h))
        offset = (w, 0)
    else:
        merged = Image.new("RGB", (w, h * 2))
        offset = (0, h)

    merged.paste(a, (0, 0))
    merged.paste(b, offset)
    return merged


def center_seam_mask(
    width: int, height: int, line_width: int, direction: SWAP_DIRECTION
) -> Image.Image:
    """Mask with white band(s) painted along the center, each ``line_width`` pixels wide.
    The band(s) divide the image into two halves *along* ``direction``.
    **This means that if** ``direction=horizontal``, **a vertical band will be drawn, and vice versa.**
    """
    mask = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask)
    if direction in ("horizontal", "both"):
        x0 = (width - line_width) // 2
        draw.rectangle([(x0, 0), (x0 + line_width - 1, height - 1)], fill=255)
    if direction in ("vertical", "both"):
        y0 = (height - line_width) // 2
        draw.rectangle([(0, y0), (width - 1, y0 + line_width - 1)], fill=255)
    return mask
