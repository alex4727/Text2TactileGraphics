"""High-pass filtering for tangent-space normal maps."""

from typing import Literal

import numpy as np
from PIL import Image

HighpassMethod = Literal["per_channel", "height_integration"]

# FWHM (full-width half-max) factor for a Gaussian — used to convert a
# "feature size in pixels" into a frequency-domain cutoff.
_GAUSSIAN_FWHM = float(np.sqrt(2 * np.log(2)))


def apply_high_pass_to_normal_map(
    normal_map: Image.Image | np.ndarray,
    freq_threshold: int = 5,
    method: HighpassMethod = "per_channel",
) -> Image.Image:
    """Apply a Gaussian high-pass filter to a normal map.

    Args:
        normal_map: PIL Image or HxWx3 array in [0, 1] or [0, 255], OpenGL
            convention (Y-up, green channel = up).
        freq_threshold: Features smaller than this many pixels are kept.
            Typical values: 5–50 for high filtering, 100–200 for gentle filtering.
        method: ``"per_channel"`` (fast, simple) or ``"height_integration"``
            (slower, geometrically correct).
    """
    normal_rgb = to_float01(normal_map)
    h, w = normal_rgb.shape[:2]

    # Convert freq_threshold (pixels) to a cutoff_ratio in [0.001, 0.5].
    # See module docstring of _highpass_filter_fft for the meaning.
    max_dist = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
    cutoff_ratio = float(
        np.clip(h / (freq_threshold * max_dist * _GAUSSIAN_FWHM), 0.001, 0.5)
    )

    result_rgb = _METHODS[method](normal_rgb, cutoff_ratio)

    image = Image.fromarray((np.clip(result_rgb, 0, 1) * 255).astype(np.uint8))
    return image


# =============================================================================
# Filtering strategies
# =============================================================================


def _hp_per_channel(normal_rgb: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """Per-channel HP filter, then re-normalize."""
    normal = rgb_to_normal(normal_rgb)
    normal_hp = highpass_filter_fft(normal, cutoff_ratio)
    #  HP filtering removes the DC component, including the ~1 mean
    #  of the Z channel of a "flat" normal map. We add (0, 0, 1)
    #  back before re-normalizing.
    normal_hp[..., 2] += 1.0
    normal_hp = normalize_vectors(normal_hp)
    return normal_to_rgb(normal_hp)


def _hp_height_integration(normal_rgb: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """Integrate → HP filter the height → differentiate back to a normal map."""
    normal = rgb_to_normal(normal_rgb)
    height = normal_to_height(normal)
    height_hp = highpass_filter_fft(height, cutoff_ratio)
    normal_hp = height_to_normal(height_hp)
    return normal_to_rgb(normal_hp)


_METHODS = {
    "per_channel": _hp_per_channel,
    "height_integration": _hp_height_integration,
}


# =============================================================================
# Frequency-domain Gaussian high-pass
# =============================================================================


def highpass_filter_fft(data: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    """Per-channel Gaussian high-pass filter on a 2D or 3D array.

    `cutoff_ratio` is in units of the half-diagonal of the image: 0 keeps
    everything, ~0.5 removes almost everything.
    """
    h, w = data.shape[:2]
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    cutoff = cutoff_ratio * np.sqrt(cx**2 + cy**2)
    hp_filter = 1.0 - np.exp(-(dist**2) / (2 * cutoff**2))

    if data.ndim == 3:
        hp_filter = hp_filter[..., None]  # broadcast over channels

    spectrum = np.fft.fftshift(np.fft.fft2(data, axes=(0, 1)), axes=(0, 1))
    filtered = np.fft.ifft2(
        np.fft.ifftshift(spectrum * hp_filter, axes=(0, 1)), axes=(0, 1)
    )
    return np.real(filtered).astype(np.float32)


# =============================================================================
# Normal ↔ height-field conversion (FFT Poisson / FFT gradient)
# =============================================================================


def normal_to_height(normal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Integrate a normal map to a zero-mean height field via FFT Poisson.

    ``normal`` is in [-1, 1] with OpenGL convention (Y-up). The image Y axis
    is flipped relative to OpenGL Y, so dz/dy_image = ny/nz.
    """
    h, w = normal.shape[:2]
    nx, ny = normal[..., 0], normal[..., 1]
    nz = np.maximum(normal[..., 2], eps)  # avoid divide-by-zero
    dzdx = -nx / nz
    dzdy_image = ny / nz

    kx, ky = _freq_grid_2d(h, w)
    div_hat = 1j * kx * np.fft.fft2(dzdx) + 1j * ky * np.fft.fft2(dzdy_image)

    # Poisson solve: H(k) = -Div(k) / (kx² + ky²); DC pinned to 0.
    denom = kx * kx + ky * ky
    denom[0, 0] = 1.0
    h_hat = -div_hat / denom
    h_hat[0, 0] = 0.0
    return np.real(np.fft.ifft2(h_hat)).astype(np.float32)


def height_to_normal(height: np.ndarray) -> np.ndarray:
    """Compute an OpenGL-convention normal map from a height field via FFT gradient."""
    h, w = height.shape
    kx, ky = _freq_grid_2d(h, w)

    h_hat = np.fft.fft2(height)
    dzdx = np.real(np.fft.ifft2(1j * kx * h_hat))
    dzdy_image = np.real(np.fft.ifft2(1j * ky * h_hat))

    # N = normalize(-dz/dx, dz/dy_image, 1) — image-y flip accounts for OpenGL.
    nx_ny_nz = np.stack([-dzdx, dzdy_image, np.ones_like(dzdx)], axis=-1)
    return normalize_vectors(nx_ny_nz).astype(np.float32)


# =============================================================================
# Small helpers
# =============================================================================


def to_float01(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL Image or ndarray (uint8 or float) to a float32 array in [0, 1]."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = image.astype(np.float32)
    return arr / 255.0 if arr.max() > 1.0 else arr


def rgb_to_normal(rgb: np.ndarray) -> np.ndarray:
    """[0, 1] RGB → [-1, 1] tangent normal."""
    return rgb * 2.0 - 1.0


def normal_to_rgb(normal: np.ndarray) -> np.ndarray:
    """[-1, 1] tangent normal → [0, 1] RGB."""
    return (normal + 1.0) / 2.0


def normalize_vectors(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale vectors along the last axis to unit length."""
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (length + eps)


def _freq_grid_2d(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (KX, KY) angular-frequency grids in radians per pixel."""
    kx = 2.0 * np.pi * np.fft.fftfreq(w)
    ky = 2.0 * np.pi * np.fft.fftfreq(h)
    return np.meshgrid(kx, ky)
