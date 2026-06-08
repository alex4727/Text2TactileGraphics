"""Mesh displacement utilities for applying surface details to a 2.5D relief."""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import trimesh
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from tactilegen.generation.utils import tile_image
from tactilegen.geometry.braille import (
    STANDARD_DOT_HEIGHT,
    BraillePlacement,
    create_braille_displacement_map,
    create_standard_braille_displacement_map,
)
from tactilegen.geometry.filtering import normal_to_height, rgb_to_normal

logger = logging.getLogger(__name__)

_UVMode = Literal["planar_xy", "planar_xz", "planar_yz", "existing", "box"]
DisplacementDirection = Literal["normal", "z"]
NormalFormat = Literal["directx", "opengl"]

_PLANAR_AXES: dict[str, int] = {
    "planar_yz": 0,
    "planar_xz": 1,
    "planar_xy": 2,
}


@dataclass
class TexturedSegment:
    """Represents a segment of a 2.5D relief with a given texture."""

    mask: np.ndarray
    tileable_patch: Image.Image  # Texture represented by normal patch
    displacement_scale: float | None = None
    displacement_direction: DisplacementDirection | None = None
    tile_repeat: int = 3
    enabled: bool = True


def apply_segment_displacements(
    mesh: trimesh.Trimesh,
    segments: list[TexturedSegment],
    scale: float = 0.005,
    direction: DisplacementDirection = "normal",
    target_resolution: int = 512,
    normal_format: NormalFormat = "opengl",
) -> trimesh.Trimesh:
    """Apply displacements from multiple textured segments to a mesh.

    Args:
        mesh: Base mesh.
        segments: Textured segments to apply to the mesh.
        scale: Displacement scale in meters; higher = greater displacement. Can be overridden per-segment.
        direction: Displacement direction; "normal" displaces along vertex normals, "z" along the
            Z-axis. Can be overridden per-segment.
        target_resolution: Tiled samples will be downsampled to this resolution.
        normal_format: "opengl" (green up) or "directx" (green down).
    """
    result_mesh = mesh.copy()

    enabled_segments = [s for s in segments if s.enabled]
    logger.info("%d enabled segments to apply", len(enabled_segments))

    for i, seg in enumerate(enabled_segments, start=1):
        tile_repeat = seg.tile_repeat or 3
        tiled_texture = tile_image(
            seg.tileable_patch, rows=tile_repeat, cols=tile_repeat
        )
        if tiled_texture.size != (target_resolution, target_resolution):
            tiled_texture = tiled_texture.resize(
                (target_resolution, target_resolution), Image.Resampling.LANCZOS
            )

        displacement = tileable_patch_to_displacement(
            tiled_texture, normal_format=normal_format
        )
        seg_scale = seg.displacement_scale or scale
        seg_direction = seg.displacement_direction or direction

        result_mesh = apply_displacement_to_mesh(
            mesh=result_mesh,
            displacement_map=displacement,
            scale=seg_scale,
            direction=seg_direction,
            uv_mode="planar_xy",
            relative_scale=False,
            preserve_aspect=True,
            mask=seg.mask,
        )

        logger.info(
            "✓ Applied segment %d/%d: scale=%s, tile=%dx%d, res=%d×%d",
            i,
            len(enabled_segments),
            seg_scale,
            tile_repeat,
            tile_repeat,
            target_resolution,
            target_resolution,
        )

    return result_mesh


def apply_braille_displacements(
    mesh: trimesh.Trimesh,
    placements: list[BraillePlacement],
    image_size: tuple[int, int],
    target_resolution: int = 512,
    dot_height: float = 0.005,
) -> trimesh.Trimesh:
    """Apply braille dot displacements (bounding-box mode) to a mesh.

    Args:
        mesh: Base mesh.
        placements: List of braille placements.
        image_size: (width, height) of the original image the placements
            were authored against.
        target_resolution: Resolution for the intermediate displacement map.
        dot_height: Maximum dot height in meters.
    """
    result_mesh = mesh.copy()
    img_w, img_h = image_size
    scale_x = target_resolution / img_w
    scale_y = target_resolution / img_h

    enabled_placements = [p for p in placements if p.enabled]
    logger.info("%d placements segments to apply", len(enabled_placements))

    for placement in enabled_placements:
        px1 = placement.x * scale_x
        py1 = placement.y * scale_y
        pw = placement.width * scale_x
        ph = placement.height * scale_y

        scaled_placement = BraillePlacement(
            text=placement.text,
            x=px1,
            y=py1,
            width=pw,
            height=ph,
            padding=placement.padding,
        )

        # actual dome height is controlled by `scale` below; this just composites
        # the placement directly onto the full-resolution canvas.
        full_displacement = create_braille_displacement_map(
            [scaled_placement],
            width=target_resolution,
            height=target_resolution,
            dot_height=1.0,
        )

        result_mesh = apply_displacement_to_mesh(
            mesh=result_mesh,
            displacement_map=full_displacement,
            scale=dot_height,
            direction="normal",
            uv_mode="planar_xy",
            relative_scale=False,
            preserve_aspect=True,
        )
        logger.info(
            "    ✓ Applied braille dots at (%d, %d) size %d×%d",
            px1,
            py1,
            pw,
            ph,
        )

    return result_mesh


def apply_standard_braille_displacement(
    mesh: trimesh.Trimesh,
    text: str,
    plate_size: float,
    target_resolution: int = 1024,
    flat_top_ratio: float = 0.3,
    padding: float = 0.005,
    bottom_padding: float = 0.005,
) -> trimesh.Trimesh:
    """Apply standard braille displacement using US/international dimensions.

    Args:
        mesh: Base mesh.
        text: Text to render.
        plate_size: Side length of the square plate, in meters.
        target_resolution: Resolution of the intermediate displacement map.
        flat_top_ratio: Fraction of dot radius rendered as a flat plateau.
        padding: Side padding in meters.
        bottom_padding: Padding from the bottom edge of the plate, in meters.
    """
    displacement_map = create_standard_braille_displacement_map(
        text=text,
        plate_size=plate_size,
        resolution=target_resolution,
        padding=padding,
        bottom_padding=bottom_padding,
        flat_top_ratio=flat_top_ratio,
    )

    result = apply_displacement_to_mesh(
        mesh=mesh,
        displacement_map=displacement_map,
        scale=STANDARD_DOT_HEIGHT,
        direction="normal",
        uv_mode="planar_xy",
        relative_scale=False,
        preserve_aspect=True,
    )
    height_mm = STANDARD_DOT_HEIGHT * 1000
    logger.info("✓ Applied standard braille displacement (height = %.2fmm)", height_mm)
    return result


# =============================================================================
# Core displacement
# =============================================================================


def apply_displacement_to_mesh(
    mesh: trimesh.Trimesh,
    displacement_map: np.ndarray,
    scale: float = 0.01,
    direction: DisplacementDirection = "normal",
    uv_mode: _UVMode = "planar_xy",
    relative_scale: bool = False,
    preserve_aspect: bool = True,
    tiling: tuple[float, float] = (1.0, 1.0),
    mask: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Apply a displacement/height map to a mesh by deforming vertices.

    IMPORTANT: Mesh resolution (number of vertices) should be similar to the
    displacement map resolution for smooth results. For a 3072x3072 map, use
    depth_resolution=3072 when creating the base mesh.

    Args:
        mesh: Base mesh.
        displacement_map: H x W height map in [0, 1].
        scale: Displacement strength. In meters when `relative_scale=False`;
            relative to the mesh bounding-box diagonal otherwise.
        direction: "normal" displaces along vertex normals, "z" along the
            Z-axis.
        uv_mode: UV projection. "planar_xy" is the default and is what relief
            meshes want. "existing" falls back to "planar_xy" if the mesh has
            no UVs.
        relative_scale: Treat `scale` as a fraction of bbox diagonal.
        preserve_aspect: Keep the displacement-map aspect ratio in UV space.
        tiling: (u_repeat, v_repeat) — number of texture repeats.
        mask: Optional H x W boolean mask; vertices outside the mask are not
            displaced. For `direction="normal"`, displacement is additionally
            clamped so vertices never move past the mask boundary in x-y (see
            `_clamp_normal_disp_to_mask`).
    """
    vertices = mesh.vertices.copy()

    u, v = _compute_uv(mesh, vertices, uv_mode, preserve_aspect)
    # Tile + wrap to [0, 1] for seamless sampling.
    u = (u * tiling[0]) % 1.0
    v = (v * tiling[1]) % 1.0

    displacement_values = _normalize_01(_sample_at_uv(displacement_map, u, v))

    if mask is not None:
        # Hard threshold at 0.5 after interpolation.
        vertex_inside = _sample_at_uv(mask.astype(np.float32), u, v) > 0.5
        displacement_values = displacement_values * vertex_inside

    if relative_scale:
        bbox_diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        effective_scale = scale * bbox_diagonal
    else:
        effective_scale = scale

    if direction == "normal":
        disp_vec = mesh.vertex_normals * (
            displacement_values[:, np.newaxis] * effective_scale
        )
        if mask is not None:
            # Keep displaced vertices within the segment's x-y footprint: the
            # normal's lateral component would otherwise push boundary vertices
            # outside the mask.
            disp_vec = _clamp_normal_disp_to_mask(
                vertices, disp_vec, uv_mode, preserve_aspect, mask, tiling
            )
        vertices += disp_vec
    elif direction == "z":
        vertices[:, 2] += displacement_values * effective_scale
    else:
        raise ValueError(f"Unknown direction: {direction}. Use 'normal' or 'z'.")

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy())


# =============================================================================
# Tileable patch to displacement
# =============================================================================


def tileable_patch_to_displacement(
    patch: Image.Image, normal_format: NormalFormat = "opengl"
) -> np.ndarray:
    """Convert a tileable patch image to an HxW [0, 1] displacement map.

    RGB/RGBA inputs are treated as tangent-space normal maps and converted
    via Poisson integration. Grayscale inputs are normalized directly.
    """
    if patch.mode in ("RGB", "RGBA"):
        rgb = patch.convert("RGB") if patch.mode == "RGBA" else patch
        normal_rgb = np.array(rgb, dtype=np.float32)
        normal_rgb /= 255.0
        displacement = _integrate_normal_to_displacement(
            rgb_to_normal(normal_rgb),
            normal_format=normal_format,
            strength=1.0,
            eps=1e-4,
            remove_linear_trend=True,
            smooth_sigma=0.8,
        )
        logger.info(
            "Converted normal map to displacement via Poisson integration (%s format)",
            normal_format,
        )
        return displacement

    gray = patch if patch.mode == "L" else patch.convert("L")
    logger.info("Using grayscale image directly as displacement")
    displacement = np.array(gray, dtype=np.float32)
    displacement /= 255.0
    return displacement


def _integrate_normal_to_displacement(
    normal: np.ndarray,
    *,
    normal_format: NormalFormat = "opengl",
    strength: float = 1.0,
    eps: float = 1e-6,
    remove_linear_trend: bool = True,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Integrate a [-1, 1] normal map to a [0, 1] displacement map.

    Args:
        normal: HxWx3 float array in [0,1].
        normal_format: "opengl" (green up) or "directx" (green down).
        strength: Multiplier on the integrated height before normalization.
        eps: Floor for Nz to avoid divide-by-zero on near-tangent normals.
        remove_linear_trend: Subtract the best-fit plane (removes global tilt).
        smooth_sigma: Optional Gaussian blur σ applied to the X/Y normal
            channels before integration.
    """
    H, W, _ = normal.shape

    if normal_format not in ("opengl", "directx"):
        raise ValueError(
            f"Unknown normal format: {normal_format}. Use 'opengl' or 'directx'."
        )

    if normal_format == "directx":
        normal[..., 1] = -normal[..., 1]

    if smooth_sigma > 0.0:
        normal[..., 0] = gaussian_filter(normal[..., 0], smooth_sigma, mode="wrap")
        normal[..., 1] = gaussian_filter(normal[..., 1], smooth_sigma, mode="wrap")

    h = normal_to_height(normal, eps=eps)

    if remove_linear_trend:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        A = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W, dtype=np.float32)], axis=1)
        a, b, c = np.linalg.lstsq(A, h.ravel(), rcond=None)[0]
        h = h - (a * xx + b * yy + c)

    return _percentile_normalize(h * strength)


# =============================================================================
# UV projection helpers
# =============================================================================


def _planar_uv_transform(
    vertices: np.ndarray, axis: int, preserve_aspect: bool
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Affine map from in-plane vertex coords to UV, for `_planar_projection`.

    Returns `(perp_axes, mins, scale)` such that for the two non-projection axes
    `perp_axes`, `uv = (vertices[:, perp_axes] - mins) / scale`. Exposing the
    transform lets callers map *moved* positions through the same mapping the
    original vertices used (needed for boundary-aware displacement).
    """
    # The two non-projection axes, in order.
    perp_axes = [i for i in range(3) if i != axis]
    coords = vertices[:, perp_axes]

    mins = coords.min(axis=0)
    ranges = coords.max(axis=0) - mins
    ranges[ranges == 0] = 1.0  # avoid divide-by-zero on collapsed axes

    if preserve_aspect:
        scale = np.array([ranges.max(), ranges.max()])
    else:
        scale = ranges
    return perp_axes, mins, scale


def _planar_projection(
    vertices: np.ndarray, axis: int = 2, preserve_aspect: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Project vertices onto the plane perpendicular to `axis` (0=X, 1=Y, 2=Z).

    Returns (u, v) in [0, 1].
    """
    perp_axes, mins, scale = _planar_uv_transform(vertices, axis, preserve_aspect)
    coords = vertices[:, perp_axes]
    return (coords[:, 0] - mins[0]) / scale[0], (coords[:, 1] - mins[1]) / scale[1]


def _box_projection(
    vertices: np.ndarray, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Box projection: pick a planar projection per vertex based on its normal."""
    dominant_axis = np.abs(normals).argmax(axis=1)
    u = np.zeros(len(vertices))
    v = np.zeros(len(vertices))
    for axis in range(3):
        mask = dominant_axis == axis
        if mask.any():
            u[mask], v[mask] = _planar_projection(
                vertices[mask], axis=axis, preserve_aspect=True
            )
    return u, v


def _compute_uv(
    mesh: trimesh.Trimesh, vertices: np.ndarray, uv_mode: _UVMode, preserve_aspect: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve UV coordinates from the chosen `uv_mode`."""
    if uv_mode == "existing":
        existing = getattr(getattr(mesh, "visual", None), "uv", None)
        if existing is not None:
            return existing[:, 0].copy(), existing[:, 1].copy()
        # Fall through to planar_xy if existing UVs are missing.
        uv_mode = "planar_xy"

    if uv_mode in _PLANAR_AXES:
        return _planar_projection(
            vertices, axis=_PLANAR_AXES[uv_mode], preserve_aspect=preserve_aspect
        )

    if uv_mode == "box":
        return _box_projection(mesh.vertices, mesh.vertex_normals)

    raise ValueError(f"Unknown uv_mode: {uv_mode}")


def _clamp_normal_disp_to_mask(
    vertices: np.ndarray,
    disp_vec: np.ndarray,
    uv_mode: _UVMode,
    preserve_aspect: bool,
    mask: np.ndarray,
    tiling: tuple[float, float],
    iterations: int = 16,
) -> np.ndarray:
    """Cap each vertex's normal-direction displacement at the mask boundary.

    Normal-direction displacement moves a vertex along its surface normal, which
    for relief edges has an in-plane (lateral) component. That pushes
    near-boundary vertices' (x, y) outside the segment mask. Here each vertex
    rides up its normal ray only as far as the fraction where its in-plane
    position would still sample inside the mask — so geometry stays within the
    object's x-y footprint. The out-of-plane ("up") motion is limited by the
    same fraction, i.e. the vertex stops at the mask wall.

    Only planar UV modes have an affine x-y -> UV map; other modes are returned
    unchanged.
    """
    axis = _PLANAR_AXES.get(uv_mode)
    if axis is None:
        return disp_vec

    perp_axes, mins, scale = _planar_uv_transform(vertices, axis, preserve_aspect)
    base = vertices[:, perp_axes]  # (N, 2) original in-plane coords
    delta = disp_vec[:, perp_axes]  # (N, 2) in-plane part of the displacement

    h, w = mask.shape
    mask_spline = RectBivariateSpline(
        np.linspace(0, 1, h), np.linspace(0, 1, w), mask.astype(np.float32)
    )

    def inside(frac: np.ndarray) -> np.ndarray:
        xy = base + frac[:, np.newaxis] * delta
        u = (xy[:, 0] - mins[0]) / scale[0]
        v = (xy[:, 1] - mins[1]) / scale[1]
        in_range = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        uu = (u * tiling[0]) % 1.0
        vv = (v * tiling[1]) % 1.0
        sampled = mask_spline(1.0 - vv, uu, grid=False) > 0.5
        return sampled & in_range

    full = np.ones(len(vertices))
    needs_clamp = ~inside(full)
    # Binary search the largest fraction that keeps the vertex inside the mask.
    # `lo` always stays inside (frac 0 = the vertex's original, in-mask spot),
    # `hi` is the first known-outside fraction.
    lo = np.zeros(len(vertices))
    hi = full.copy()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        mid_inside = inside(mid)
        lo = np.where(needs_clamp & mid_inside, mid, lo)
        hi = np.where(needs_clamp & ~mid_inside, mid, hi)

    frac = np.where(needs_clamp, lo, full)
    return disp_vec * frac[:, np.newaxis]


def _sample_at_uv(arr: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample a 2D array at fractional UV coordinates with cubic B-splines.

    Image y is flipped (row 0 is at the top, v=0 is at the bottom).
    """
    h, w = arr.shape
    spline = RectBivariateSpline(np.linspace(0, 1, h), np.linspace(0, 1, w), arr)
    return spline(1.0 - v, u, grid=False)


# =============================================================================
# Small numpy helpers
# =============================================================================


def _normalize_01(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Linearly rescale `arr` so its min/max land at 0/1."""
    return (arr - arr.min()) / (arr.max() - arr.min() + eps)


def _percentile_normalize(
    arr: np.ndarray, low: float = 1, high: float = 99
) -> np.ndarray:
    """Clip `arr` to its [low, high] percentile range and rescale to [0, 1]."""
    lo, hi = np.percentile(arr, [low, high])
    return (np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-8)
