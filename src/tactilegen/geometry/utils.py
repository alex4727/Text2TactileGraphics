import logging
from collections import Counter

import numpy as np
import pyvista as pv
import trimesh
from PIL import Image
from scipy.interpolate import RectBivariateSpline

pv.OFF_SCREEN = True

logger = logging.getLogger(__name__)


# =============================================================================
# Depth to mesh
# =============================================================================


def pv2trimesh(surface: pv.PolyData) -> trimesh.Trimesh:
    """Convert a PyVista PolyData surface to a `trimesh.Trimesh`."""
    surface = surface.triangulate()
    return trimesh.Trimesh(
        vertices=surface.points,
        faces=surface.faces.reshape(-1, 4)[:, 1:],
    )


def depth2mesh(
    depth: np.ndarray,
    *,
    width: float = 0.25,
    max_depth: float = 0.05,
    smooth: bool = True,
    decimate: bool = True,
) -> pv.PolyData:
    """Convert an H×W depth map into a 3D PyVista PolyData mesh.

    Args:
        depth: H × W depth map.
        width: Physical width of the output mesh, in meters.
        max_depth: Maximum depth in meters.
        smooth: If True, apply 100 iterations of mesh smoothing.
        decimate: If True, decimate the triangulated mesh by 50%.
    """
    depth: np.ndarray = max_depth * (depth - depth.min()) / (depth.max() - depth.min())
    full_mask = np.ones(depth.shape[:2], bool)
    vertices = _depth_to_point_cloud(
        -depth, full_mask, step_size=width / depth.shape[1]
    )
    facets = _construct_facets_from(full_mask)
    surface = pv.PolyData(vertices, facets)

    # These operations are faster with pyvista than trimesh
    if smooth:
        surface = surface.smooth(n_iter=100)
    if decimate:
        surface = surface.triangulate().decimate_pro(0.5)
    return surface


def _depth_to_point_cloud(
    depth_map: np.ndarray, mask: np.ndarray, step_size: float = 1.0
) -> np.ndarray:
    """Lift a masked depth map to an N×3 point cloud (orthographic projection).

    Coordinate system::

        y
        |  z
        | /
        |/
        o ---x
    """
    h, w = mask.shape
    yy, xx = np.meshgrid(range(w), range(h))
    xx = np.flip(xx, axis=0)  # flip image-Y to world-Y

    vertices = np.zeros((h, w, 3))
    vertices[..., 0] = yy * step_size
    vertices[..., 1] = xx * step_size
    vertices[..., 2] = -depth_map
    return vertices[mask]


def _construct_facets_from(mask: np.ndarray) -> np.ndarray:
    """Construct PyVista quad facets covering every 2×2 patch of `mask` that's all-True."""
    idx = np.zeros_like(mask, dtype=int)
    idx[mask] = np.arange(np.sum(mask))

    # A quad's top-left corner is valid iff itself, the cell above it,
    # the cell to its left, and the cell above-and-left are all in the mask.
    top_left = np.logical_and.reduce(
        (move_top(mask), move_left(mask), move_top_left(mask), mask)
    )
    top_right = move_right(top_left)
    bottom_left = move_bottom(top_left)
    bottom_right = move_bottom_right(top_left)

    n_quads = np.sum(top_left)
    return np.stack(
        [
            4 * np.ones(n_quads),
            idx[top_left],
            idx[bottom_left],
            idx[bottom_right],
            idx[top_right],
        ],
        axis=-1,
    ).astype(int)


# =============================================================================
# Mask shift helpers
# =============================================================================
# All eight functions are thin wrappers around `_shifted_mask`. They keep their
# individual names because tests and callers reference them directly.


def _shifted_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return `mask` shifted by (dy, dx), with new pixels filled with zeros."""
    pad = ((max(-dy, 0), max(dy, 0)), (max(-dx, 0), max(dx, 0)))
    padded = np.pad(mask, pad, "constant", constant_values=0)
    y0 = max(dy, 0)
    x0 = max(dx, 0)
    return padded[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]


def move_left(mask: np.ndarray) -> np.ndarray:
    """Shift `mask` left by 1, fill right edge with zeros."""
    return _shifted_mask(mask, 0, 1)


def move_right(mask: np.ndarray) -> np.ndarray:
    """Shift `mask` right by 1, fill left edge with zeros."""
    return _shifted_mask(mask, 0, -1)


def move_top(mask: np.ndarray) -> np.ndarray:
    """Shift `mask` up by 1, fill bottom edge with zeros."""
    return _shifted_mask(mask, 1, 0)


def move_bottom(mask: np.ndarray) -> np.ndarray:
    """Shift `mask` down by 1, fill top edge with zeros."""
    return _shifted_mask(mask, -1, 0)


def move_top_left(mask: np.ndarray) -> np.ndarray:
    """Shift `mask` up-left by 1."""
    return _shifted_mask(mask, 1, 1)


def move_top_right(mask: np.ndarray) -> np.ndarray:
    """Shift `mask` up-right by 1."""
    return _shifted_mask(mask, 1, -1)


def move_bottom_left(mask: np.ndarray) -> np.ndarray:
    """Shift `mask` down-left by 1."""
    return _shifted_mask(mask, -1, 1)


def move_bottom_right(mask: np.ndarray) -> np.ndarray:
    """Shift `mask` down-right by 1."""
    return _shifted_mask(mask, -1, -1)


# =============================================================================
# Plate flattening + mesh closing
# =============================================================================


# Each rectangle corner is a (curr_edge, next_edge, top_corner_idx) tuple.
# `_stitch_boundary_to_corners` uses these to fill the gap between the
# meshed boundary and the rectangular frame.
_CORNER_GAPS: tuple[tuple[str, str, int], ...] = (
    ("left", "bottom", 0),  # bottom-left
    ("bottom", "right", 1),  # bottom-right
    ("right", "top", 2),  # top-right
    ("top", "left", 3),  # top-left
)


def flatten_and_close_plate(
    mesh: trimesh.Trimesh,
    plate_mask: np.ndarray,
    *,
    plate_thickness: float = 0.002,
    foreground_z_threshold: float = 0.002,
) -> trimesh.Trimesh:
    """Flatten the plate region to its mean z, then close the mesh with walls + a bottom.

    Steps:
        1. Identify vertices inside the plate-mask region (UV-mapped).
        2. Average their z to get the plate's nominal height.
        3. Flatten every non-foreground vertex (plate + edges outside the
           plate bbox) to that height.
        4. Build a rectangular frame at the plate's bounding box: 8 corner
           vertices (4 top, 4 bottom), 8 wall triangles, 2 bottom-face
           triangles, and corner-stitch triangles connecting the meshed
           boundary to the frame.
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    x_coords, y_coords, z_coords = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    max_range = max(x_max - x_min, y_max - y_min)

    # ---------- UV mapping + mask sampling ----------
    u = (x_coords - x_min) / max_range
    v = (y_coords - y_min) / max_range
    is_plate_vertex = _sample_mask_at_vertices(plate_mask, u, v) > 0.5

    if not is_plate_vertex.any():
        logger.warning("No vertices found in plate mask region", stacklevel=2)
        return mesh

    # ---------- Foreground bbox from the plate mask ----------
    plate_rows, plate_cols = np.where(plate_mask)
    fg_margin_px = 20
    plate_h, plate_w = plate_mask.shape
    py_min = plate_rows.min() + fg_margin_px
    py_max = plate_rows.max() - fg_margin_px
    px_min = plate_cols.min() + fg_margin_px
    px_max = plate_cols.max() - fg_margin_px

    is_inside_fg_bbox = (
        (u > px_min / plate_w)
        & (u < px_max / plate_w)
        & (v > (plate_h - py_max) / plate_h)
        & (v < (plate_h - py_min) / plate_h)
    )

    # ---------- Flatten everything that's not foreground relief ----------
    plate_z = z_coords[is_plate_vertex]
    avg_plate_z = float(plate_z.mean())
    num_plate_vertices = int(is_plate_vertex.sum())

    logger.info("  Plate region: %d vertices", num_plate_vertices)
    logger.info("  Inside FG bbox: %d vertices", int(is_inside_fg_bbox.sum()))
    logger.info("  Plate z range: [%.6f, %.6f]", plate_z.min(), plate_z.max())
    logger.info("  Average plate z: %.6f", avg_plate_z)
    logger.info("  Full mesh z range: [%.6f, %.6f]", z_coords.min(), z_coords.max())

    is_foreground = (
        (z_coords > avg_plate_z + foreground_z_threshold)
        & is_inside_fg_bbox
        & ~is_plate_vertex
    )
    num_foreground = int(is_foreground.sum())
    logger.info(
        "  Foreground (z > %.6f, inside bbox): %d",
        avg_plate_z + foreground_z_threshold,
        num_foreground,
    )

    flatten = ~is_foreground
    vertices[flatten, 2] = avg_plate_z
    logger.info("  Flattened %d vertices to z=%.6f", int(flatten.sum()), avg_plate_z)

    # ---------- Build the closed rectangular shell ----------
    bottom_z = avg_plate_z - plate_thickness
    temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    boundary_sorted = _find_sorted_boundary(temp_mesh)
    if boundary_sorted is None:
        logger.warning(
            "No boundary edges found, mesh may already be closed", stacklevel=2
        )
        return temp_mesh

    # Ensure all boundary vertices sit exactly at plate height — otherwise the
    # walls won't connect cleanly to the meshed boundary.
    vertices[boundary_sorted, 2] = avg_plate_z

    # Build corner vertices (CCW: bottom-left → bottom-right → top-right → top-left).
    top_corners = np.array(
        [
            [x_min, y_min, avg_plate_z],
            [x_max, y_min, avg_plate_z],
            [x_max, y_max, avg_plate_z],
            [x_min, y_max, avg_plate_z],
        ]
    )
    bottom_corners = top_corners.copy()
    bottom_corners[:, 2] = bottom_z

    base = len(vertices)
    all_vertices = np.vstack([vertices, top_corners, bottom_corners])
    top_corner_idx = np.arange(base, base + 4)
    bottom_corner_idx = np.arange(base + 4, base + 8)

    # Wall + bottom faces.
    new_faces = _make_wall_faces(top_corner_idx, bottom_corner_idx)
    new_faces.extend(
        [
            [bottom_corner_idx[0], bottom_corner_idx[2], bottom_corner_idx[1]],
            [bottom_corner_idx[0], bottom_corner_idx[3], bottom_corner_idx[2]],
        ]
    )
    new_faces.extend(
        _stitch_boundary_to_corners(
            all_vertices,
            boundary_sorted,
            top_corner_idx,
            (x_min, x_max, y_min, y_max),
            tol=max_range * 0.02,
        )
    )

    all_faces = np.vstack([faces, np.array(new_faces)])
    closed_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
    closed_mesh.update_faces(closed_mesh.nondegenerate_faces())
    closed_mesh.update_faces(closed_mesh.unique_faces())
    closed_mesh.remove_unreferenced_vertices()

    logger.info(
        "  Closed mesh: %d vertices, %d faces",
        len(closed_mesh.vertices),
        len(closed_mesh.faces),
    )
    logger.info("  Bottom at z=%.6f", bottom_z)

    return closed_mesh


def _sample_mask_at_vertices(
    mask: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Bilinearly sample a 2D mask at UV coordinates (mask Y is image-flipped)."""
    flipped = np.flipud(mask.astype(np.float32))
    h, w = flipped.shape
    spline = RectBivariateSpline(np.linspace(0, 1, h), np.linspace(0, 1, w), flipped)
    return spline(v, u, grid=False)


def _make_wall_faces(top_idx: np.ndarray, bottom_idx: np.ndarray) -> list[list[int]]:
    """Two triangles per wall, CCW around a rectangular frame."""
    faces: list[list[int]] = []
    for i in range(4):
        j = (i + 1) % 4
        faces.append([top_idx[i], bottom_idx[i], bottom_idx[j]])
        faces.append([top_idx[i], bottom_idx[j], top_idx[j]])
    return faces


def _stitch_boundary_to_corners(
    vertices: np.ndarray,
    boundary_sorted: np.ndarray,
    top_corner_idx: np.ndarray,
    rect_bounds: tuple[float, float, float, float],
    tol: float,
) -> list[list[int]]:
    """Fill gaps between the meshed boundary and the rectangular frame.

    For each consecutive pair of boundary vertices that straddle one of the
    four rectangle corners, emit a triangle to that corner.
    """
    x_min, x_max, y_min, y_max = rect_bounds
    n = len(boundary_sorted)

    def on_edge(v: np.ndarray) -> dict[str, bool]:
        return {
            "left": abs(v[0] - x_min) < tol,
            "right": abs(v[0] - x_max) < tol,
            "bottom": abs(v[1] - y_min) < tol,
            "top": abs(v[1] - y_max) < tol,
        }

    faces: list[list[int]] = []
    for i in range(n):
        curr_idx = boundary_sorted[i]
        next_idx = boundary_sorted[(i + 1) % n]
        curr_on = on_edge(vertices[curr_idx])
        next_on = on_edge(vertices[next_idx])

        for edge_a, edge_b, corner_top_i in _CORNER_GAPS:
            spans_corner = (curr_on[edge_a] and next_on[edge_b]) or (
                curr_on[edge_b] and next_on[edge_a]
            )
            if spans_corner:
                faces.append([curr_idx, top_corner_idx[corner_top_i], next_idx])
    return faces


def _find_sorted_boundary(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Sort the boundary vertices CCW around the centroid; None if no boundary."""
    edges = mesh.edges_sorted
    edge_counts = Counter(tuple(e) for e in edges)
    boundary_edges = [e for e, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return None

    boundary_indices = np.array(sorted({v for edge in boundary_edges for v in edge}))
    boundary_verts = mesh.vertices[boundary_indices]
    centre = mesh.vertices[boundary_indices].mean(axis=0)
    angles = np.arctan2(
        boundary_verts[:, 1] - centre[1], boundary_verts[:, 0] - centre[0]
    )
    return boundary_indices[np.argsort(angles)]


# =============================================================================
# Visualization
# =============================================================================


def render_mesh_to_image(
    mesh: pv.PolyData,
    size: tuple[int, int] = (512, 512),
    camera_position: str = "iso",
    background: str = "white",
) -> Image.Image:
    """Render a PyVista PolyData mesh off-screen and return a PIL Image."""
    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    plotter.set_background(background)
    plotter.add_mesh(
        mesh, color="tan", smooth_shading=True, show_edges=False, lighting=True
    )

    if camera_position == "iso":
        plotter.camera_position = "iso"
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 30
    else:
        plotter.camera_position = camera_position

    plotter.reset_camera()
    try:
        return Image.fromarray(plotter.screenshot(return_img=True))  # type:ignore
    finally:
        plotter.close()
