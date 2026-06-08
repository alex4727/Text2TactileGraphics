import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pybraille
from PIL import Image, ImageDraw

# PIL fill color: int (mode "L"), RGB(A) tuple, or named-color string.
_PILColor = int | tuple[int, ...] | str
_Point = tuple[float, float]


@dataclass
class BraillePlacement:
    text: str
    x: float  # Top-left x
    y: float  # Top-left y
    width: float
    height: float
    padding: float = 0.0  # Buffer between braille and box, as a fraction of the box
    enabled: bool = True

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.width, self.y + self.height


# =============================================================================
# Standard Braille Dimensions (in meters)
# =============================================================================
# Based on US Library of Congress and international standards. Slightly
# inflated by DOT_SCALE to give 3D-printing tolerances.
DOT_SCALE = 1.2
STANDARD_DOT_HEIGHT = 0.0005 * 1.5  # nominal 0.5 mm, raised a touch for printing
STANDARD_DOT_DIAMETER = 0.0015 * DOT_SCALE  # 1.5 mm
STANDARD_DOT_RADIUS = STANDARD_DOT_DIAMETER / 2  # 0.75 mm
STANDARD_INTRA_CELL_SPACING = 0.0025 * DOT_SCALE  # 2.5 mm, dot-to-dot inside a cell
STANDARD_CELL_SPACING = 0.006 * DOT_SCALE  # 6 mm, cell-to-cell
STANDARD_LINE_SPACING = 0.01 * DOT_SCALE  # 10 mm, line-to-line


# =============================================================================
# Braille dot layout
# =============================================================================
# Unicode braille range. Bit i of (codepoint - _BRAILLE_BASE) indicates that
# dot number i+1 is raised.
_BRAILLE_BASE = 0x2800
_BRAILLE_END = 0x28FF

# Standard 6-dot braille layout. The cell is 2 columns × 3 rows of dots,
# numbered:  1 4
#            2 5
#            3 6
# Map each dot number to (column, row_from_top), both in 0-indexed dot units.
# Dots 7 and 8 (8-dot Braille extensions) are intentionally omitted — the
# renderers in this module only draw the standard 6-dot layout.
_DOT_POSITIONS: dict[int, tuple[int, int]] = {
    1: (0, 0),
    2: (0, 1),
    3: (0, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
}


def _text_to_dot_positions(text: str) -> list[list[tuple[int, int]]]:
    """Translate text into a sequence of braille cells (lists of dot positions)."""
    return [_char_to_dot_positions(ch) for ch in pybraille.convertText(text)]


def _char_to_dot_positions(ch: str) -> list[tuple[int, int]]:
    """Decode a single Unicode braille character into its dot positions."""
    if len(ch) != 1:
        raise ValueError("Please supply a single Unicode character")
    code = ord(ch)
    if not (_BRAILLE_BASE <= code <= _BRAILLE_END):
        return []
    bits = code - _BRAILLE_BASE
    dot_nums = [i + 1 for i in range(8) if bits & (1 << i)]
    return [_DOT_POSITIONS[n] for n in dot_nums if n in _DOT_POSITIONS]


# =============================================================================
# BrailleGeometry — layout engine
# =============================================================================


class BrailleGeometry:
    """Geometric layout of braille text as a grid of dot positions."""

    def __init__(self, text: str | Iterable[str]):
        if isinstance(text, str):
            lines = text.split("\n") if text else []
        else:
            lines = list(text)
        self._lines: list[list[list[tuple[int, int]]]] = [
            _text_to_dot_positions(line) for line in lines
        ]

    # ---- inspection --------------------------------------------------------

    @property
    def n_lines(self) -> int:
        return len(self._lines)

    @property
    def max_cells_per_line(self) -> int:
        return max((len(line) for line in self._lines), default=0)

    # ---- dimensions --------------------------------------------------------

    def standard_width(self) -> float:
        """Width of the tight dot-bounding-box (extent from outer dot edge to outer dot edge)."""
        n_cells = self.max_cells_per_line
        if n_cells == 0:
            return 0.0
        return (
            (n_cells - 1) * STANDARD_CELL_SPACING
            + STANDARD_INTRA_CELL_SPACING
            + STANDARD_DOT_DIAMETER
        )

    def standard_height(self) -> float:
        """Height of the tight dot-bounding-box."""
        n_lines = self.n_lines
        if n_lines == 0:
            return 0.0
        return (
            (n_lines - 1) * STANDARD_LINE_SPACING
            + 2 * STANDARD_INTRA_CELL_SPACING
            + STANDARD_DOT_DIAMETER
        )

    # ---- positioning -------------------------------------------------------

    def position_braille_in_box(
        self,
        corner1: _Point,
        corner2: _Point,
        *,
        padding: float = 0.0,
    ) -> tuple[list[_Point], float]:
        """
        Fit the braille layout inside the axis-aligned box ``[corner1, corner2]``,
        with optional padding (specified as a percentage of width and height).
        """
        if not 0.0 <= padding < 0.5:
            raise ValueError(f"padding must be in [0, 0.5); got {padding}")

        x1, y1 = corner1
        x2, y2 = corner2
        box_w = abs(x2 - x1) * (1 - 2 * padding)
        box_h = abs(y2 - y1) * (1 - 2 * padding)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        w = self.standard_width()
        h = self.standard_height()
        if w == 0.0 or h == 0.0:
            return [], STANDARD_DOT_RADIUS

        scale = min(box_w / w, box_h / h)
        scaled_w, scaled_h = w * scale, h * scale
        x0 = cx - scaled_w / 2
        y0 = cy - scaled_h / 2
        points = [
            (x0 + px * scale, y0 + py * scale)
            for px, py in self._dot_centers_from_topleft()
        ]
        return points, STANDARD_DOT_RADIUS * scale

    def _dot_centers_from_topleft(self) -> list[_Point]:
        """Dot centers with the bbox's top-left at (0, 0) (y-down)."""
        points: list[_Point] = []
        for line_idx, cells in enumerate(self._lines):
            for cell_idx, dots in enumerate(cells):
                base_x = cell_idx * STANDARD_CELL_SPACING + STANDARD_DOT_RADIUS
                base_y = line_idx * STANDARD_LINE_SPACING + STANDARD_DOT_RADIUS
                for col, row in dots:
                    points.append(
                        (
                            base_x + col * STANDARD_INTRA_CELL_SPACING,
                            base_y + row * STANDARD_INTRA_CELL_SPACING,
                        )
                    )
        return points


# =============================================================================
# 2D preview rendering
# =============================================================================


def render_braille_on_image(
    image: Image.Image,
    placements: Iterable[BraillePlacement],
    *,
    dot_color: _PILColor = (50, 50, 50),
    draw_bbox: bool = False,
    bbox_color: _PILColor = (100, 100, 255),
) -> Image.Image:
    """Render braille placements on top of a 2D image.

    If ``draw_bbox=True``, outline each placement with a blue rectangle.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    if draw_bbox:
        enabled_placements = [p for p in placements if p.enabled]
        for placement in enabled_placements:
            draw.rectangle(list(placement.bbox), outline=bbox_color, width=2)

    for p, r in _get_placement_dots(placements):
        _draw_dot(draw, p, r, dot_color)

    return img


def render_braille(
    text: str,
    *,
    width: int = 256,
    height: int = 64,
    dot_color: tuple = (50, 50, 50),
    bg_color: tuple = (240, 240, 240),
    padding: float = 0.1,
) -> Image.Image:
    """Render braille text onto a blank background."""
    background = Image.new("RGB", (width, height), bg_color)
    placement = BraillePlacement(
        text=text,
        x=0,
        y=0,
        width=width,
        height=height,
        padding=padding,
    )
    return render_braille_on_image(
        background,
        [placement],
        dot_color=dot_color,
        draw_bbox=False,
    )


def render_standard_braille_on_image(
    image: Image.Image,
    text: str,
    *,
    dot_color: tuple = (50, 50, 50),
    plate_size: float = 0.12,
    padding: float = 0.005,
    bottom_padding: float = 0.005,
) -> Image.Image:
    """Render standard-dimensioned braille on top of ``image``.

    2D analogue of ``render_standard_displacement_map``.
    """
    placement = _make_standard_placement(
        text,
        plate_size=plate_size,
        resolution=image.width,
        padding=padding,
        bottom_padding=bottom_padding,
    )
    return render_braille_on_image(
        image, [placement], dot_color=dot_color, draw_bbox=False
    )


def render_standard_braille(
    text: str,
    *,
    width: int = 256,
    height: int = 64,
    dot_color: tuple = (50, 50, 50),
    bg_color: tuple = (240, 240, 240),
    plate_size: float = 0.12,
    padding: float = 0.005,
) -> Image.Image:
    """
    Render braille onto a blank background, wrapping text  as it would
    on a plate of size ``plate_size`` with a side padding of ``padding``.
    """
    usable_size = plate_size - 2 * padding
    wrapped = _wrap_braille_text(text, usable_size)
    background = Image.new("RGB", (width, height), bg_color)
    placement = BraillePlacement(
        text=wrapped,
        x=0,
        y=0,
        width=width,
        height=height,
        padding=0.1,  # Display padding (not plate padding)
    )
    return render_braille_on_image(
        background, [placement], dot_color=dot_color, draw_bbox=False
    )


def draw_pending_box(
    image: Image.Image,
    start_point: tuple[int, int] | None,
    end_point: tuple[int, int] | None = None,
    text: str = "",
    box_color: _PILColor = (100, 150, 255),
) -> Image.Image:
    """Overlay a pending bounding box (dashed outline + braille preview) onto ``image``.

    Returns:
        - A copy of ``image`` if ``start_point`` is None.
        - A copy with just the start marker if ``end_point`` is None.
        - Otherwise, a copy with the dashed box and the
          rendered braille for ``text`` inside.
    """
    img = image.copy()
    if start_point is None:
        return img

    draw = ImageDraw.Draw(img)
    sx, sy = start_point

    # Start-point marker
    _draw_dot(draw, start_point, radius=6, color=(255, 100, 100))
    draw.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], outline="white", width=2)

    if end_point is None:
        return img

    ex, ey = end_point
    x1, y1 = min(sx, ex), min(sy, ey)
    x2, y2 = max(sx, ex), max(sy, ey)

    for a, b, c, d in (
        (x1, y1, x2, y1),  # top
        (x2, y1, x2, y2),  # right
        (x2, y2, x1, y2),  # bottom
        (x1, y2, x1, y1),  # left
    ):
        _draw_dashed_line(draw, a, b, c, d, box_color)

    preview = BraillePlacement(
        text=text, x=x1, y=y1, width=x2 - x1, height=y2 - y1, enabled=True
    )
    img = render_braille_on_image(img, [preview], dot_color=box_color, draw_bbox=False)

    return img


# =============================================================================
# Displacement maps (3D rendering)
# =============================================================================


def create_braille_displacement_map(
    placements: Iterable[BraillePlacement],
    *,
    width: int,
    height: int,
    dot_height: float = 1.0,
    flat_top_ratio: float = 0.0,
) -> np.ndarray:
    """Composite braille `placements` onto a fresh `(height, width)` displacement map."""
    displacement = np.zeros((height, width), dtype=np.float32)
    for p, r in _get_placement_dots(placements):
        _add_dome(
            displacement,
            p,
            r,
            height=dot_height,
            flat_top_ratio=flat_top_ratio,
        )
    return displacement


def create_standard_braille_displacement_map(
    text: str,
    *,
    plate_size: float,
    resolution: int = 1024,
    padding: float = 0.005,
    bottom_padding: float = 0.005,
    flat_top_ratio: float = 0.3,
) -> np.ndarray:
    """Create a braille displacement map using US/international standard dimensions.

    Args:
        text: Text to convert to braille.
        plate_size: Side length of the square plate, in meters.
        resolution: Output resolution in pixels (square).
        padding: Side padding in meters.
        flat_top_ratio: Fraction of dot radius that is flat on top.
        bottom_padding: Distance from the bottom edge of the usable area
            to the outer (lower) edge of the bottom row of dots, in
            meters.
    """
    placement = _make_standard_placement(
        text,
        plate_size=plate_size,
        resolution=resolution,
        padding=padding,
        bottom_padding=bottom_padding,
    )
    displacement = create_braille_displacement_map(
        [placement],
        width=resolution,
        height=resolution,
        flat_top_ratio=flat_top_ratio,
    )
    return displacement


# =============================================================================
# Standard-braille placement builder
# =============================================================================


def _make_standard_placement(
    text: str,
    *,
    plate_size: float,
    resolution: int,
    padding: float = 0.005,
    bottom_padding: float = 0.005,
) -> BraillePlacement:
    """Build a `BraillePlacement` for standard-dimensioned braille on a square plate.

    The placement is in pixel coordinates on a ``resolution × resolution`` image of
    a ``plate_size × plate_size`` plate.

    Text is word-wrapped to fit inside ``plate_size - 2 * padding``, possibly
    with multiple lines. A single line is centered horizontally on the plate;
    multi-line text is left-justified at the left edge of the usable area.

    The bbox bottom sits ``bottom_padding`` meters above the bottom edge
    of the *usable area* (``padding + bottom_padding`` from the bottom edge
    of the plate), with lines stacking upward.
    """
    pixels_per_meter = resolution / plate_size
    usable_size = plate_size - 2 * padding
    half_plate = plate_size / 2

    text = text.strip()
    text = _wrap_braille_text(text, usable_size)

    geom = BrailleGeometry(text)

    # Bbox in physical (centred-origin, y-up) plate coordinates.
    width_phys = geom.standard_width()
    height_phys = geom.standard_height()
    bbox_bottom_yup = -usable_size / 2 + bottom_padding
    bbox_top_yup = bbox_bottom_yup + height_phys
    if geom.n_lines == 1:
        bbox_left_phys = -width_phys / 2  # centered
    else:
        bbox_left_phys = -usable_size / 2  # left-justified

    # Convert (centered, y-up physical) → (top-left, y-down pixel).
    placement = BraillePlacement(
        text=text,
        x=(bbox_left_phys + half_plate) * pixels_per_meter,
        y=(half_plate - bbox_top_yup) * pixels_per_meter,
        width=width_phys * pixels_per_meter,
        height=height_phys * pixels_per_meter,
    )
    return placement


def _wrap_braille_text(text: str, max_line_width: float) -> str:
    """
    Greedily word-wrap `text` so each line fits in `max_line_width`
    (physical meters), assuming standard dimensions.
    """
    text = text.strip()

    lines: list[list[str]] = []
    current: list[str] = []
    current_width = 0.0

    for word in text.split():
        cells = len(pybraille.convertText(word))
        width = cells * STANDARD_CELL_SPACING
        if current:
            width += STANDARD_CELL_SPACING  # inter-word space

        if not current or current_width + width <= max_line_width:
            current.append(word)
            current_width += width
        else:
            lines.append(current)
            current = [word]
            current_width = cells * STANDARD_CELL_SPACING

    if current:
        lines.append(current)

    return "\n".join(" ".join(word for word in line) for line in lines)


def _get_placement_dots(
    placements: Iterable[BraillePlacement],
) -> list[tuple[_Point, float]]:
    """Get the coordinates and radii of all dots in the set of placements."""
    result = []
    enabled_placements = [p for p in placements if p.enabled]
    for placement in enabled_placements:
        geom = BrailleGeometry(placement.text)
        x1, y1, x2, y2 = placement.bbox
        points, radius = geom.position_braille_in_box(
            (x1, y1), (x2, y2), padding=placement.padding
        )
        result.extend((p, radius) for p in points)
    return result


# =============================================================================
# 2D drawing primitives
# =============================================================================


def _draw_dot(
    draw: ImageDraw.ImageDraw, p: _Point, radius: float, color: _PILColor
) -> None:
    """Draw a filled circle centered at ``p``."""
    cx, cy = p
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: _PILColor,
    dash_len: int = 8,
    gap_len: int = 4,
    width: int = 2,
) -> None:
    """Draw a dashed line from (x1, y1) to (x2, y2)."""
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg_len = dash_len if drawing else gap_len
        end_pos = min(pos + seg_len, length)
        if drawing:
            draw.line(
                [x1 + dx * pos, y1 + dy * pos, x1 + dx * end_pos, y1 + dy * end_pos],
                fill=color,
                width=width,
            )
        pos = end_pos
        drawing = not drawing


# =============================================================================
# Displacement primitives
# =============================================================================


def _add_dome(
    displacement: np.ndarray,
    p: _Point,
    radius: float,
    *,
    height: float = 1.0,
    flat_top_ratio: float = 0.0,
) -> None:
    """Composite a single braille dot dome into `displacement` (in-place, max-blend).

    Cosine falloff from the dot center, optionally with a flat-topped plateau
    of radius `flat_top_ratio * radius` for smoother fingertip readability.
    """
    cx, cy = p
    h, w = displacement.shape
    y_coords, x_coords = np.ogrid[:h, :w]

    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    dot_mask = dist <= radius

    if flat_top_ratio > 0.0:
        flat_radius = radius * flat_top_ratio
        flat_mask = dist <= flat_radius
        normalized = (dist - flat_radius) / max(radius - flat_radius, 1e-8)
        dome = np.where(flat_mask, 1.0, np.cos(normalized * np.pi / 2) ** 2)
    else:
        dome = np.cos(dist / radius * np.pi / 2) ** 2

    dome = np.where(dot_mask, dome * height, 0.0)
    np.maximum(displacement, dome, out=displacement)
