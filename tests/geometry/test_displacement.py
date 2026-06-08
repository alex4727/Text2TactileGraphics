from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pytest_regressions.image_regression import ImageRegressionFixture
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from tactilegen import TexturedSegment
from tactilegen.generation.utils import displacement_to_image
from tactilegen.geometry.braille import BraillePlacement
from tactilegen.geometry.displacement import (
    NormalFormat,
    apply_braille_displacements,
    apply_displacement_to_mesh,
    apply_segment_displacements,
    apply_standard_braille_displacement,
    tileable_patch_to_displacement,
)
from tactilegen.geometry.filtering import normal_to_rgb

_DATA_DIR = Path(__file__).parent / "data"
_NORMAL_IMG = Image.open(_DATA_DIR / "normal_tileable.png").convert("RGB")

# =============================================================================
# Test applying displacements to mesh
# =============================================================================


def _make_flat_normal_img(h: int = 64, w: int = 64) -> Image.Image:
    """Flat normal map (all pointing +Z)."""
    n: np.ndarray = np.zeros((h, w, 3), dtype=np.float32)
    n[..., 2] = 1.0  # (0, 0, 1) in [-1, 1]
    return Image.fromarray(normal_to_rgb(n).astype(np.uint8))


def _subdivided_plane(n: int = 128):
    """Return an `n × n`-vertex unit plane at z=0 for displacement tests."""
    trimesh = pytest.importorskip("trimesh")
    ii, jj = np.mgrid[0:n, 0:n]
    vertices = np.stack(
        [ii.ravel() / (n - 1), jj.ravel() / (n - 1), np.zeros(n * n)], axis=1
    )
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b, c, d = a + 1, a + n, a + n + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


class TestApplySegmentDisplacements:
    def test_single_segment(
        self, ndarrays_regression: NDArraysRegressionFixture, tmp_path: Path
    ):
        mesh = _subdivided_plane(n=128)
        seg = TexturedSegment(
            mask=np.ones((128, 128), dtype=bool), tileable_patch=_NORMAL_IMG
        )

        out_mesh = apply_segment_displacements(mesh, [seg])

        out_mesh.export(tmp_path / "output.glb")
        ndarrays_regression.check({"vertices": out_mesh.vertices.astype(np.float32)})

    def test_two_segments(
        self, ndarrays_regression: NDArraysRegressionFixture, tmp_path: Path
    ):
        # Two complementary half-masks at different displacement scales
        mesh = _subdivided_plane(n=128)
        left = np.zeros((128, 128), dtype=bool)
        right = np.zeros((128, 128), dtype=bool)
        left[:, :64] = True
        right[:, 64:] = True

        segs = [
            TexturedSegment(
                mask=left, tileable_patch=_NORMAL_IMG, displacement_scale=0.005
            ),
            TexturedSegment(
                mask=right, tileable_patch=_NORMAL_IMG, displacement_scale=0.010
            ),
        ]
        out_mesh = apply_segment_displacements(mesh, segs)

        out_mesh.export(tmp_path / "output.glb")
        ndarrays_regression.check({"vertices": out_mesh.vertices.astype(np.float32)})


class TestApplyBrailleDisplacements:
    def test_apply_braille_displacements(
        self, ndarrays_regression: NDArraysRegressionFixture, tmp_path: Path
    ):
        mesh = _subdivided_plane(n=128)
        placements = [
            BraillePlacement(text="test", x=10, y=10, width=50, height=20, padding=0.1)
        ]
        out_mesh = apply_braille_displacements(
            mesh,
            placements,
            image_size=(100, 100),
            target_resolution=128,
            dot_height=0.01,
        )
        out_mesh.export(tmp_path / "output.glb")
        ndarrays_regression.check({"vertices": out_mesh.vertices.astype(np.float32)})


class TestApplyStandardBrailleDisplacement:
    def test_apply_standard_braille_displacement(
        self, ndarrays_regression: NDArraysRegressionFixture, tmp_path: Path
    ):
        mesh = _subdivided_plane(n=128)
        out_mesh = apply_standard_braille_displacement(
            mesh, text="test", plate_size=0.1, target_resolution=128
        )
        out_mesh.export(tmp_path / "output.glb")
        ndarrays_regression.check({"vertices": out_mesh.vertices.astype(np.float32)})


class TestApplyDisplacementToMesh:
    def test_zero_scale_preserves_geometry(self):
        mesh = _subdivided_plane(n=8)
        dm = np.random.RandomState(0).rand(16, 16).astype(np.float32)
        out = apply_displacement_to_mesh(mesh, dm, scale=0.0, direction="z")
        np.testing.assert_allclose(out.vertices, mesh.vertices)

    def test_apply_displacement_to_mesh(
        self, ndarrays_regression: NDArraysRegressionFixture, tmp_path: Path
    ):
        mesh = _subdivided_plane(n=32)
        dm = np.linspace(0, 1, 32 * 32).reshape(32, 32).astype(np.float32)
        out = apply_displacement_to_mesh(mesh, dm, scale=0.1, direction="z")
        out.export(tmp_path / "output.glb")
        ndarrays_regression.check({"vertices": out.vertices.astype(np.float32)})


class TestClampNormalDispToMask:
    """Boundary-aware clamping of normal-direction displacement."""

    @staticmethod
    def _left_half_mask(size: int = 64) -> np.ndarray:
        """Mask that is inside for u < 0.5 (left half in x), outside otherwise."""
        mask = np.zeros((size, size), dtype=bool)
        mask[:, : size // 2] = True
        return mask

    # The UV transform derives its x-y bounds from the *given* vertices, so
    # tests anchor the [0, 1] x [0, 1] domain with two corner vertices (which
    # carry zero displacement) and probe the third, real vertex at index 2.
    _ANCHORS = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    _ANCHOR_DISP = np.zeros((2, 3))

    def _run(self, probe_vertex, probe_disp):
        from tactilegen.geometry.displacement import _clamp_normal_disp_to_mask

        vertices = np.vstack([self._ANCHORS, [probe_vertex]])
        disp_vec = np.vstack([self._ANCHOR_DISP, [probe_disp]])
        out = _clamp_normal_disp_to_mask(
            vertices,
            disp_vec,
            uv_mode="planar_xy",
            preserve_aspect=True,
            mask=self._left_half_mask(),
            tiling=(1.0, 1.0),
        )
        return out[2]  # displacement applied to the probe vertex

    def test_lateral_motion_clamped_at_boundary(self):
        # Probe starts inside (x=0.4, u=0.4) and is pushed +x past the u=0.5
        # boundary; it must be clamped so its displaced x lands ~at the wall.
        probe = [0.4, 0.5, 0.0]
        disp = [0.4, 0.0, 0.2]  # +x lateral, +z up
        out = self._run(probe, disp)
        moved_x = probe[0] + out[0]
        assert moved_x <= 0.5 + 1e-2  # never crosses the mask boundary
        assert moved_x == pytest.approx(0.5, abs=2e-2)  # stops *at* the wall
        # Up motion is scaled by the same fraction (rides up the normal ray).
        frac = out[0] / disp[0]
        assert out[2] == pytest.approx(disp[2] * frac, rel=1e-5)

    def test_interior_vertex_unclamped(self):
        # Stays inside after displacement (0.1 -> 0.3, both u < 0.5): no clamp.
        out = self._run([0.1, 0.5, 0.0], [0.2, 0.0, 0.3])
        np.testing.assert_allclose(out, [0.2, 0.0, 0.3])

    def test_purely_vertical_motion_unaffected(self):
        # No lateral component -> x-y never leaves the mask -> full displacement.
        out = self._run([0.45, 0.5, 0.0], [0.0, 0.0, 0.5])
        np.testing.assert_allclose(out, [0.0, 0.0, 0.5])


# =============================================================================
# Test tileable patch to displacement
# =============================================================================


class TestTileablePatchToDisplacement:
    def test_rgb_normal_map(self):
        img = _make_flat_normal_img(32, 32)
        out = tileable_patch_to_displacement(img)
        assert out.shape == (32, 32)
        assert np.all(np.isfinite(out))

    def test_rgba_input_handled(self):
        rgb = _make_flat_normal_img(16, 16)
        rgba = rgb.convert("RGBA")
        out = tileable_patch_to_displacement(rgba)
        assert out.shape == (16, 16)

    def test_grayscale_input_direct_normalize(self):
        gray = Image.new("L", (16, 16), 128)
        out = tileable_patch_to_displacement(gray)
        assert out.shape == (16, 16)
        # 128/255 -> ~0.502
        assert out.min() == pytest.approx(128 / 255, abs=1e-5)
        assert out.max() == pytest.approx(128 / 255, abs=1e-5)

    @pytest.mark.parametrize("normal_format", ["opengl", "directx"])
    def test_tileable_patch_to_displacement(
        self, normal_format: NormalFormat, image_regression: ImageRegressionFixture
    ):
        displacement = tileable_patch_to_displacement(
            _NORMAL_IMG, normal_format=normal_format
        )
        image_regression.check(
            displacement_to_image(displacement, normalize=False),
            basename=f"tileable_patch_displacement_{normal_format}",
        )
