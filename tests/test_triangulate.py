import numpy as np
import pytest
from shapely import Polygon

from pyspade import triangulate, triangulate_polygon


class TestTriangulatePolygon:
    """Tests for the triangulate_polygon function."""

    def test_simple_square(self):
        """A square should produce 2 triangles."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        verts, faces = triangulate_polygon(poly)

        assert verts.shape == (4, 2)
        assert faces.shape == (2, 3)

    def test_triangle_input(self):
        """A triangle should produce 1 triangle."""
        poly = Polygon([(0, 0), (1, 0), (0.5, 1)])
        verts, faces = triangulate_polygon(poly)

        assert verts.shape == (3, 2)
        assert faces.shape == (1, 3)

    def test_pentagon(self):
        """A pentagon should produce 3 triangles."""
        # Regular pentagon vertices
        angles = np.linspace(0, 2 * np.pi, 6)[:-1]
        coords = [(np.cos(a), np.sin(a)) for a in angles]
        poly = Polygon(coords)
        verts, faces = triangulate_polygon(poly)

        assert verts.shape == (5, 2)
        assert faces.shape == (3, 3)

    def test_output_dtypes(self):
        """Check that output arrays have correct dtypes."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        verts, faces = triangulate_polygon(poly)

        assert verts.dtype == np.float64
        assert faces.dtype == np.uint32

    def test_vertices_match_input(self):
        """Without refinement, output vertices should match input."""
        input_coords = [(0, 0), (2, 0), (2, 2), (0, 2)]
        poly = Polygon(input_coords)
        verts, faces = triangulate_polygon(poly)

        # Check that all input coordinates are present in output
        for x, y in input_coords:
            found = np.any((verts[:, 0] == x) & (verts[:, 1] == y))
            assert found, f"Vertex ({x}, {y}) not found in output"

    def test_valid_face_indices(self):
        """All face indices should be valid vertex indices."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        verts, faces = triangulate_polygon(poly)

        assert np.all(faces >= 0)
        assert np.all(faces < len(verts))


class TestRefinement:
    """Tests for mesh refinement parameters."""

    def test_max_area_increases_triangles(self):
        """Setting max_area should produce more triangles."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        _, faces_basic = triangulate_polygon(poly)
        _, faces_refined = triangulate_polygon(poly, max_area=0.1)

        assert faces_refined.shape[0] > faces_basic.shape[0]

    def test_min_angle_with_max_area(self):
        """Setting min_angle with max_area should produce quality triangles."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        # min_angle alone may not trigger refinement if triangles already satisfy it
        # but combined with max_area it should work
        _, faces_refined = triangulate_polygon(poly, max_area=0.2, min_angle=25.0)

        assert faces_refined.shape[0] > 2

    def test_smaller_max_area_more_triangles(self):
        """Smaller max_area should produce more triangles."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        _, faces_large = triangulate_polygon(poly, max_area=0.5)
        _, faces_small = triangulate_polygon(poly, max_area=0.1)

        assert faces_small.shape[0] > faces_large.shape[0]

    def test_combined_refinement(self):
        """Both max_area and min_angle can be used together."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        verts, faces = triangulate_polygon(poly, max_area=0.1, min_angle=20.0)

        # Should produce a refined mesh
        assert faces.shape[0] > 2
        assert verts.shape[0] > 4


class TestTriangulateDirect:
    """Tests for the low-level triangulate function."""

    def test_direct_coordinates(self):
        """Test triangulate with raw coordinate list."""
        coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        verts, faces = triangulate(coords)

        assert verts.shape == (4, 2)
        assert faces.shape == (2, 3)

    def test_direct_with_refinement(self):
        """Test triangulate with refinement parameters."""
        coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        verts, faces = triangulate(coords, max_area=0.1)

        assert faces.shape[0] > 2


class TestEdgeCases:
    """Tests for edge cases and special polygons."""

    def test_large_polygon(self):
        """Test with a polygon with many vertices."""
        # Create a circle-like polygon with 20 vertices
        n = 20
        angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
        coords = [(np.cos(a), np.sin(a)) for a in angles]
        poly = Polygon(coords)

        verts, faces = triangulate_polygon(poly)

        assert verts.shape == (n, 2)
        # A convex n-gon triangulates to n-2 triangles
        assert faces.shape == (n - 2, 3)

    def test_non_convex_polygon(self):
        """Test with a non-convex (L-shaped) polygon."""
        # L-shaped polygon
        coords = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        poly = Polygon(coords)

        verts, faces = triangulate_polygon(poly)

        assert verts.shape[0] == 6
        # L-shape with 6 vertices needs at least 4 triangles (n-2 for convex)
        # but constrained Delaunay may produce more for non-convex shapes
        assert faces.shape[0] >= 4

    def test_narrow_polygon(self):
        """Test with a very narrow polygon."""
        coords = [(0, 0), (10, 0), (10, 0.1), (0, 0.1)]
        poly = Polygon(coords)

        verts, faces = triangulate_polygon(poly)

        assert verts.shape == (4, 2)
        assert faces.shape == (2, 3)
