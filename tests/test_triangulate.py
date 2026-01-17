import numpy as np
import pytest
from shapely import Polygon, to_wkb

from polytrio import triangulate, triangulate_polygon


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
        """Test triangulate with WKB bytes."""
        polygon = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        polygon_wkb = to_wkb(polygon)
        verts, faces, subdomain_ids = triangulate(polygon_wkb, None, None, None, False)

        assert verts.shape == (4, 2)
        assert faces.shape == (2, 3)
        assert subdomain_ids is None  # No subdomains, no IDs requested

    def test_direct_with_refinement(self):
        """Test triangulate with refinement parameters and WKB bytes."""
        polygon = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        polygon_wkb = to_wkb(polygon)
        verts, faces, subdomain_ids = triangulate(polygon_wkb, None, 0.1, None, False)

        assert faces.shape[0] > 2
        assert subdomain_ids is None  # No subdomains, no IDs requested


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


class TestPolygonsWithHoles:
    """Tests for polygons with holes."""

    def test_square_with_square_hole(self):
        """A square with a square hole."""
        from shapely.geometry import Point

        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(2, 2), (8, 2), (8, 8), (2, 8)]
        poly = Polygon(exterior, holes=[hole])

        verts, faces = triangulate_polygon(poly)

        # Should have vertices from both exterior and hole (at least)
        assert verts.shape[0] >= 8  # At least 4 exterior + 4 hole

        # Verify no triangles have centroids inside the hole
        hole_poly = Polygon(hole)
        for face in faces:
            centroid = verts[face].mean(axis=0)
            centroid_point = Point(centroid[0], centroid[1])
            assert not hole_poly.contains(centroid_point)

    def test_polygon_with_multiple_holes(self):
        """Polygon with multiple holes."""
        from shapely.geometry import Point

        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole1 = [(2, 2), (8, 2), (8, 8), (2, 8)]
        hole2 = [(12, 12), (18, 12), (18, 18), (12, 18)]
        poly = Polygon(exterior, holes=[hole1, hole2])

        verts, faces = triangulate_polygon(poly)

        # Should have vertices from exterior and both holes
        assert verts.shape[0] >= 12  # At least 4+4+4

        # Verify no faces inside either hole
        hole1_poly = Polygon(hole1)
        hole2_poly = Polygon(hole2)
        for face in faces:
            centroid = verts[face].mean(axis=0)
            centroid_point = Point(centroid[0], centroid[1])
            assert not hole1_poly.contains(centroid_point)
            assert not hole2_poly.contains(centroid_point)

    def test_hole_with_refinement(self):
        """Test that refinement works with holes."""
        from shapely.geometry import Point

        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        poly = Polygon(exterior, holes=[hole])

        verts, faces = triangulate_polygon(poly, max_area=0.5)

        # Should have many more vertices due to refinement
        assert verts.shape[0] > 8

        # No faces should be inside the hole
        hole_poly = Polygon(hole)
        for face in faces:
            centroid = verts[face].mean(axis=0)
            centroid_point = Point(centroid[0], centroid[1])
            assert not hole_poly.contains(centroid_point)

    def test_invalid_polygon_raises_error(self):
        """Test that invalid polygons raise ValueError."""
        # Self-intersecting polygon (bowtie shape)
        exterior = [(0, 0), (2, 2), (2, 0), (0, 2)]
        poly = Polygon(exterior)

        with pytest.raises(ValueError, match="Invalid polygon"):
            triangulate_polygon(poly)


class TestWKBIntegration:
    """Tests for WKB-based polygon passing."""

    def test_wkb_coordinate_preservation(self):
        """Verify coordinates preserved exactly through WKB."""
        coords = [(0.123456789, 0.987654321), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        poly = Polygon(coords)
        verts, faces = triangulate_polygon(poly)

        # Check first coordinate preserved to machine precision
        assert any(
            np.allclose(verts[i], [0.123456789, 0.987654321], atol=1e-9)
            for i in range(len(verts))
        )

    def test_polygon_with_holes_wkb(self):
        """Holes should work correctly through WKB."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(2, 2), (8, 2), (8, 8), (2, 8)]
        poly = Polygon(exterior, [hole])
        verts, faces = triangulate_polygon(poly)

        # Should produce triangulation excluding hole interior
        assert faces.shape[0] > 0

        # Verify no faces are inside the hole
        from shapely.geometry import Point
        hole_poly = Polygon(hole)
        for face in faces:
            centroid = verts[face].mean(axis=0)
            centroid_point = Point(centroid[0], centroid[1])
            assert not hole_poly.contains(centroid_point)

    def test_subdomains_through_wkb(self):
        """Subdomains should work correctly through WKB."""
        exterior = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        sub1 = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
        sub2 = Polygon([(6, 6), (9, 6), (9, 9), (6, 9)])

        verts, faces, ids = triangulate_polygon(
            exterior,
            subdomains=[sub1, sub2],
            return_subdomain_ids=True
        )

        # Should have three categories: -1, 0, 1
        unique_ids = set(ids)
        assert unique_ids == {-1, 0, 1}

    def test_complex_polygon_through_wkb(self):
        """Complex polygon with many vertices should work through WKB."""
        # Create polygon with 100 vertices
        angles = np.linspace(0, 2 * np.pi, 100)
        coords = [(np.cos(a), np.sin(a)) for a in angles]
        poly = Polygon(coords)

        verts, faces = triangulate_polygon(poly)

        # Should produce valid triangulation
        assert verts.shape[0] >= 100
        assert faces.shape[0] > 0
        assert verts.dtype == np.float64
        assert faces.dtype == np.uint32
