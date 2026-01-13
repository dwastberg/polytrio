import numpy as np
import pytest
from shapely.geometry import Polygon, Point
from pyspade import triangulate_polygon

def test_subdomain_edges_preserved():
    exterior = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    subdomain = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    
    print("Subdomain coords:", list(subdomain.exterior.coords)[:-1])

    # Triangulate with subdomain
    verts, faces = triangulate_polygon(exterior, subdomains=[subdomain])
    
    def is_segment_covered(v1, v2, verts, faces):
        v1 = np.array(v1)
        v2 = np.array(v2)
        segment_vec = v2 - v1
        segment_len = np.linalg.norm(segment_vec)
        if segment_len < 1e-9:
            return True
        segment_dir = segment_vec / segment_len
        
        intervals = []
        
        for face in faces:
            for i in range(3):
                p1 = verts[face[i]]
                p2 = verts[face[(i+1)%3]]
                
                # Check collinearity
                vec1 = p1 - v1
                proj1 = np.dot(vec1, segment_dir)
                dist1 = np.linalg.norm(vec1 - proj1 * segment_dir)
                
                vec2 = p2 - v1
                proj2 = np.dot(vec2, segment_dir)
                dist2 = np.linalg.norm(vec2 - proj2 * segment_dir)
                
                if dist1 < 1e-9 and dist2 < 1e-9:
                    # Check if strictly within segment (or close enough)
                    start = min(proj1, proj2)
                    end = max(proj1, proj2)
                    
                    # Intersect with [0, segment_len]
                    start = max(start, 0.0)
                    end = min(end, segment_len)
                    
                    if end > start + 1e-9:
                        intervals.append((start, end))
        
        # Merge intervals
        intervals.sort()
        if not intervals:
            return False
            
        merged = []
        curr_start, curr_end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start < curr_end + 1e-9:
                curr_end = max(curr_end, next_end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged.append((curr_start, curr_end))
        
        # Check if we cover [0, segment_len]
        if len(merged) == 1:
            return merged[0][0] < 1e-9 and merged[0][1] > segment_len - 1e-9
        return False

    # Check each edge of the subdomain
    sub_coords = list(subdomain.exterior.coords)[:-1]
    for i in range(len(sub_coords)):
        v1 = sub_coords[i]
        v2 = sub_coords[(i+1) % len(sub_coords)]
        
        assert is_segment_covered(v1, v2, verts, faces), f"Edge {v1}-{v2} not found in triangulation"

    # Check if there are faces inside the subdomain
    faces_inside = 0
    for face in faces:
        # Calculate centroid
        pts = verts[face]
        centroid = pts.mean(axis=0)
        point = Point(centroid[0], centroid[1])
        if subdomain.contains(point):
            faces_inside += 1
    
    print(f"Faces inside subdomain: {faces_inside}")
    assert faces_inside > 0, "Subdomain should be triangulated, not a hole!"


def test_overlapping_subdomains_raise_value_error():
    exterior = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    sub1 = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
    sub2 = Polygon([(5, 5), (8, 5), (8, 8), (5, 8)])  # overlaps sub1

    with pytest.raises(
        ValueError,
        match="Subdomains must be pairwise disjoint",
    ):
        triangulate_polygon(exterior, subdomains=[sub1, sub2])


def test_subdomain_ids_returned():
    """Test that subdomain IDs are correctly assigned to faces."""
    exterior = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    subdomain1 = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
    subdomain2 = Polygon([(6, 6), (9, 6), (9, 9), (6, 9)])

    verts, faces, subdomain_ids = triangulate_polygon(
        exterior,
        subdomains=[subdomain1, subdomain2],
        return_subdomain_ids=True,
    )

    # Check subdomain_ids array shape
    assert len(subdomain_ids) == len(faces)
    assert subdomain_ids.dtype == np.int32

    # Check that some faces are in subdomain 0, some in 1, some in neither (-1)
    unique_ids = set(subdomain_ids)
    assert -1 in unique_ids  # Faces outside both subdomains
    assert 0 in unique_ids   # Faces in subdomain 0
    assert 1 in unique_ids   # Faces in subdomain 1

    # Verify faces are correctly classified
    for i, face in enumerate(faces):
        pts = verts[face]
        centroid = pts.mean(axis=0)
        point = Point(centroid[0], centroid[1])

        if subdomain1.contains(point):
            assert subdomain_ids[i] == 0, f"Face {i} centroid in subdomain1 but ID is {subdomain_ids[i]}"
        elif subdomain2.contains(point):
            assert subdomain_ids[i] == 1, f"Face {i} centroid in subdomain2 but ID is {subdomain_ids[i]}"
        # Note: boundary faces might be tricky, so allow some flexibility


def test_subdomain_ids_not_returned_by_default():
    """Test that subdomain IDs are not returned unless requested."""
    exterior = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    subdomain = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])

    result = triangulate_polygon(exterior, subdomains=[subdomain])

    # Should return only 2 arrays (verts, faces)
    assert len(result) == 2
    verts, faces = result


def test_subdomain_ids_with_no_subdomains():
    """Test that requesting subdomain IDs with no subdomains returns None."""
    exterior = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    result = triangulate_polygon(exterior, return_subdomain_ids=True)

    # Should still return only 2 arrays if no subdomains present
    assert len(result) == 2


if __name__ == "__main__":
    test_subdomain_edges_preserved()
    print("Test passed!")
