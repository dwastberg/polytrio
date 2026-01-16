from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely import is_valid_reason, to_wkb
from shapely.strtree import STRtree


from pyspade._pyspade import triangulate

from typing import TYPE_CHECKING, Optional, List, Union

from shapely.geometry import Polygon, MultiPolygon, LineString
def _all_disjoint(polygons):
    """
    Returns True if all polygons are disjoint (touching is allowed),
    and False as soon as any overlapping pair is found.
    """
    tree = STRtree(polygons)

    for i, poly in enumerate(polygons):
        # Only check likely candidates via bounding boxes
        for idx in tree.query(poly):
            if idx == i:
                continue
            other = polygons[idx]
            # Overlap = interiors intersect (not just touching)
            if poly.intersects(other) and not poly.touches(other):
                return False
    return True

def triangulate_polygon(
    polygon: Polygon,
    subdomains: Optional[List[Polygon]] = None,
    max_area: float | None = None,
    min_angle: float | None = None,
    return_shapely: bool = False,
    return_subdomain_ids: bool = False,
    ) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], MultiPolygon]:
    """Triangulate a shapely polygon into a mesh.

    Supports polygons with holes. Holes are automatically excluded from the mesh.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon to triangulate. Must be valid (no self-intersections).
    subdomains : list of internal subdomains, optional
    max_area : float, optional
        Maximum triangle area for mesh refinement.
    min_angle : float, optional
        Minimum angle in degrees for mesh refinement.
    return_shapely : bool, optional
        If True, return a shapely MultiPolygon representing the mesh instead of raw arrays.
    return_subdomain_ids : bool, optional
        If True, return subdomain IDs for each face. Only applicable when subdomains are provided.

    Returns
    -------
    verts : np.ndarray
        Nx2 array of vertex coordinates.
    faces : np.ndarray
        Mx3 array of triangle vertex indices.
    subdomain_ids : np.ndarray, optional
        Mx1 array of int32 subdomain IDs (if return_subdomain_ids=True).
        Each element is the 0-based index of the subdomain containing that face, or -1 if not in any subdomain.
    or
    multipolygon : shapely.MultiPolygon
        The resulting MultiPolygon representing the mesh.
    Raises
    ------
    ValueError
        If the polygon is not valid (self-intersections, holes touching exterior, etc.)
    """
    # Validate polygon before processing
    if not polygon.is_valid:
        reason = is_valid_reason(polygon)
        raise ValueError(
            f"Invalid polygon: {reason}. "
            "Polygon must not have self-intersections, holes touching the exterior, etc."
        )

    # Validate subdomains if provided
    if subdomains:
        for idx, subdomain in enumerate(subdomains):
            if not subdomain.is_valid:
                reason = is_valid_reason(subdomain)
                raise ValueError(
                    f"subdomain polygon {idx} is invalid: {reason}. "
                )
        if not _all_disjoint(subdomains):
            raise ValueError(
                "Subdomains must be pairwise disjoint (they can touch at the boundary but not overlap)."
            )
        for subdomain in subdomains:
            # Check that subdomain is within the main polygon
            if not polygon.contains(subdomain):
                raise ValueError(
                    "Each subdomain polygon must be strictly contained within the main polygon interior "
                    "(not outside, and not touching the exterior boundary or any hole boundary)."
                )

    # Convert Shapely polygons to WKB bytes
    polygon_wkb = to_wkb(polygon)
    subdomains_wkb = [to_wkb(sub) for sub in subdomains] if subdomains else None

    verts, faces, subdomain_ids = triangulate(
        polygon_wkb,
        subdomains_wkb,
        max_area,
        min_angle,
        return_subdomain_ids,
    )

    # Filter unused vertices
    unique_indices = np.unique(faces)
    if len(unique_indices) < len(verts):
        # Create a mapping from old index to new index
        new_indices = np.full(len(verts), -1, dtype=int)
        new_indices[unique_indices] = np.arange(len(unique_indices))

        # Update faces
        faces = new_indices[faces]

        # Update vertices
        verts = verts[unique_indices]

    if return_shapely:
        multipolygon = mesh_to_multipolygon(verts, faces)
        return multipolygon

    # Return with or without subdomain_ids
    if return_subdomain_ids and subdomain_ids is not None:
        return verts, faces, subdomain_ids
    else:
        return verts, faces

def mesh_to_multipolygon(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> MultiPolygon:
    """Convert mesh vertices and faces to a shapely MultiPolygon.

    Parameters
    ----------
    vertices : np.ndarray
        Nx2 array of vertex coordinates.
    faces : np.ndarray
        Mx3 array of triangle vertex indices.

    Returns
    -------
    multipolygon : shapely.MultiPolygon
        The resulting MultiPolygon representing the mesh.
    """

    polygons = []
    for face in faces:
        triangle_coords = vertices[face]
        polygon = Polygon(triangle_coords)
        polygons.append(polygon)

    return MultiPolygon(polygons)


__all__ = ["triangulate", "triangulate_polygon"]
