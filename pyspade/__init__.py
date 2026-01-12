from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely import is_valid_reason

from pyspade._pyspade import triangulate

if TYPE_CHECKING:
    from shapely import Polygon


def triangulate_polygon(
    polygon: Polygon,
    max_area: float | None = None,
    min_angle: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a shapely polygon into a mesh.

    Supports polygons with holes. Holes are automatically excluded from the mesh.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon to triangulate. Must be valid (no self-intersections).
    max_area : float, optional
        Maximum triangle area for mesh refinement.
    min_angle : float, optional
        Minimum angle in degrees for mesh refinement.

    Returns
    -------
    vertices : np.ndarray
        Nx2 array of vertex coordinates (float64).
    faces : np.ndarray
        Mx3 array of triangle vertex indices (uint32).

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

    # Get exterior coordinates (remove closing duplicate vertex)
    exterior = list(polygon.exterior.coords)[:-1]

    # Get hole coordinates
    holes = [list(interior.coords)[:-1] for interior in polygon.interiors]

    return triangulate(exterior, holes if holes else None, max_area, min_angle)


__all__ = ["triangulate", "triangulate_polygon"]
