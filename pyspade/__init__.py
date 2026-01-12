from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyspade._pyspade import triangulate

if TYPE_CHECKING:
    from shapely import Polygon


def triangulate_polygon(
    polygon: Polygon,
    max_area: float | None = None,
    min_angle: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a shapely polygon into a mesh.

    Parameters
    ----------
    polygon : shapely.Polygon
        The polygon to triangulate.
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
    """
    # Get exterior coordinates, removing the closing duplicate vertex
    coords = list(polygon.exterior.coords)[:-1]
    return triangulate(coords, max_area, min_angle)


__all__ = ["triangulate", "triangulate_polygon"]
