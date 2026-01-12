"""Visualize polygon triangulation with matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from shapely import Polygon

from pyspade import triangulate_polygon


def plot_polygon_and_mesh(
    polygon: Polygon,
    max_area: float | None = None,
    min_angle: float | None = None,
) -> None:
    """Plot the input polygon and its triangulation side by side.

    Parameters
    ----------
    polygon : Polygon
        The shapely polygon to triangulate.
    max_area : float, optional
        Maximum triangle area for mesh refinement.
    min_angle : float, optional
        Minimum angle in degrees for mesh refinement.
    """
    # Triangulate the polygon
    vertices, faces = triangulate_polygon(polygon, max_area, min_angle)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot input polygon on the left
    x, y = polygon.exterior.xy
    ax1.fill(x, y, alpha=0.3, facecolor="blue", edgecolor="blue", linewidth=2)
    ax1.set_title("Input Polygon", fontsize=14)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Plot triangulation on the right
    triangles = vertices[faces]
    collection = PolyCollection(
        triangles,
        facecolors="lightblue",
        edgecolors="darkblue",
        linewidths=0.5,
        alpha=0.7,
    )
    ax2.add_collection(collection)

    # Plot vertices
    ax2.scatter(vertices[:, 0], vertices[:, 1], c="red", s=10, zorder=5)

    # Set axis limits
    ax2.set_xlim(vertices[:, 0].min() - 0.1, vertices[:, 0].max() + 0.1)
    ax2.set_ylim(vertices[:, 1].min() - 0.1, vertices[:, 1].max() + 0.1)
    ax2.set_title(
        f"Triangulation ({len(faces)} triangles, {len(vertices)} vertices)",
        fontsize=14,
    )
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run example visualizations."""
    # Example 1: Simple square
    print("Example 1: Simple square")
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    plot_polygon_and_mesh(square)

    # Example 2: Square with refinement
    print("Example 2: Square with refinement")
    plot_polygon_and_mesh(square, max_area=0.05, min_angle=25.0)

    # Example 3: L-shaped polygon
    print("Example 3: L-shaped polygon")
    l_shape = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    plot_polygon_and_mesh(l_shape, max_area=0.1)

    # Example 4: Star shape
    print("Example 4: Star shape")
    n_points = 5
    outer_radius = 1.0
    inner_radius = 0.4
    angles = np.linspace(0, 2 * np.pi, 2 * n_points + 1)[:-1]
    radii = [outer_radius if i % 2 == 0 else inner_radius for i in range(2 * n_points)]
    coords = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles)]
    star = Polygon(coords)
    plot_polygon_and_mesh(star, max_area=0.02, min_angle=20.0)


if __name__ == "__main__":
    main()
