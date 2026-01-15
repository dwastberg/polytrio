"""Visualize polygon triangulation with matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from shapely import Polygon

from pyspade import triangulate_polygon


def plot_polygon_and_mesh(
    polygon: Polygon,
    subdomains: list[Polygon] | None = None,
    max_area: float | None = None,
    min_angle: float | None = None,
    color_by_subdomain: bool = False,
) -> None:
    """Plot the input polygon and its triangulation side by side.

    Parameters
    ----------
    polygon : Polygon
        The shapely polygon to triangulate.
    subdomains : list[Polygon], optional
        List of subdomains to enforce as constraints.
    max_area : float, optional
        Maximum triangle area for mesh refinement.
    min_angle : float, optional
        Minimum angle in degrees for mesh refinement.
    color_by_subdomain : bool, optional
        If True and subdomains are provided, color triangles by subdomain ID.
    """
    # Triangulate the polygon
    if color_by_subdomain and subdomains:
        vertices, faces, subdomain_ids = triangulate_polygon(
            polygon,
            subdomains=subdomains,
            max_area=max_area,
            min_angle=min_angle,
            return_subdomain_ids=True
        )
    else:
        result = triangulate_polygon(
            polygon, subdomains=subdomains, max_area=max_area, min_angle=min_angle
        )
        vertices, faces = result
        subdomain_ids = None

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot input polygon on the left
    x, y = polygon.exterior.xy
    ax1.fill(x, y, alpha=0.3, facecolor="blue", edgecolor="blue", linewidth=2)
    
    # Plot subdomains
    if subdomains:
        for i, sub in enumerate(subdomains):
            x_sub, y_sub = sub.exterior.xy
            ax1.plot(x_sub, y_sub, color="green", linewidth=2, linestyle="--", label="Subdomain" if i == 0 else "")
        ax1.legend()

    ax1.set_title("Input Polygon", fontsize=14)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Plot triangulation on the right
    triangles = vertices[faces]

    # Color by subdomain if requested
    if subdomain_ids is not None:
        # Create a colormap for subdomains
        import matplotlib as mpl

        unique_ids = np.unique(subdomain_ids)
        n_subdomains = len([sid for sid in unique_ids if sid >= 0])

        # Generate distinct colors for each subdomain
        cmap = mpl.colormaps.get_cmap('tab10' if n_subdomains <= 10 else 'tab20')
        colors = []

        for sub_id in subdomain_ids:
            if sub_id == -1:
                # Not in any subdomain - use light gray
                colors.append('lightgray')
            else:
                # Use colormap for subdomain
                colors.append(cmap(sub_id / max(1, n_subdomains - 1)))

        collection = PolyCollection(
            triangles,
            facecolors=colors,
            edgecolors="darkblue",
            linewidths=0.5,
            alpha=0.7,
        )

        # Add legend for subdomains
        from matplotlib.patches import Patch
        legend_elements = []
        for i in range(n_subdomains):
            legend_elements.append(
                Patch(facecolor=cmap(i / max(1, n_subdomains - 1)),
                      edgecolor='darkblue',
                      label=f'Subdomain {i}')
            )
        legend_elements.append(
            Patch(facecolor='lightgray', edgecolor='darkblue', label='No subdomain')
        )
        ax2.legend(handles=legend_elements, loc='upper right')
    else:
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
    # # Example 1: Simple square
    # print("Example 1: Simple square")
    # square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    # plot_polygon_and_mesh(square)
    #
    # # Example 2: Square with refinement
    # print("Example 2: Square with refinement")
    # plot_polygon_and_mesh(square, max_area=0.05, min_angle=25.0)
    #
    # # Example 3: L-shaped polygon
    # print("Example 3: L-shaped polygon")
    # l_shape = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    # plot_polygon_and_mesh(l_shape, max_area=0.1)
    #
    # # Example 4: Star shape
    # print("Example 4: Star shape")
    # n_points = 5
    # outer_radius = 1.0
    # inner_radius = 0.4
    # angles = np.linspace(0, 2 * np.pi, 2 * n_points + 1)[:-1]
    # radii = [outer_radius if i % 2 == 0 else inner_radius for i in range(2 * n_points)]
    # coords = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles)]
    # star = Polygon(coords)
    # plot_polygon_and_mesh(star, max_area=0.02, min_angle=20.0)
    #
    # # Example 5: Polygon with hole
    # print("Example 5: Square with square hole")
    # exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
    # hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    # poly_with_hole = Polygon(exterior, holes=[hole])
    # plot_polygon_and_mesh(poly_with_hole, max_area=0.5)
    #
    # # Example 6: Polygon with multiple holes
    # print("Example 6: Rectangle with two holes")
    # exterior = [(0, 0), (20, 0), (20, 10), (0, 10)]
    # hole1 = [(2, 2), (6, 2), (6, 8), (2, 8)]
    # hole2 = [(14, 2), (18, 2), (18, 8), (14, 8)]
    # poly_with_holes = Polygon(exterior, holes=[hole1, hole2])
    # plot_polygon_and_mesh(poly_with_holes, max_area=0.5, min_angle=20.0)

    # Example 7: Polygon with subdomains (colored by subdomain)
    print("Example 7: Square with internal subdomain (colored by subdomain)")
    exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
    subdomain = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])
    poly = Polygon(exterior)
    plot_polygon_and_mesh(poly, subdomains=[subdomain], max_area=1.0, color_by_subdomain=True)

    # Example 8: Complex polygon with multiple subdomains and a hole (colored by subdomain)
    print("Example 8: Complex polygon with multiple subdomains and a hole (colored by subdomain)")
    exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
    hole = [(15, 15), (18, 15), (18, 18), (15, 18)]

    # sub1 has a hole where sub3 will be
    sub1_ext = [(2, 2), (8, 2), (8, 8), (2, 8)]
    sub1_hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
    sub1 = Polygon(sub1_ext, holes=[sub1_hole])

    sub2 = Polygon([(10, 2), (18, 2), (18, 8), (10, 8)])

    poly = Polygon(exterior, holes=[hole])
    plot_polygon_and_mesh(poly, subdomains=[sub1, sub2], max_area=1.0, color_by_subdomain=True)


if __name__ == "__main__":
    main()
