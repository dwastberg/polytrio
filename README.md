# Polytrio

Fast Delaunay triangulation for polygons with holes and subdomains, powered by Rust. PySpade provides Python bindings to the Rust `spade` library for high-performance constrained Delaunay triangulation (CDT) with support for mesh refinement.

## Installation

```bash
pip install polytrio
```

Requires Python ≥3.10, numpy, and shapely.

## Examples

### Basic Polygon Triangulation

```python
from shapely.geometry import Polygon
from polytrio import triangulate_polygon

# Create a simple polygon
polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

# Triangulate it
vertices, faces = triangulate_polygon(polygon)

# vertices: (N, 2) array of vertex coordinates
# faces: (M, 3) array of triangle vertex indices
```

### Polygon with Holes

```python
# Polygon with a hole
exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
hole = [(2, 2), (2, 8), (8, 8), (8, 2)]
polygon = Polygon(exterior, [hole])

vertices, faces = triangulate_polygon(polygon)
# Triangles inside the hole are automatically excluded
```

### Mesh Refinement

```python
# Refine mesh with quality constraints
vertices, faces = triangulate_polygon(
    polygon,
    max_area=0.5,      # Maximum triangle area
    min_angle=25.0     # Minimum angle in degrees
)
```

### Subdomain Markers

```python
# Create mesh with subdomain markers for multi-material simulations
exterior = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
subdomain1 = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
subdomain2 = Polygon([(6, 6), (9, 6), (9, 9), (6, 9)])

vertices, faces, subdomain_ids = triangulate_polygon(
    exterior,
    subdomains=[subdomain1, subdomain2],
    return_subdomain_ids=True
)
# subdomain_ids: (M,) array where each element is the subdomain index (0, 1, ...) or -1
```

## Features

- **Constrained Delaunay Triangulation (CDT)** using Rust spade library
- **Holes support** - automatically excludes triangles inside holes
- **Subdomain markers** - tag triangles by region for multi-material FEM simulations
- **Mesh refinement** - control triangle quality with `max_area` and `min_angle` parameters

## License

MIT License. See [LICENSE](LICENSE) for details.
