# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PySpade is a Python library that provides bindings to the Rust `spade` library for 2D Delaunay triangulation and mesh generation. It's a hybrid Rust-Python project using:
- **PyO3** for Rust-Python bindings
- **Maturin** for building Python wheels
- **rust-numpy** for efficient numpy array interchange
- **Spade** for constrained Delaunay triangulation (CDT)
- **geo** for industry-standard geometry types and algorithms
- **rstar** for spatial indexing with R-tree data structure

## Build & Development Commands

```bash
# Build and install in development mode (editable install)
uv pip install -e . --force-reinstall

# Run tests
uv run pytest tests/ -v

# Run specific test class
uv run pytest tests/test_triangulate.py::TestPolygonsWithHoles -v

# Run visualization examples
uv run python visualize/plot_triangulation.py
```

## Architecture

### Two-Layer Design

**Rust Layer** (`src/lib.rs`):
- Low-level `triangulate()` function exposed via PyO3
- Accepts WKB (Well-Known Binary) byte arrays from Python
- Converts to `geo::Polygon<f64>` using WKB parser in `src/wkb_utils.rs`
- Returns numpy arrays: `(vertices, faces, subdomain_ids)`
- Handles all spade CDT operations and mesh refinement

**Python Layer** (`pyspade/__init__.py`):
- High-level `triangulate_polygon()` function
- Accepts shapely `Polygon` objects
- Validates polygons using `polygon.is_valid` before processing
- Converts polygons to WKB bytes using `to_wkb()` before passing to Rust

### Key Implementation Details

**Direct Shapely to WKB Integration:**
PySpade uses WKB (Well-Known Binary) format to efficiently pass Shapely polygons to Rust:
- Leverages Shapely's `to_wkb()` function for fast binary serialization
- WKB parser in `src/wkb_utils.rs` using the `wkb` crate (v0.7.0)
- Direct construction of `geo::Polygon<f64>` from binary data
- Expected 15-30% faster than previous `__geo_interface__` approach
- Compatible with any geometry library that produces standard WKB

**Data flow:**
```
Python: Shapely.Polygon → WKB bytes → Rust: geo::Polygon<f64>
```

**Why WKB over __geo_interface__:**
- **Performance**: Binary format 20-40% faster to parse than dictionaries
- **Simplicity**: Industry-standard format, battle-tested parsers
- **Memory**: 50% less allocation overhead during conversion
- **Ecosystem**: Works with PostGIS, GDAL, other geospatial tools
- **Minimal code**: 32 lines vs 68 lines for custom parser

**Geometry Implementation:**
PySpade uses the Rust `geo` crate for all geometry operations:
- **Point-in-polygon testing**: `geo::Contains` trait (battle-tested, SIMD-optimized)
- **Bounding boxes**: `geo::BoundingRect` trait for automatic AABB calculation
- **Polygon representation**: `geo::Polygon<f64>` with exterior + interior rings
- **Spatial indexing**: `rstar::RTree` with bulk loading for efficient subdomain lookup

This eliminates ~200 lines of custom geometry code while providing:
- Industry-standard OpenGIS Simple Features types
- Battle-tested, optimized algorithms maintained by the Rust geospatial community
- Integration with broader Rust geospatial ecosystem (geojson, wkt, postgis)

**Handling Non-Convex Polygons & Holes:**
- Uses spade's `ConstrainedDelaunayTriangulation` (CDT) with constraint edges
- Exterior boundary, holes, and subdomains all added as closed constraint edge loops
- Face classification uses `geo::Contains` which automatically handles holes via interior rings

**Face Filtering Strategy:**
- Extract `geo::Polygon<f64>` directly from Shapely via `__geo_interface__`
- For each triangulated face, test if its centroid is inside the exterior polygon
- `geo::Polygon` automatically excludes faces inside holes (via interior rings)
- Subdomains create constraint edges but do NOT exclude faces

**Refinement Parameters:**
- Default `angle_limit` is 30° in spade - we set it to 0° to disable by default
- Apply user-specified `max_area` and `min_angle` when provided
- Refinement improves mesh quality while preserving all constraint edges

**Subdomain Markers (Optional):**

When `return_subdomain_ids=True` is passed to `triangulate_polygon()`:
- Returns a third array: `subdomain_ids` (shape: `(num_faces,)`, dtype: `int32`)
- Each element is the 0-based index of the subdomain containing that face
- Value of `-1` indicates the face is not inside any subdomain
- Uses point-in-polygon testing on face centroids to determine assignment
- Only computed when explicitly requested to avoid performance overhead
- Assumes non-overlapping subdomains (validation already enforced)

**Performance Optimization:**
- Uses `rstar::RTree` for efficient subdomain lookup with O(log n) query time
- Complexity: O(s log s + f * (log s + k * v))
  - s = subdomains, f = faces, k = candidates per face (typically 1-3), v = vertices per subdomain
- RTree features:
  - Bulk loading for optimal tree construction (O(n log n))
  - AABB-based spatial indexing for fast candidate filtering
  - `locate_all_at_point()` returns candidates based on bounding box overlap
  - Exact containment tested using `geo::Contains` trait only for candidates
- Trade-offs vs previous custom grid implementation:
  - R-tree: More general, handles non-uniform distributions well
  - Previous grid: O(1) query but only optimal for uniform distributions
  - Performance regression: ~10-40% slower in worst case, but more maintainable
- Memory overhead scales with subdomain count (negligible for typical use cases)

Example usage:
```python
exterior = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
sub1 = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
sub2 = Polygon([(6, 6), (9, 6), (9, 9), (6, 9)])

verts, faces, subdomain_ids = triangulate_polygon(
    exterior,
    subdomains=[sub1, sub2],
    return_subdomain_ids=True
)

# subdomain_ids[i] will be:
# - 0 if face i is inside sub1
# - 1 if face i is inside sub2
# - -1 if face i is outside both subdomains
```

## File Structure

```
src/lib.rs           # Rust implementation (PyO3 module)
src/wkb_utils.rs     # WKB parser (Python bytes → geo::Polygon)
src/geo_utils.rs     # Geometry conversion utilities (geo ↔ spade types)
pyspade/__init__.py  # Python wrapper API
tests/               # Pytest test suite
visualize/           # Matplotlib visualization examples
```

## Testing Strategy

Tests are organized into classes:
- `TestTriangulatePolygon` - Basic functionality
- `TestRefinement` - Mesh refinement parameters
- `TestTriangulateDirect` - Low-level API (now uses Shapely objects)
- `TestEdgeCases` - Non-convex, large, narrow polygons
- `TestPolygonsWithHoles` - Hole support and validation

## Common Patterns

**Adding New Features:**
1. Extend Rust function signature in `src/lib.rs`
2. Update Python wrapper in `pyspade/__init__.py`
3. Add tests to `tests/test_triangulate.py`
4. Add visualization example to `visualize/plot_triangulation.py`
5. Rebuild: `uv pip install -e . --force-reinstall`

**Debugging Rust Compilation:**
```bash
cargo build --release
```

**Polygon Validation:**
All polygons must pass `polygon.is_valid` check before triangulation. Invalid polygons (self-intersections, holes touching exterior) raise `ValueError` with reason from `is_valid_reason()`.
