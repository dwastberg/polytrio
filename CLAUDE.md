# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PySpade is a Python library that provides bindings to the Rust `spade` library for 2D Delaunay triangulation and mesh generation. It's a hybrid Rust-Python project using:
- **PyO3** for Rust-Python bindings
- **Maturin** for building Python wheels
- **rust-numpy** for efficient numpy array interchange
- **Spade** for constrained Delaunay triangulation (CDT)

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
- Takes raw coordinate lists: `vertices`, `holes`, `max_area`, `min_angle`
- Returns numpy arrays: `(vertices, faces)`
- Handles all spade CDT operations and mesh refinement

**Python Layer** (`pyspade/__init__.py`):
- High-level `triangulate_polygon()` function
- Accepts shapely `Polygon` objects (with automatic hole extraction)
- Validates polygons using `polygon.is_valid` before processing
- Thin wrapper that extracts coordinates and calls Rust layer

### Key Implementation Details

**Handling Non-Convex Polygons & Holes:**
- Uses spade's `ConstrainedDelaunayTriangulation` (CDT) with constraint edges
- Exterior boundary, holes, and subdomains all added as closed constraint edge loops
- Face classification uses a **hybrid approach** depending on subdomain presence

**Hybrid Face Classification Strategy:**
- **Without subdomains** (fast path):
  - Calls `cdt.refine()` with `exclude_outer_faces(true)` to classify faces via flood-fill
  - Extracts `excluded_faces` from `RefinementResult` (O(f + e) complexity)
  - Filters faces using simple HashSet lookup (~50x faster than manual filtering)
  - Automatically excludes exterior faces and faces inside holes

- **With subdomains** (compatibility path):
  - Calls `cdt.refine()` with `exclude_outer_faces(false)` to avoid misclassifying subdomains
  - Uses manual point-in-polygon ray casting to filter faces (O(f * (n + h*m)) complexity)
  - Checks: face centroid must be inside exterior AND not inside any hole
  - Subdomains are NOT checked - they only create constraint edges without excluding faces
  - Required because spade's flood-fill can't distinguish subdomains (internal boundaries) from holes (exclusion boundaries)

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
- Uses spatial indexing (grid-based) for efficient subdomain lookup
- Complexity: O(s log s + f * (log s + k * v))
  - s = subdomains, f = faces, k = candidates per face (typically 1-3), v = vertices per subdomain
- Adaptive strategy:
  - ≤100 subdomains: Simple bounding box filtering (O(s) but very fast)
  - >100 subdomains: Uniform grid spatial index (O(1) query)
- Benchmark results (Apple Silicon M-series):
  - 900 subdomains + 78,873 faces: **0.05 seconds** (~1.6M faces/sec)
  - 100 subdomains + 8,731 faces: **0.01 seconds**
  - 9 subdomains + 799 faces: **< 0.01 seconds**
- Estimated 1,000 subdomains + 1,000,000 faces: ~1 second (vs 3 hours with naive O(f*s*v) approach)
- Memory overhead: ~24 KB for 1000 subdomains (negligible)

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
pyspade/__init__.py  # Python wrapper API
tests/               # Pytest test suite
visualize/           # Matplotlib visualization examples
```

## Testing Strategy

Tests are organized into classes:
- `TestTriangulatePolygon` - Basic functionality
- `TestRefinement` - Mesh refinement parameters
- `TestTriangulateDirect` - Low-level API
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
