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
- Exterior boundary and holes both added as closed constraint edge loops
- Calls `cdt.refine()` with `exclude_outer_faces(true)` to classify faces via flood-fill
- Faces reachable from outer face without crossing constraint edges are excluded
- This automatically excludes both exterior faces and faces inside holes

**Refinement Parameters:**
- Default `angle_limit` is 30° in spade - we set it to 0° to disable by default
- Only apply user-specified `max_area` and `min_angle` when provided
- Always call `refine()` even without user params to get face classification

**Face Filtering:**
- Extract `excluded_faces` from `RefinementResult`
- Filter out faces in the excluded set when building output
- Creates vertex index mapping (spade handles → sequential 0,1,2... indices)

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
