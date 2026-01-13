use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use spade::{
    AngleLimit, ConstrainedDelaunayTriangulation, Point2, RefinementParameters, Triangulation,
};

/// Point-in-polygon test using ray casting algorithm.
/// Only used when subdomains are present (spade's face classification can't handle them).
fn is_point_in_polygon(point: Point2<f64>, polygon: &[Point2<f64>]) -> bool {
    let mut inside = false;
    let n = polygon.len();
    let x = point.x;
    let y = point.y;

    for i in 0..n {
        let j = (i + 1) % n;
        let p1 = polygon[i];
        let p2 = polygon[j];

        if ((p1.y > y) != (p2.y > y))
            && (x < (p2.x - p1.x) * (y - p1.y) / (p2.y - p1.y) + p1.x)
        {
            inside = !inside;
        }
    }
    inside
}

/// Bounding box for spatial indexing of subdomains.
#[derive(Clone, Copy, Debug)]
struct BoundingBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl BoundingBox {
    fn from_polygon(points: &[Point2<f64>]) -> Self {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for pt in points {
            min_x = min_x.min(pt.x);
            min_y = min_y.min(pt.y);
            max_x = max_x.max(pt.x);
            max_y = max_y.max(pt.y);
        }

        BoundingBox { min_x, min_y, max_x, max_y }
    }

    fn contains_point(&self, pt: &Point2<f64>) -> bool {
        pt.x >= self.min_x && pt.x <= self.max_x &&
        pt.y >= self.min_y && pt.y <= self.max_y
    }
}

/// Spatial grid for efficient subdomain lookup.
/// Uses uniform grid partitioning for O(1) point queries.
struct SpatialGrid {
    cells: Vec<Vec<usize>>,  // Each cell contains subdomain indices
    grid_size: usize,
    bbox: BoundingBox,
    cell_width: f64,
    cell_height: f64,
}

impl SpatialGrid {
    fn query_point(&self, pt: &Point2<f64>) -> Vec<usize> {
        // Find grid cell containing the point
        let col = ((pt.x - self.bbox.min_x) / self.cell_width)
            .floor()
            .max(0.0)
            .min((self.grid_size - 1) as f64) as usize;

        let row = ((pt.y - self.bbox.min_y) / self.cell_height)
            .floor()
            .max(0.0)
            .min((self.grid_size - 1) as f64) as usize;

        let cell_idx = row * self.grid_size + col;
        self.cells[cell_idx].clone()
    }
}

/// Build a spatial grid index for efficient subdomain lookup.
fn build_spatial_grid(
    bboxes: &[BoundingBox],
    global_bbox: BoundingBox,
    grid_size: usize,
) -> SpatialGrid {
    let cell_width = (global_bbox.max_x - global_bbox.min_x) / grid_size as f64;
    let cell_height = (global_bbox.max_y - global_bbox.min_y) / grid_size as f64;

    let mut cells = vec![Vec::new(); grid_size * grid_size];

    // Insert each subdomain into all grid cells it overlaps
    for (subdomain_idx, bbox) in bboxes.iter().enumerate() {
        let min_col = ((bbox.min_x - global_bbox.min_x) / cell_width)
            .floor()
            .max(0.0) as usize;
        let max_col = ((bbox.max_x - global_bbox.min_x) / cell_width)
            .ceil()
            .min(grid_size as f64) as usize;

        let min_row = ((bbox.min_y - global_bbox.min_y) / cell_height)
            .floor()
            .max(0.0) as usize;
        let max_row = ((bbox.max_y - global_bbox.min_y) / cell_height)
            .ceil()
            .min(grid_size as f64) as usize;

        for row in min_row..max_row {
            for col in min_col..max_col {
                let cell_idx = row * grid_size + col;
                cells[cell_idx].push(subdomain_idx);
            }
        }
    }

    SpatialGrid {
        cells,
        grid_size,
        bbox: global_bbox,
        cell_width,
        cell_height,
    }
}

/// Compute the global bounding box encompassing all subdomain bounding boxes.
fn compute_global_bbox(bboxes: &[BoundingBox]) -> BoundingBox {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for bbox in bboxes {
        min_x = min_x.min(bbox.min_x);
        min_y = min_y.min(bbox.min_y);
        max_x = max_x.max(bbox.max_x);
        max_y = max_y.max(bbox.max_y);
    }

    BoundingBox { min_x, min_y, max_x, max_y }
}

/// Triangulate a polygon defined by its boundary vertices.
///
/// # Arguments
/// * `vertices` - List of (x, y) coordinates defining the polygon exterior boundary
/// * `holes` - Optional list of hole boundaries, each a list of (x, y) coordinates
/// * `max_area` - Optional maximum triangle area for mesh refinement
/// * `min_angle` - Optional minimum angle in degrees for mesh refinement
///
/// # Returns
/// A tuple of (vertices, faces, subdomain_ids) where:
/// * vertices is an Nx2 array of float64 coordinates
/// * faces is an Mx3 array of uint32 vertex indices
/// * subdomain_ids is an optional Mx1 array of int32 subdomain IDs (if return_subdomain_ids=True)
#[pyfunction]
#[pyo3(signature = (vertices, holes=None, subdomains=None, max_area=None, min_angle=None, return_subdomain_ids=None))]
fn triangulate<'py>(
    py: Python<'py>,
    vertices: Vec<(f64, f64)>,
    holes: Option<Vec<Vec<(f64, f64)>>>,
    subdomains: Option<Vec<Vec<(f64, f64)>>>,
    max_area: Option<f64>,
    min_angle: Option<f64>,
    return_subdomain_ids: Option<bool>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<u32>>,
    Option<Bound<'py, PyArray1<i32>>>,
)> {
    // Create constrained Delaunay triangulation
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();

    // Convert exterior to Point2 and add as constraint edges (closed polygon)
    let exterior_points: Vec<Point2<f64>> = vertices
        .iter()
        .map(|(x, y)| Point2::new(*x, *y))
        .collect();

    // Add constraint edges for exterior boundary
    // Clone exterior_points because we may need it later for manual face filtering
    cdt.add_constraint_edges(exterior_points.clone(), true)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;

    // Add constraint edges for each hole
    if let Some(hole_list) = &holes {
        for hole_vertices in hole_list {
            let hole_points: Vec<Point2<f64>> = hole_vertices
                .iter()
                .map(|(x, y)| Point2::new(*x, *y))
                .collect();

            cdt.add_constraint_edges(hole_points, true)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
        }
    }

    // Add constraint edges for each subdomain
    if let Some(ref subdomain_list) = subdomains {
        for subdomain_vertices in subdomain_list {
            let subdomain_points: Vec<Point2<f64>> = subdomain_vertices
                .iter()
                .map(|(x, y)| Point2::new(*x, *y))
                .collect();

            cdt.add_constraint_edges(subdomain_points, true)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
        }
    }

    // Store subdomain polygons for optional subdomain ID assignment
    let subdomain_polygons: Vec<Vec<Point2<f64>>> = if let Some(ref subdomain_list) = subdomains {
        subdomain_list
            .iter()
            .map(|subdomain_vertices| {
                subdomain_vertices
                    .iter()
                    .map(|(x, y)| Point2::new(*x, *y))
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };

    // Build spatial index for efficient subdomain lookup
    // For large numbers of subdomains (>100), use a grid-based spatial index
    // For smaller counts, simple bounding box filtering is faster
    let subdomain_bboxes: Vec<BoundingBox> = subdomain_polygons
        .iter()
        .map(|poly| BoundingBox::from_polygon(poly))
        .collect();

    let use_spatial_grid = subdomain_polygons.len() > 100;
    let spatial_grid = if use_spatial_grid {
        let global_bbox = compute_global_bbox(&subdomain_bboxes);
        let grid_size = (subdomain_polygons.len() as f64).sqrt().ceil() as usize;
        Some(build_spatial_grid(&subdomain_bboxes, global_bbox, grid_size))
    } else {
        None
    };

    // Face classification strategy depends on whether subdomains are present:
    // - NO subdomains: Use spade's exclude_outer_faces (fast, topologically correct)
    // - WITH subdomains: Cannot use spade's classification (it can't distinguish subdomains from holes)
    //   Instead, use manual point-in-polygon filtering later
    let has_subdomains = subdomains.is_some();

    let mut params = RefinementParameters::<f64>::new()
        .exclude_outer_faces(!has_subdomains)  // Only enable if no subdomains
        .with_angle_limit(AngleLimit::from_deg(0.0));

    if let Some(area) = max_area {
        params = params.with_max_allowed_area(area);
    }

    if let Some(angle) = min_angle {
        params = params.with_angle_limit(AngleLimit::from_deg(angle));
    }

    // Refine the mesh (with or without face classification)
    let result = if has_subdomains || max_area.is_some() || min_angle.is_some() {
        cdt.refine(params)
    } else {
        cdt.refine(params)  // Always refine for face classification when no subdomains
    };

    let excluded_faces = if !has_subdomains {
        result.excluded_faces
    } else {
        Vec::new()  // Will use manual filtering instead
    };

    // Build vertex index mapping (handle -> sequential index)
    let mut vertex_map: HashMap<_, usize> = HashMap::new();
    let mut out_vertices: Vec<[f64; 2]> = Vec::new();

    for vertex in cdt.vertices() {
        let pos = vertex.position();
        let idx = out_vertices.len();
        vertex_map.insert(vertex.fix(), idx);
        out_vertices.push([pos.x, pos.y]);
    }

    // Extract only interior faces - strategy depends on whether subdomains are present
    let mut out_faces: Vec<[u32; 3]> = Vec::new();

    if !has_subdomains {
        // Fast path: Use spade's face classification (no subdomains)
        let excluded_set: HashSet<_> = excluded_faces.iter().cloned().collect();
        for face in cdt.inner_faces() {
            if excluded_set.contains(&face.fix()) {
                continue;
            }
            let face_vertices = face.vertices();
            let i0 = vertex_map[&face_vertices[0].fix()] as u32;
            let i1 = vertex_map[&face_vertices[1].fix()] as u32;
            let i2 = vertex_map[&face_vertices[2].fix()] as u32;
            out_faces.push([i0, i1, i2]);
        }
    } else {
        // Slow path: Manual point-in-polygon filtering (subdomains present)
        // Prepare hole polygons for checking
        let mut hole_polygons: Vec<Vec<Point2<f64>>> = Vec::new();
        if let Some(hole_list) = &holes {
            for hole_vertices in hole_list {
                hole_polygons.push(
                    hole_vertices
                        .iter()
                        .map(|(x, y)| Point2::new(*x, *y))
                        .collect(),
                );
            }
        }

        // Filter faces: must be inside exterior AND not inside any hole
        // Note: Subdomains are NOT checked - they only create constraint edges
        for face in cdt.inner_faces() {
            let center = face.center();

            // Check 1: Must be inside exterior
            if !is_point_in_polygon(center, &exterior_points) {
                continue;
            }

            // Check 2: Must NOT be inside any hole
            let mut in_hole = false;
            for hole in &hole_polygons {
                if is_point_in_polygon(center, hole) {
                    in_hole = true;
                    break;
                }
            }
            if in_hole {
                continue;
            }

            let face_vertices = face.vertices();
            let i0 = vertex_map[&face_vertices[0].fix()] as u32;
            let i1 = vertex_map[&face_vertices[1].fix()] as u32;
            let i2 = vertex_map[&face_vertices[2].fix()] as u32;
            out_faces.push([i0, i1, i2]);
        }
    }

    // Optionally compute subdomain IDs for each face
    let subdomain_ids = if return_subdomain_ids.unwrap_or(false) && !subdomain_polygons.is_empty() {
        let mut ids = Vec::with_capacity(out_faces.len());

        for face in &out_faces {
            // Calculate face centroid from its 3 vertices
            let v0 = out_vertices[face[0] as usize];
            let v1 = out_vertices[face[1] as usize];
            let v2 = out_vertices[face[2] as usize];
            let centroid = Point2::new(
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
            );

            // Find which subdomain (if any) contains this face
            // Use spatial indexing to narrow down candidates (O(log n) or O(1))
            let mut subdomain_id: i32 = -1;  // -1 = not in any subdomain

            let candidates: Vec<usize> = if let Some(ref grid) = spatial_grid {
                // Large number of subdomains: use spatial grid (O(1) query)
                grid.query_point(&centroid)
            } else {
                // Small number of subdomains: filter by bounding box (O(n) but fast)
                (0..subdomain_polygons.len())
                    .filter(|&idx| subdomain_bboxes[idx].contains_point(&centroid))
                    .collect()
            };

            // Test only the candidate subdomains (typically 1-3 instead of thousands)
            for &idx in &candidates {
                if is_point_in_polygon(centroid, &subdomain_polygons[idx]) {
                    subdomain_id = idx as i32;
                    break;  // Assume non-overlapping subdomains
                }
            }
            ids.push(subdomain_id);
        }
        Some(ids)
    } else {
        None
    };

    // Convert to numpy arrays
    let n_verts = out_vertices.len();
    let n_faces = out_faces.len();

    let vertices_flat: Vec<f64> = out_vertices.into_iter().flatten().collect();
    let faces_flat: Vec<u32> = out_faces.into_iter().flatten().collect();

    let vertices_array = Array2::from_shape_vec((n_verts, 2), vertices_flat)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let faces_array = Array2::from_shape_vec((n_faces, 3), faces_flat)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Convert subdomain IDs to numpy array if requested
    let subdomain_ids_array = if let Some(ids) = subdomain_ids {
        let ids_array = Array1::from_vec(ids);
        Some(ids_array.into_pyarray(py))
    } else {
        None
    };

    Ok((
        vertices_array.into_pyarray(py),
        faces_array.into_pyarray(py),
        subdomain_ids_array,
    ))
}

/// PySpade: Python bindings for the Spade 2D meshing library
#[pymodule]
fn _pyspade(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(triangulate, m)?)?;
    Ok(())
}
