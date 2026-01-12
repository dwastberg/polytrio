use std::collections::{HashMap, HashSet};

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
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

/// Triangulate a polygon defined by its boundary vertices.
///
/// # Arguments
/// * `vertices` - List of (x, y) coordinates defining the polygon exterior boundary
/// * `holes` - Optional list of hole boundaries, each a list of (x, y) coordinates
/// * `max_area` - Optional maximum triangle area for mesh refinement
/// * `min_angle` - Optional minimum angle in degrees for mesh refinement
///
/// # Returns
/// A tuple of (vertices, faces) where:
/// * vertices is an Nx2 array of float64 coordinates
/// * faces is an Mx3 array of uint32 vertex indices
#[pyfunction]
#[pyo3(signature = (vertices, holes=None, subdomains=None, max_area=None, min_angle=None))]
fn triangulate<'py>(
    py: Python<'py>,
    vertices: Vec<(f64, f64)>,
    holes: Option<Vec<Vec<(f64, f64)>>>,
    subdomains: Option<Vec<Vec<(f64, f64)>>>,
    max_area: Option<f64>,
    min_angle: Option<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<u32>>)> {
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

    // Convert to numpy arrays
    let n_verts = out_vertices.len();
    let n_faces = out_faces.len();

    let vertices_flat: Vec<f64> = out_vertices.into_iter().flatten().collect();
    let faces_flat: Vec<u32> = out_faces.into_iter().flatten().collect();

    let vertices_array = Array2::from_shape_vec((n_verts, 2), vertices_flat)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let faces_array = Array2::from_shape_vec((n_faces, 3), faces_flat)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((
        vertices_array.into_pyarray(py),
        faces_array.into_pyarray(py),
    ))
}

/// PySpade: Python bindings for the Spade 2D meshing library
#[pymodule]
fn _pyspade(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(triangulate, m)?)?;
    Ok(())
}
