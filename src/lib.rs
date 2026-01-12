use std::collections::{HashMap, HashSet};

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use spade::{
    AngleLimit, ConstrainedDelaunayTriangulation, Point2, RefinementParameters, Triangulation,
};

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
#[pyo3(signature = (vertices, holes=None, max_area=None, min_angle=None))]
fn triangulate<'py>(
    py: Python<'py>,
    vertices: Vec<(f64, f64)>,
    holes: Option<Vec<Vec<(f64, f64)>>>,
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
    cdt.add_constraint_edges(exterior_points, true)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;

    // Add constraint edges for each hole
    if let Some(hole_list) = holes {
        for hole_vertices in hole_list {
            let hole_points: Vec<Point2<f64>> = hole_vertices
                .iter()
                .map(|(x, y)| Point2::new(*x, *y))
                .collect();

            cdt.add_constraint_edges(hole_points, true)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
        }
    }

    // Always call refine with exclude_outer_faces to get face classification
    // This uses spade's internal flood-fill to identify outer faces
    // Default angle_limit is 30 degrees - set to 0 to disable refinement when not requested
    let mut params = RefinementParameters::<f64>::new()
        .exclude_outer_faces(true)
        .with_angle_limit(AngleLimit::from_deg(0.0)); // Disable angle-based refinement by default

    if let Some(area) = max_area {
        params = params.with_max_allowed_area(area);
    }

    if let Some(angle) = min_angle {
        params = params.with_angle_limit(AngleLimit::from_deg(angle));
    }

    let refinement_result = cdt.refine(params);

    // Collect excluded (outer) faces into a HashSet for fast lookup
    let excluded_faces: HashSet<_> = refinement_result.excluded_faces.iter().collect();

    // Build vertex index mapping (handle -> sequential index)
    let mut vertex_map: HashMap<_, usize> = HashMap::new();
    let mut out_vertices: Vec<[f64; 2]> = Vec::new();

    for vertex in cdt.vertices() {
        let pos = vertex.position();
        let idx = out_vertices.len();
        vertex_map.insert(vertex.fix(), idx);
        out_vertices.push([pos.x, pos.y]);
    }

    // Extract only interior faces (not in excluded_faces set)
    let mut out_faces: Vec<[u32; 3]> = Vec::new();

    for face in cdt.inner_faces() {
        // Skip faces that are outside the polygon (in excluded set)
        if excluded_faces.contains(&face.fix()) {
            continue;
        }

        let face_vertices = face.vertices();
        let i0 = vertex_map[&face_vertices[0].fix()] as u32;
        let i1 = vertex_map[&face_vertices[1].fix()] as u32;
        let i2 = vertex_map[&face_vertices[2].fix()] as u32;
        out_faces.push([i0, i1, i2]);
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
