use std::collections::HashMap;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use spade::{
    AngleLimit, ConstrainedDelaunayTriangulation, Point2, RefinementParameters, Triangulation,
};

/// Check if a point is inside a polygon using ray casting algorithm
fn point_in_polygon(point: (f64, f64), polygon: &[(f64, f64)]) -> bool {
    let (px, py) = point;
    let n = polygon.len();
    let mut inside = false;

    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];

        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Triangulate a polygon defined by its boundary vertices.
///
/// # Arguments
/// * `vertices` - List of (x, y) coordinates defining the polygon boundary
/// * `max_area` - Optional maximum triangle area for mesh refinement
/// * `min_angle` - Optional minimum angle in degrees for mesh refinement
///
/// # Returns
/// A tuple of (vertices, faces) where:
/// * vertices is an Nx2 array of float64 coordinates
/// * faces is an Mx3 array of uint32 vertex indices
#[pyfunction]
#[pyo3(signature = (vertices, max_area=None, min_angle=None))]
fn triangulate<'py>(
    py: Python<'py>,
    vertices: Vec<(f64, f64)>,
    max_area: Option<f64>,
    min_angle: Option<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<u32>>)> {
    // Create constrained Delaunay triangulation
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();

    // Convert to Point2 and add as constraint edges (closed polygon)
    let points: Vec<Point2<f64>> = vertices
        .iter()
        .map(|(x, y)| Point2::new(*x, *y))
        .collect();

    // Add constraint edges to form a closed polygon
    cdt.add_constraint_edges(points, true)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;

    // Apply mesh refinement if parameters are provided
    if max_area.is_some() || min_angle.is_some() {
        let mut params = RefinementParameters::<f64>::new().exclude_outer_faces(true);

        if let Some(area) = max_area {
            params = params.with_max_allowed_area(area);
        }

        if let Some(angle) = min_angle {
            params = params.with_angle_limit(AngleLimit::from_deg(angle));
        }

        cdt.refine(params);
    }

    // Build vertex index mapping (handle -> sequential index)
    let mut vertex_map: HashMap<_, usize> = HashMap::new();
    let mut out_vertices: Vec<[f64; 2]> = Vec::new();

    for vertex in cdt.vertices() {
        let pos = vertex.position();
        let idx = out_vertices.len();
        vertex_map.insert(vertex.fix(), idx);
        out_vertices.push([pos.x, pos.y]);
    }

    // Extract only interior faces (centroid must be inside the input polygon)
    let mut out_faces: Vec<[u32; 3]> = Vec::new();

    for face in cdt.inner_faces() {
        let face_vertices = face.vertices();
        let p0 = face_vertices[0].position();
        let p1 = face_vertices[1].position();
        let p2 = face_vertices[2].position();

        // Compute centroid of the triangle
        let centroid = ((p0.x + p1.x + p2.x) / 3.0, (p0.y + p1.y + p2.y) / 3.0);

        // Only include face if centroid is inside the input polygon
        if point_in_polygon(centroid, &vertices) {
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
