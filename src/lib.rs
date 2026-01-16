use std::collections::HashMap;

use geo::{BoundingRect, Contains, Coord, Polygon};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use rstar::{RTree, AABB};
use spade::{
    AngleLimit, ConstrainedDelaunayTriangulation, Point2, RefinementParameters, Triangulation,
};

mod geo_utils;
use geo_utils::{linestring_to_spade_points, polygon_to_spade_points};

mod wkb_utils;
use wkb_utils::extract_polygon;

/// Entry for RTree containing subdomain index and its polygon
struct SubdomainEntry {
    index: usize,
    polygon: Polygon<f64>,
}

impl rstar::RTreeObject for SubdomainEntry {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        let rect = self.polygon.bounding_rect().unwrap();
        AABB::from_corners(
            [rect.min().x, rect.min().y],
            [rect.max().x, rect.max().y],
        )
    }
}

impl rstar::PointDistance for SubdomainEntry {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        // Return 0 if point is inside, positive value if outside
        // Note: We use locate_all_at_point which only needs AABB envelope,
        // so precise distance calculation is not critical
        let coord = Coord { x: point[0], y: point[1] };

        if self.polygon.contains(&coord) {
            0.0
        } else {
            // Return a positive value for points outside
            // (Actual distance calculation not needed for locate_all_at_point)
            1.0
        }
    }
}

/// Triangulate a polygon using WKB (Well-Known Binary) format input.
///
/// This function accepts polygon geometries encoded in WKB format,
/// eliminating Python object parsing overhead.
///
/// # Arguments
/// * `polygon` - Polygon geometry as WKB bytes (from Shapely's to_wkb())
/// * `subdomains` - Optional list of subdomain polygons as WKB bytes
/// * `max_area` - Optional maximum triangle area for mesh refinement
/// * `min_angle` - Optional minimum angle in degrees for mesh refinement
/// * `return_subdomain_ids` - Whether to return subdomain ID array
///
/// # Returns
/// A tuple of (vertices, faces, subdomain_ids) where:
/// * vertices is an Nx2 array of float64 coordinates
/// * faces is an Mx3 array of uint32 vertex indices
/// * subdomain_ids is an optional Mx1 array of int32 subdomain IDs (if return_subdomain_ids=True)
#[pyfunction]
#[pyo3(signature = (polygon, subdomains=None, max_area=None, min_angle=None, return_subdomain_ids=None))]
fn triangulate<'py>(
    py: Python<'py>,
    polygon: &[u8],
    subdomains: Option<Vec<Vec<u8>>>,
    max_area: Option<f64>,
    min_angle: Option<f64>,
    return_subdomain_ids: Option<bool>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<u32>>,
    Option<Bound<'py, PyArray1<i32>>>,
)> {
    // Extract geo::Polygon from WKB bytes
    let exterior_polygon = extract_polygon(polygon)?;

    // Extract subdomain polygons from WKB bytes
    let subdomain_polygons = match subdomains {
        Some(wkb_list) => {
            wkb_list.iter()
                .map(|wkb| extract_polygon(wkb.as_slice()))
                .collect::<PyResult<Vec<_>>>()?
        }
        None => Vec::new(),
    };

    // Create constrained Delaunay triangulation
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();

    // Add exterior constraint edges
    let exterior_points = polygon_to_spade_points(&exterior_polygon);
    cdt.add_constraint_edges(exterior_points, true)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;

    // Add interior holes
    for hole_ls in exterior_polygon.interiors() {
        let hole_points = linestring_to_spade_points(hole_ls);
        cdt.add_constraint_edges(hole_points, true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
    }

    // Add constraint edges for each subdomain (exterior and holes)
    for subdomain in &subdomain_polygons {
        // Add exterior
        let exterior_points = polygon_to_spade_points(subdomain);
        cdt.add_constraint_edges(exterior_points, true)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;

        // Add each hole
        for hole_ls in subdomain.interiors() {
            let hole_points = linestring_to_spade_points(hole_ls);
            cdt.add_constraint_edges(hole_points, true)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
        }
    }

    // Build RTree for subdomain lookup
    let rtree: Option<RTree<SubdomainEntry>> = if !subdomain_polygons.is_empty() {
        let entries: Vec<SubdomainEntry> = subdomain_polygons
            .iter()
            .enumerate()
            .map(|(index, polygon)| SubdomainEntry {
                index,
                polygon: polygon.clone(),
            })
            .collect();
        Some(RTree::bulk_load(entries))
    } else {
        None
    };

    // Face classification strategy depends on whether subdomains are present:
    // - NO subdomains: Use spade's exclude_outer_faces (fast, topologically correct)
    // - WITH subdomains: Cannot use spade's classification (it can't distinguish subdomains from holes)
    //   Instead, use manual point-in-polygon filtering later
    let has_subdomains = !subdomain_polygons.is_empty();

    let mut params = RefinementParameters::<f64>::new()
        .exclude_outer_faces(!has_subdomains)  // Only enable if no subdomains
        .with_angle_limit(AngleLimit::from_deg(0.0));

    if let Some(area) = max_area {
        params = params.with_max_allowed_area(area);
    }

    if let Some(angle) = min_angle {
        params = params.with_angle_limit(AngleLimit::from_deg(angle));
    }

    // Refine the mesh
    if max_area.is_some() || min_angle.is_some() {
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

    // Extract only interior faces using geo::Contains
    // geo::Polygon automatically handles holes via interior rings
    let mut out_faces: Vec<[u32; 3]> = Vec::new();

    for face in cdt.inner_faces() {
        let center = face.center();
        let center_coord = Coord { x: center.x, y: center.y };

        // Check if inside exterior polygon (automatically excludes holes)
        if !exterior_polygon.contains(&center_coord) {
            continue;
        }

        let face_vertices = face.vertices();
        let i0 = vertex_map[&face_vertices[0].fix()] as u32;
        let i1 = vertex_map[&face_vertices[1].fix()] as u32;
        let i2 = vertex_map[&face_vertices[2].fix()] as u32;
        out_faces.push([i0, i1, i2]);
    }

    // Optionally compute subdomain IDs for each face using RTree
    let subdomain_ids = if return_subdomain_ids.unwrap_or(false) && !subdomain_polygons.is_empty() {
        let mut ids = Vec::with_capacity(out_faces.len());

        for face in &out_faces {
            // Calculate face centroid from its 3 vertices
            let v0 = out_vertices[face[0] as usize];
            let v1 = out_vertices[face[1] as usize];
            let v2 = out_vertices[face[2] as usize];
            let centroid_point = [
                (v0[0] + v1[0] + v2[0]) / 3.0,
                (v0[1] + v1[1] + v2[1]) / 3.0,
            ];
            let centroid_coord = Coord {
                x: centroid_point[0],
                y: centroid_point[1],
            };

            // Use RTree to find candidates (O(log n) query)
            let mut subdomain_id: i32 = -1;  // -1 = not in any subdomain

            if let Some(ref tree) = rtree {
                // Get all entries at this point (candidates based on AABB overlap)
                for entry in tree.locate_all_at_point(&centroid_point) {
                    // Test exact containment using geo::Contains
                    if entry.polygon.contains(&centroid_coord) {
                        subdomain_id = entry.index as i32;
                        break;  // Assume non-overlapping subdomains
                    }
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
