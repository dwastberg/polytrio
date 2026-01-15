use geo::{Coord, LineString, Polygon};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySequence};

/// Extract geo::Polygon from Python object via __geo_interface__
pub fn extract_polygon(obj: &Bound<PyAny>) -> PyResult<Polygon<f64>> {
    // Get __geo_interface__ attribute
    let geo_interface = obj.getattr("__geo_interface__")?;

    // Extract the dictionary
    let geo_dict = geo_interface.downcast::<PyDict>()?;

    // Get the "type" field to verify it's a Polygon
    let geom_type: String = geo_dict.get_item("type")?.unwrap().extract()?;
    if geom_type != "Polygon" {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            format!("Expected Polygon, got {}", geom_type)
        ));
    }

    // Get the "coordinates" field
    let coordinates_item = geo_dict
        .get_item("coordinates")?
        .unwrap();
    let coordinates = coordinates_item.downcast::<PySequence>()?;

    // Parse exterior ring (first element)
    let exterior_item = coordinates.get_item(0)?;
    let exterior_seq = exterior_item.downcast::<PySequence>()?;
    let exterior_coords: Vec<Coord<f64>> = extract_coords(exterior_seq)?;
    let exterior_ring = LineString::new(exterior_coords);

    // Parse interior rings (holes) if any
    let mut interior_rings = Vec::new();
    for i in 1..coordinates.len()? {
        let hole_item = coordinates.get_item(i)?;
        let hole_seq = hole_item.downcast::<PySequence>()?;
        let hole_coords: Vec<Coord<f64>> = extract_coords(hole_seq)?;
        interior_rings.push(LineString::new(hole_coords));
    }

    Ok(Polygon::new(exterior_ring, interior_rings))
}

/// Extract coordinates from a Python sequence of (x, y) pairs
fn extract_coords(coords_seq: &Bound<PySequence>) -> PyResult<Vec<Coord<f64>>> {
    let len = coords_seq.len()?;
    let mut coords = Vec::with_capacity(len);

    for i in 0..len {
        let item = coords_seq.get_item(i)?;
        let pair = item.downcast::<PySequence>()?;
        let x: f64 = pair.get_item(0)?.extract()?;
        let y: f64 = pair.get_item(1)?.extract()?;
        coords.push(Coord { x, y });
    }

    Ok(coords)
}

/// Extract optional list of polygons
pub fn extract_polygons(objs: Option<Vec<Bound<PyAny>>>) -> PyResult<Vec<Polygon<f64>>> {
    match objs {
        Some(list) => list.iter().map(|obj| extract_polygon(obj)).collect(),
        None => Ok(Vec::new()),
    }
}
