use geo::{Geometry, Polygon};
use pyo3::prelude::*;
use wkb::wkb_to_geom;

/// Extract geo::Polygon from WKB bytes
pub fn extract_polygon(wkb_bytes: &[u8]) -> PyResult<Polygon<f64>> {
    // Parse WKB to geo::Geometry
    let geometry: Geometry<f64> = wkb_to_geom(&mut &wkb_bytes[..])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Failed to parse WKB: {:?}", e)
        ))?;

    // Extract Polygon from Geometry enum
    match geometry {
        Geometry::Polygon(polygon) => Ok(polygon),
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected Polygon geometry, got different type"
        )),
    }
}
