use geo::{LineString, Polygon};
use spade::Point2;

/// Convert geo::Polygon to Vec<Point2<f64>> for spade
pub fn polygon_to_spade_points(polygon: &Polygon<f64>) -> Vec<Point2<f64>> {
    polygon
        .exterior()
        .coords()
        .map(|c| Point2::new(c.x, c.y))
        .collect()
}

/// Convert LineString to Vec<Point2<f64>> for spade
pub fn linestring_to_spade_points(ls: &LineString<f64>) -> Vec<Point2<f64>> {
    ls.coords()
        .map(|c| Point2::new(c.x, c.y))
        .collect()
}
