use early_ketupa::{dirs, Look, View};

fn main() {
    // Points can be of any type stored in any collection.
    let points = [[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]];

    // Direction functor that would rotate points' coordinates.
    // Provided angle is the "look at" kind.
    let dir = dirs::degrees(90f32);

    // Map of indices to points converted to mint::Point2.
    let map = |i: u16| points[i as usize].into();

    let look = Look::new(map, dir);

    // This takes O(nlogn).
    let view = View::new(look, 0..points.len() as _);
    println!("visible indices: {:?}", view.hull.indices);

    // This takes O(logn).
    let projection = view.proj(dir([0.5, -10.0].into())).unwrap();
    println!("projection adjacent: {:?}", projection.adjacent);
    println!("projection point: {:?}", projection.point);
}
