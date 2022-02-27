use {
    core::ops::{Mul, Neg, Sub},
    mint::Point2,
    num_traits::Float,
};

pub fn radians<A, T>(angle: A) -> impl Fn(Point2<T>) -> <T::Output as Sub>::Output + Copy
where
    A: Float,
    T: Mul<A>,
    T::Output: Sub,
{
    let (sin, cos) = angle.sin_cos();
    move |p| {
        let Point2 { x, y } = p;
        x * sin - y * cos
    }
}

#[inline]
pub fn degrees<A, T>(angle: A) -> impl Fn(Point2<T>) -> <T::Output as Sub>::Output + Copy
where
    A: Float,
    T: Mul<A>,
    T::Output: Sub,
{
    radians(angle.to_radians())
}

#[inline]
pub fn up<T>(p: Point2<T>) -> T {
    p.x
}

#[inline]
pub fn down<T>(p: Point2<T>) -> T::Output
where
    T: Neg,
{
    -p.x
}

#[inline]
pub fn left<T>(p: Point2<T>) -> T {
    p.y
}

#[inline]
pub fn right<T>(p: Point2<T>) -> T::Output
where
    T: Neg,
{
    -p.y
}
