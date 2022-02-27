pub mod core;
pub mod dirs;
mod utils;

use {
    crate::{core::*, utils::index_norm},
    ::core::{cmp::Ordering::Equal, ops::Range},
    mint::Point2,
    num_traits::{Float, NumAssign, PrimInt},
};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Look<G, D> {
    pub get: G,
    pub dir: D,
}

impl<G, D> Look<G, D> {
    pub fn new(get: G, dir: D) -> Self {
        Self { get, dir }
    }
}

macro_rules! impl_look_traits {
    ($d:ident(&$($mut:ident)?) => $p:ident::$p_fn:ident+ $n:ident::$n_fn:ident) => {
        impl<G, D, T, Idx> $p<Idx, T> for Look<G, D>
        where
            G: $p<Idx, T>,
        {
            #[inline]
            fn $p_fn(&$($mut)? self, index: Idx) -> Point2<T> {
                self.get.$p_fn(index)
            }
        }

        impl<G, D, Idx, T> $n<Idx, T> for Look<G, D>
        where
            G: $p<Idx, T>,
            D: $d(Point2<T>) -> T,
        {
            #[inline]
            fn $n_fn(&$($mut)? self, index: Idx) -> T {
                (self.dir)(self.get.$p_fn(index))
            }
        }
    };
}

impl_look_traits!(FnMut(&mut) => PointMut::point_mut + NormMut::norm_mut);
impl_look_traits!(Fn(&) => Point::point + Norm::norm);

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Proj<Idx, T> {
    pub hit: Hit,
    pub adjacent: (Idx, Idx),
    pub point: Point2<T>,
}

impl<Idx, T> Proj<Idx, T> {
    pub fn new<F>(mut f: F, range: Range<Idx>, hull: &Hull<Idx, T>, projectile: T) -> Option<Self>
    where
        T: Float,
        Idx: PrimInt + NumAssign + TryFrom<usize>,
        F: NormMut<Idx, T> + PointMut<Idx, T>,
    {
        let hit = Hit::new(&hull.norms, &hull.partials, &projectile);
        if hit.trace(hull.norms.len()) == Equal {
            let adjacent = hit.adjacent(&range, &hull.indices, &hull.partials);
            let prev = index_norm(&mut f, adjacent.0);
            let next = index_norm(&mut f, adjacent.1);
            let segment = Segment::new(prev, next);
            let point = segment.lerp(f, projectile);
            Some(Self {
                hit,
                adjacent,
                point,
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct View<F, Idx, T> {
    pub f: F,
    pub range: Range<Idx>,
    pub hull: Hull<Idx, T>,
}

impl<F, Idx, T> View<F, Idx, T> {
    #[inline]
    pub fn with_hull(f: F, range: Range<Idx>, hull: Hull<Idx, T>) -> Self {
        Self { f, range, hull }
    }
}

macro_rules! impl_view_getters {
    ($($field:ident: $t:ty),+) => {
        impl<F, Idx, T> View<F, Idx, T> {
           $(
            pub fn $field(&self) -> &[$t] {
                &self.hull.$field
            }
           )+
        }
    };
}

impl_view_getters!(indices: Idx, norms: T, partials: (Idx, Idx));

fn hull<F, Idx, T, Iter>(f: F, range: Range<Idx>, iter: Iter) -> Hull<Idx, T>
where
    F: PointMut<Idx, T>,
    Idx: PrimInt + NumAssign + TryInto<usize> + TryFrom<usize>,
    T: Float,
    Iter: Iterator<Item = (Category, Segment<Idx, T>)>,
{
    let size_hint = iter.size_hint();
    let capacity = size_hint.1.unwrap_or(size_hint.0);
    let mut hull = Hull::<Idx, T>::with_capacity(capacity);
    hull.extend(f, range, iter);
    hull
}

struct Ptr<F>(pub *mut F);

macro_rules! impl_ptr_trait {
    ($trait:ident::$fn:ident -> $output:ty) => {
        impl<F, Idx, T> $trait<Idx, T> for Ptr<F>
        where
            F: $trait<Idx, T>,
        {
            fn $fn(&mut self, index: Idx) -> $output {
                unsafe { &mut *self.0 }.$fn(index)
            }
        }
    };
}

impl_ptr_trait!(PointMut::point_mut -> Point2<T>);
impl_ptr_trait!(NormMut::norm_mut -> T);

impl<F, Idx, T> View<F, Idx, T>
where
    F: PointMut<Idx, T> + NormMut<Idx, T>,
    Idx: PrimInt + NumAssign + TryInto<usize> + TryFrom<usize>,
    T: Float,
{
    pub fn new(mut f: F, range: Range<Idx>) -> Self {
        let minmax = Minmax::new(|i| f.norm_mut(i), range.clone());
        Self::with_minmax(f, range, minmax)
    }

    pub fn with_iter<Iter>(mut f: F, range: Range<Idx>, iter: Iter) -> Self
    where
        Iter: Iterator<Item = (Category, Segment<Idx, T>)>,
    {
        let hull = hull(|i| f.point_mut(i), range.clone(), iter);
        Self { f, range, hull }
    }

    pub fn with_minmax(mut f: F, range: Range<Idx>, minmax: Minmax<Idx, T>) -> Self {
        let ptr = Ptr(&mut f);
        let iter = Iter::new(Ptr(ptr.0), range.clone(), minmax);
        let hull = hull(ptr, range.clone(), iter);
        Self { f, range, hull }
    }
}

struct Ref<'a, F>(pub &'a F);

macro_rules! impl_ref_trait {
    ($f_fn:ident => $trait:ident::$fn:ident(&$($mut:ident)?) -> $output:ty) => {
        impl<'a, F, Idx, T> $trait<Idx, T> for Ref<'a, F>
        where
            F: Norm<Idx, T> + Point<Idx, T>,
        {
            fn $fn(&$($mut)? self, index: Idx) -> $output {
                self.0.$f_fn(index)
            }
        }
    };
}

impl_ref_trait!(point => PointMut::point_mut(&mut) -> Point2<T>);
impl_ref_trait!(point => Point::point(&) -> Point2<T>);
impl_ref_trait!(norm => NormMut::norm_mut(&mut) ->T);
impl_ref_trait!(norm => Norm::norm(&) ->T);

macro_rules! impl_view_proj {
    ($w:ident($p:ident + $n:ident) => $fn:ident(&$($mut:ident)?)) => {
        impl<F, Idx, T> View<F, Idx, T>
        where
            F: $p<Idx, T> + $n<Idx, T>,
            T: Float,
            Idx: PrimInt + NumAssign + TryFrom<usize>,
        {
            pub fn $fn(&$($mut)? self, projectile: T) -> Option<Proj<Idx, T>> {
                Proj::new($w(&$($mut)? self.f), self.range.clone(), &self.hull, projectile)
            }
        }
    };
}

impl_view_proj!(Ptr(PointMut + NormMut) => proj_mut(&mut));
impl_view_proj!(Ref(Point + Norm) => proj(&));
