use {
    crate::core::NormMut,
    core::{
        cmp::Ordering::{self, *},
        mem::replace,
        ops::{AddAssign, Mul, Range, Sub, SubAssign},
    },
    mint::Point2,
    num_traits::{one, One},
};
pub fn index_norm<F, Idx, T>(f: &mut F, index: Idx) -> (Idx, T)
where
    F: NormMut<Idx, T>,
    Idx: Clone,
{
    (index.clone(), f.norm_mut(index))
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct WrappingRange<Idx> {
    pub start: Idx,
    pub end: Idx,
    pub reset: Idx,
    pub overflow: Idx,
}

impl<Idx> WrappingRange<Idx>
where
    Idx: PartialOrd + Clone,
{
    pub fn new(inner: Range<Idx>, outer: Range<Idx>) -> Self {
        if inner.start <= inner.end {
            WrappingRange {
                start: inner.start,
                end: inner.end,
                reset: outer.start.clone(),
                overflow: outer.start,
            }
        } else {
            WrappingRange {
                start: inner.start,
                end: outer.end,
                reset: outer.start,
                overflow: inner.end,
            }
        }
    }

    pub fn empty(reset: Idx) -> Self {
        Self {
            start: reset.clone(),
            end: reset.clone(),
            reset: reset.clone(),
            overflow: reset,
        }
    }

    pub fn len(&self) -> Idx
    where
        Idx: SubAssign + AddAssign,
    {
        let mut len = self.end.clone();
        len += self.overflow.clone();
        len -= self.start.clone();
        len -= self.reset.clone();
        len
    }

    fn shrink(&mut self) -> bool
    where
        Idx: AddAssign + One,
    {
        self.start += one();
        if self.start >= self.end {
            self.start = self.reset.clone();
            self.end = replace(&mut self.overflow, self.reset.clone());
            return self.start != self.end;
        }
        true
    }
}

impl<Idx> Iterator for WrappingRange<Idx>
where
    Idx: Clone + PartialOrd + AddAssign + One,
{
    type Item = Idx;

    fn next(&mut self) -> Option<Self::Item> {
        if self.shrink() {
            Some(self.start.clone())
        } else {
            None
        }
    }
}

macro_rules! impl_is_sin {
    ($name:ident: $op:ident) => {
        pub fn $name<T>(
            Point2 {
                x: from_x,
                y: from_y,
            }: Point2<T>,
            Point2 { x: mid_x, y: mid_y }: Point2<T>,
            Point2 { x: to_x, y: to_y }: Point2<T>,
        ) -> bool
        where
            T: Sub + Clone,
            T::Output: Mul,
            <T::Output as Mul>::Output: PartialOrd,
        {
            let a = (mid_x.clone() - from_x) * (to_y - mid_y.clone());
            let b = (to_x - mid_x) * (mid_y - from_y);
            a.$op(&b)
        }
    };
}

impl_is_sin!(is_sin_positive: gt);

impl_is_sin!(is_sin_negative: lt);

pub fn cmp_quad_axes<T>(
    major_from: Point2<T>,
    major_to: Point2<T>,
    minor_from: Point2<T>,
    minor_to: Point2<T>,
) -> Ordering
where
    T: Sub + Clone,
    T::Output: Mul,
    <T::Output as Mul>::Output: PartialOrd,
{
    if is_sin_positive(minor_from.clone(), major_from.clone(), major_to.clone()) {
        Greater
    } else if is_sin_positive(minor_to, major_to, major_from) {
        Less
    } else {
        Equal
    }
}

macro_rules! unwrap_unchecked {
    ($option:expr) => {
        if let Some(unchecked) = $option {
            unchecked
        } else {
            unsafe { unreachable_unchecked() }
        }
    };
}

pub fn offset_result(result: Result<usize, usize>, additional: usize) -> (usize, bool) {
    match result {
        Ok(index) => (index + additional, true),
        Err(index) => (index + additional, false),
    }
}

macro_rules! fit {
    ($try_from:expr, $msg_start:expr, $msg_end:expr) => {
        if let Ok(value) = $try_from.try_into() {
            value
        } else {
            panic!("{} cannot be fit into {}", $msg_start, $msg_end)
        }
    };
}

pub fn wrap_decrement<Idx>(bounds: &Range<Idx>, mut index: Idx) -> Idx
where
    Idx: PartialEq + SubAssign + One + Clone,
{
    if index == bounds.start {
        index = bounds.end.clone();
    }
    index -= one();
    index
}

pub fn wrap_increment<Idx>(bounds: &Range<Idx>, mut index: Idx) -> Idx
where
    Idx: PartialEq + AddAssign + One + Clone,
{
    index += one();
    if index == bounds.end {
        index = bounds.start.clone();
    }
    index
}

pub(crate) use {fit, unwrap_unchecked};