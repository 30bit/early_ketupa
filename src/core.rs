use {
    crate::utils::{
        cmp_quad_axes, fit, index_norm, is_sin_negative, is_sin_positive, offset_result,
        unwrap_unchecked, wrap_decrement, wrap_increment, WrappingRange,
    },
    core::{
        cmp::Ordering::{self, *},
        fmt::{self, Debug},
        hint::unreachable_unchecked,
        marker::PhantomData,
        mem::{replace, swap},
        ops::{Add, AddAssign, Div, Mul, Range, Sub, SubAssign},
    },
    mint::Point2,
    num_traits::{one, One},
    Category::*,
    Minmax::*,
};

macro_rules! impl_traits {
    ($trait:ident::$fn:ident | $trait_mut:ident::$fn_mut:ident -> $output:ty) => {
        pub trait $trait_mut<Idx, T> {
            fn $fn_mut(&mut self, index: Idx) -> $output;
        }

        impl<F, Idx, T> $trait_mut<Idx, T> for F
        where
            F: FnMut(Idx) -> $output,
        {
            #[inline]
            fn $fn_mut(&mut self, index: Idx) -> $output {
                self(index)
            }
        }

        pub trait $trait<Idx, T>: $trait_mut<Idx, T> {
            fn $fn(&self, index: Idx) -> $output;
        }

        impl<F, Idx, T> $trait<Idx, T> for F
        where
            F: Fn(Idx) -> $output,
        {
            #[inline]
            fn $fn(&self, index: Idx) -> $output {
                self(index)
            }
        }
    };
}

impl_traits!(Point::point | PointMut::point_mut -> Point2<T>);
impl_traits!(Norm::norm | NormMut::norm_mut -> T);

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Minmax<Idx, T> {
    Empty,
    Single(Idx, T),
    Pair { min: (Idx, T), max: (Idx, T) },
}

impl<Idx, T> Minmax<Idx, T> {
    pub fn unwrap_single(self) -> (Idx, T) {
        if let Single(index, norm) = self {
            (index, norm)
        } else {
            unreachable!()
        }
    }

    pub fn unwrap_pair(self) -> ((Idx, T), (Idx, T)) {
        if let Pair { min, max } = self {
            (min, max)
        } else {
            unreachable!()
        }
    }

    pub fn unique(self) -> Self
    where
        T: PartialEq,
    {
        match self {
            Pair { min, max } if min.1 == max.1 => Single(min.0, min.1),
            otherwise => otherwise,
        }
    }
}

impl<Idx, T> Minmax<Idx, T>
where
    Idx: PartialEq + Clone + AddAssign,
    T: PartialOrd,
{
    pub fn new<F>(mut f: F, range: Range<Idx>) -> Self
    where
        Idx: One,
        F: NormMut<Idx, T>,
    {
        let Range { mut start, end } = range;
        let (mut min, mut max) = {
            if start == end {
                return Empty;
            }
            let mut fst = index_norm(&mut f, start.clone());
            start += one();
            if start == end {
                return Single(fst.0, fst.1);
            }
            let mut snd = index_norm(&mut f, start.clone());
            Self::sort(&mut fst, &mut snd);
            (fst, snd)
        };
        loop {
            start += one();
            if start == end {
                break;
            }
            let fst = index_norm(&mut f, start.clone());
            start += one();
            if start == end {
                Self::push(&mut min, &mut max, fst);
                break;
            } else {
                let snd = index_norm(&mut f, start.clone());
                Self::push_pair(&mut min, &mut max, fst, snd);
            }
        }
        Pair { min, max }
    }

    pub fn union(self, other: Self, at: Idx) -> Self {
        match other {
            Empty => self,
            Single(mut other_index, other_norm) => {
                other_index += at;
                match self {
                    Empty => Single(other_index, other_norm),
                    Single(index, norm) => {
                        let mut min = (other_index, other_norm);
                        let mut max = (index, norm);
                        Self::sort(&mut min, &mut max);
                        Pair { min, max }
                    }
                    Pair { mut min, mut max } => {
                        Self::push(&mut min, &mut max, (other_index, other_norm));
                        Pair { min, max }
                    }
                }
            }
            Pair {
                min: mut other_min,
                max: mut other_max,
            } => {
                other_min.0 += at.clone();
                other_max.0 += at;
                if let Single(index, norm) = self {
                    Self::push(&mut other_min, &mut other_max, (index, norm));
                } else if let Pair { min, max } = self {
                    Self::unite(&mut other_min, &mut other_max, min, max);
                }
                Pair {
                    min: other_min,
                    max: other_max,
                }
            }
        }
    }

    fn sort(fst: &mut (Idx, T), snd: &mut (Idx, T)) {
        if fst.1 > snd.1 {
            swap(fst, snd)
        }
    }

    fn push(min: &mut (Idx, T), max: &mut (Idx, T), other: (Idx, T)) {
        if other.1 <= min.1 {
            *min = other
        } else if other.1 > max.1 {
            *max = other
        }
    }

    fn unite(min: &mut (Idx, T), max: &mut (Idx, T), other_min: (Idx, T), other_max: (Idx, T)) {
        if other_min.1 <= min.1 {
            *min = other_min;
        }
        if other_max.1 > max.1 {
            *max = other_max;
        }
    }

    fn push_pair(min: &mut (Idx, T), max: &mut (Idx, T), fst: (Idx, T), snd: (Idx, T)) {
        if fst.1 > snd.1 {
            Self::unite(min, max, snd, fst)
        } else {
            Self::unite(min, max, fst, snd)
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Category {
    Obscuring,
    Contiguous,
    Partial,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Segment<Idx, T> {
    pub from: (Idx, T),
    pub to: (Idx, T),
}

impl<Idx, T> Segment<Idx, T> {
    #[inline]
    pub fn new(from: (Idx, T), to: (Idx, T)) -> Self {
        Self { from, to }
    }

    pub fn lerp<F, U>(self, mut f: F, projectile: T) -> Point2<U>
    where
        U: Clone
            + Sub
            + Add<<<U as Sub>::Output as Mul<<T::Output as Div>::Output>>::Output, Output = U>,
        <U as Sub>::Output: Mul<<T::Output as Div>::Output>,
        Idx: Clone,
        F: PointMut<Idx, U>,
        T: PartialOrd + Clone + Sub,
        T::Output: Div,
        <T::Output as Div>::Output: Clone,
    {
        let point_from = f.point_mut(self.from.0);
        if self.from.1 == projectile {
            return point_from;
        }
        let point_to = f.point_mut(self.to.0);
        if self.to.1 == projectile {
            return point_to;
        }
        let t = (projectile - self.from.1.clone()) / (self.to.1.clone() - self.from.1.clone());
        let y = point_from.y.clone() + (point_to.y - point_from.y) * t.clone();
        let x = point_from.x.clone() + (point_to.x - point_from.x) * t;
        Point2 { x, y }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Iter<F, Idx, T, U = T> {
    f: F,
    range: WrappingRange<Idx>,
    max: Option<(Idx, T)>,
    next: Option<(Category, Segment<Idx, T>)>,
    phantom: PhantomData<U>,
}

macro_rules! next {
    ($iter:ident::$category:ident.$part:ident) => {{
        if let Some(index) = $iter.range.next() {
            index_norm(&mut $iter.f, index)
        } else if let Some(max) = replace(&mut $iter.max, None) {
            let prev = unwrap_unchecked!(&mut $iter.next);
            *prev = ($category, Segment::new(prev.1.$part.clone(), max));
            return;
        } else {
            $iter.next = None;
            return;
        }
    }};
    ($iter:ident::Partial) => {{
        next!($iter::Partial.from)
    }};
    ($iter:ident::$category:ident) => {{
        next!($iter::$category.to)
    }};
}

macro_rules! wait {
    (if $iter:ident($prev:expr).$op:ident $ok:block) => {{
        loop {
            let next = next!($iter::Partial);
            if $prev.1.to.1.$op(&next.1) {
                $prev.1.to = next;
                break $ok;
            } else {
                $prev.1.from = next;
            }
        }
    }};
}

macro_rules! is_sin_at {
    ($iter:ident::$is_sin:ident($($index:expr),+)) => {
        $is_sin($($iter.f.point_mut($index.clone()),)+)
    };
}

macro_rules! obscure {
    (
        if $iter:ident::$category:ident.$op:ident $if:block
        else if $is_sin:ident $else_if:block
        else $else:block
    ) => {{
        let next = next!($iter::$category);
        let prev = unwrap_unchecked!(&mut $iter.next);
        if prev.1.to.1.$op(&next.1) {
            prev.1.from = replace(&mut prev.1.to, next);
            $if
        } else if is_sin_at!($iter::$is_sin(prev.1.from.0, prev.1.to.0, next.0)) {
            prev.1.from = next;
            wait!(if $iter(prev).$op $else_if)
        } else {
            prev.1.from = replace(&mut prev.1.to, next);
            $else
        }
    }};
}

impl<F, Idx, T, U> Iter<F, Idx, T, U>
where
    F: NormMut<Idx, T>,
    Idx: Clone + PartialOrd + AddAssign + One,
    T: PartialEq + Clone,
{
    pub fn new(f: F, range: Range<Idx>, minmax: Minmax<Idx, T>) -> Self {
        match minmax {
            Empty => Iter {
                f,
                range: WrappingRange::empty(range.start),
                max: None,
                next: None,
                phantom: Default::default(),
            },
            Single(index, norm) => Self::obscuring(
                f,
                WrappingRange::empty(range.start),
                None,
                (index.clone(), norm.clone()),
                (index, norm),
            ),
            Pair { min, max } => Self::from_pair(f, range, min, max),
        }
    }

    fn from_pair(mut f: F, range: Range<Idx>, mut min: (Idx, T), max: (Idx, T)) -> Self {
        let mut range = WrappingRange::new(min.0.clone()..max.0.clone(), range);
        while let Some(index) = range.next() {
            let (index, norm) = index_norm(&mut f, index);
            if norm == min.1 {
                min.0 = index;
            } else {
                return Self::obscuring(f, range, Some(max), min, (index, norm));
            }
        }
        Self::obscuring(f, range, None, min, max)
    }

    fn obscuring(
        f: F,
        range: WrappingRange<Idx>,
        max: Option<(Idx, T)>,
        from: (Idx, T),
        to: (Idx, T),
    ) -> Self {
        Self {
            f,
            range,
            max,
            next: Some((Obscuring, Segment::new(from, to))),
            phantom: Default::default(),
        }
    }

    fn update(&mut self)
    where
        F: PointMut<Idx, U>,
        T: PartialOrd,
        U: Sub + Clone,
        U::Output: Mul,
        <U::Output as Mul>::Output: PartialOrd,
    {
        let category = obscure!(if self::Contiguous.lt {
            Contiguous
        } else if is_sin_positive {
            Partial
        } else {
            loop {
                obscure!(if self::Obscuring.gt {
                } else if is_sin_negative {
                } else {
                    break Obscuring;
                })
            }
        });
        unwrap_unchecked!(&mut self.next).0 = category;
    }
}

impl<F, Idx, T, U> Iterator for Iter<F, Idx, T, U>
where
    F: NormMut<Idx, T> + PointMut<Idx, U>,
    Idx: Clone + PartialOrd + AddAssign + SubAssign + One + TryInto<usize>,
    T: PartialOrd + Clone,
    U: Sub + Clone,
    U::Output: Mul,
    <U::Output as Mul>::Output: PartialOrd,
{
    type Item = (Category, Segment<Idx, T>);

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next.clone();
        self.update();
        next
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let hint = self.range.len();
        if let Ok(index) = hint.try_into() {
            (0, Some(index))
        } else {
            (0, None)
        }
    }
}

impl<F, T, Idx> Debug for Iter<F, T, Idx>
where
    T: Debug,
    Idx: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Iter").field("next", &self.next).finish()
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Hit {
    pub is_exact: bool,
    pub is_partial: bool,
    pub norms_index: usize,
    pub partials_index: usize,
}

impl Hit {
    #[inline]
    pub fn new<T, Idx, Jdx>(norms: &[T], partials: &[(Idx, Jdx)], projectile: &T) -> Self
    where
        T: PartialOrd,
        Jdx: PartialOrd + TryFrom<usize>,
    {
        Self::with_offset(0, 0, norms, partials, projectile)
    }

    pub fn with_offset<T, Idx, Jdx>(
        norms_offset: usize,
        partials_offset: usize,
        norms: &[T],
        partials: &[(Idx, Jdx)],
        projectile: &T,
    ) -> Self
    where
        T: PartialOrd,
        Jdx: PartialOrd + TryFrom<usize>,
    {
        let (norms_index, is_exact) = offset_result(
            norms.binary_search_by(|norm| norm.partial_cmp(projectile).unwrap()),
            norms_offset,
        );
        let norms_jndex = fit!(norms_index, norms_index, "Jdx");
        let (partials_index, is_partial) = offset_result(
            partials.binary_search_by(|(_, j)| j.partial_cmp(&norms_jndex).unwrap()),
            partials_offset,
        );
        Self {
            is_exact,
            is_partial,
            norms_index,
            partials_index,
        }
    }

    pub fn trace(self, norms_len: usize) -> Ordering {
        if self.is_exact {
            Equal
        } else if self.norms_index == 0 {
            Less
        } else if self.norms_index == norms_len {
            Greater
        } else {
            Equal
        }
    }

    pub fn adjacent<Idx, Jdx>(
        self,
        range: &Range<Idx>,
        indices: &[Idx],
        partials: &[(Idx, Jdx)],
    ) -> (Idx, Idx)
    where
        Idx: PartialEq + SubAssign + AddAssign + Clone + One,
    {
        if self.is_exact {
            let index = indices[self.norms_index].clone();
            (index.clone(), index)
        } else if self.is_partial {
            let next = partials[self.partials_index].0.clone();
            let prev = wrap_decrement(range, next.clone());
            (prev, next)
        } else if self.norms_index == 0 || self.norms_index == indices.len() {
            let prev = indices.first().unwrap().clone();
            let next = unwrap_unchecked!(indices.last()).clone();
            (prev, next)
        } else {
            self.radjacent(range, indices)
        }
    }

    fn radjacent<Idx>(self, range: &Range<Idx>, indices: &[Idx]) -> (Idx, Idx)
    where
        Idx: PartialEq + AddAssign + Clone + One,
    {
        let prev = indices[self.norms_index - 1].clone();
        let next = wrap_increment(range, prev.clone());
        (prev, next)
    }

    pub(crate) fn partials_successor_index(self) -> usize {
        if self.is_partial {
            self.partials_index + 1
        } else {
            self.partials_index
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Hull<Idx, T, Jdx = Idx> {
    pub norms: Vec<T>,
    pub indices: Vec<Idx>,
    pub partials: Vec<(Idx, Jdx)>,
}

impl<Idx, T, Jdx> Default for Hull<Idx, T, Jdx> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<Idx, T, Jdx> Hull<Idx, T, Jdx> {
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            norms: Vec::with_capacity(capacity),
            indices: Vec::with_capacity(capacity),
            partials: Vec::with_capacity(capacity / 3),
        }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.indices.reserve(additional);
        self.norms.reserve(additional);
        self.partials.reserve(additional / 3);
    }

    #[inline]
    pub fn extend<F, U, Iter>(&mut self, f: F, range: Range<Idx>, iter: Iter)
    where
        F: PointMut<Idx, U>,
        Idx: PartialEq + TryInto<usize> + SubAssign + One + Clone,
        T: PartialOrd + Clone,
        Jdx: TryFrom<usize> + TryInto<usize> + PartialOrd + Clone,
        U: Sub + Clone,
        U::Output: Mul,
        <U::Output as Mul>::Output: PartialOrd,
        Iter: IntoIterator<Item = (Category, Segment<Idx, T>)>,
    {
        Extender::new(f, range, self).extend(iter)
    }
}

struct Extender<'a, F, Idx, T, Jdx = Idx, U = T> {
    f: F,
    range: Range<Idx>,
    hull: &'a mut Hull<Idx, T, Jdx>,
    phantom: PhantomData<U>,
}

impl<'a, F, Idx, T, Jdx, U> Extender<'a, F, Idx, T, Jdx, U>
where
    F: PointMut<Idx, U>,
    Idx: PartialEq + TryInto<usize> + SubAssign + One + Clone,
    T: PartialOrd + Clone,
    Jdx: TryFrom<usize> + TryInto<usize> + PartialOrd + Clone,
    U: Sub + Clone,
    U::Output: Mul,
    <U::Output as Mul>::Output: PartialOrd,
{
    pub fn new(f: F, range: Range<Idx>, hull: &'a mut Hull<Idx, T, Jdx>) -> Self {
        Extender {
            f,
            range,
            hull,
            phantom: Default::default(),
        }
    }

    pub fn extend<Iter>(&mut self, iter: Iter)
    where
        Iter: IntoIterator<Item = (Category, Segment<Idx, T>)>,
    {
        let mut iter = iter.into_iter();
        while let Some((category, Segment { from, to })) = iter.next() {
            match category {
                Contiguous => self.push_visible(to.0, to.1),
                Obscuring => {
                    let hit = self.hit(0, 0, &from.1);
                    let p = self.f.point_mut(from.0.clone());
                    let partials_len = hit.partials_successor_index();
                    if self.is_obscured(&hit, p) {
                        let obstacle = self.obstacle(partials_len);
                        self.try_protrude_loop(&mut iter, obstacle, from.0, to)
                    } else {
                        self.truncate(hit.norms_index, partials_len);
                        if from.1 != to.1 {
                            self.push_visible(from.0, from.1);
                        }
                        self.push_visible(to.0, to.1);
                    }
                }
                Partial => {
                    self.push_partial(to.0.clone());
                    self.push_visible(to.0, to.1);
                }
            }
        }
    }

    fn is_obscured(&mut self, hit: &Hit, p: Point2<U>) -> bool {
        if hit.norms_index == 0 || hit.partials_index == self.hull.partials.len() {
            return false;
        }
        let (from, mid) = if hit.is_partial {
            let mid = self.hull.partials[hit.partials_index].0.clone();
            let from = wrap_decrement(&self.range, mid.clone());
            (from, mid)
        } else {
            let mid = self.hull.indices[hit.norms_index].clone();
            let from = self.hull.indices[hit.norms_index - 1].clone();
            (from, mid)
        };
        is_sin_positive(self.f.point_mut(from), self.f.point_mut(mid), p)
    }

    fn try_protrude_loop<Iter>(
        &mut self,
        iter: &mut Iter,
        mut obstacle: (usize, T),
        mut from_index: Idx,
        mut to: (Idx, T),
    ) where
        Iter: Iterator<Item = (Category, Segment<Idx, T>)>,
    {
        loop {
            if let Some(protrusion) = self.try_protrude(&mut obstacle, from_index, &to) {
                self.truncate(self.norms_index(protrusion), protrusion + 1);
                self.hull.partials[protrusion].0 = to.0.clone();
                self.push_visible(to.0, to.1);
                return;
            } else if let Some((_, s)) = iter.next() {
                from_index = s.from.0;
                to = s.to;
            } else {
                return;
            }
        }
    }

    fn try_protrude(
        &mut self,
        obstacle: &mut (usize, T),
        from_index: Idx,
        to: &(Idx, T),
    ) -> Option<usize> {
        if to.1 <= obstacle.1 {
            return None;
        }
        let point_to = self.f.point_mut(to.0.clone());
        let hit = self.hit(self.norms_index(obstacle.0), obstacle.0, &to.1);
        if self.is_obscured(&hit, point_to.clone()) {
            *obstacle = self.obstacle(hit.partials_index);
            return None;
        }
        let point_from = self.f.point_mut(from_index);
        let partials_len = hit.partials_successor_index();
        match self.search_protrusion(obstacle.0..partials_len, point_from, point_to) {
            (index, true) => Some(index),
            _ => Some(0),
        }
    }

    fn search_protrusion(
        &mut self,
        partials_range: Range<usize>,
        from: Point2<U>,
        to: Point2<U>,
    ) -> (usize, bool) {
        let slice = &self.hull.partials[partials_range.clone()];
        let result = slice.binary_search_by(|(i, j)| {
            let ceil = i.clone();
            let floor_index = fit!(j.clone(), "Jdx", "usize");
            let floor = self.hull.indices[floor_index - 1].clone();
            cmp_quad_axes(
                from.clone(),
                to.clone(),
                self.f.point_mut(floor),
                self.f.point_mut(ceil),
            )
        });
        offset_result(result, partials_range.start)
    }

    fn norms_index(&self, partial_index: usize) -> usize {
        let j = self.hull.partials[partial_index].1.clone();
        fit!(j, "Jdx", "usize")
    }

    fn obstacle(&self, partial_index: usize) -> (usize, T) {
        let norms_index = self.norms_index(partial_index);
        let norm = self.hull.norms[norms_index - 1].clone();
        (partial_index, norm)
    }

    fn hit(&mut self, norms_start: usize, partials_start: usize, projectile: &T) -> Hit {
        let norms = &self.hull.norms[norms_start..];
        let partials = &self.hull.partials[partials_start..];
        Hit::with_offset(norms_start, partials_start, norms, partials, projectile)
    }

    fn truncate(&mut self, norms_len: usize, partials_len: usize) {
        self.hull.norms.truncate(norms_len);
        self.hull.indices.truncate(norms_len);
        self.hull.partials.truncate(partials_len);
    }

    fn push_visible(&mut self, index: Idx, norm: T) {
        self.hull.indices.push(index);
        self.hull.norms.push(norm);
    }

    fn push_partial(&mut self, index: Idx) {
        let len = self.hull.norms.len();
        let len = fit!(len, len, "Jdx");
        let partial = (index, len);
        self.hull.partials.push(partial)
    }
}
