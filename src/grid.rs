pub mod parse;

use core::f64;
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    mem,
    str::FromStr,
};

use ahash::{AHashMap, AHashSet, AHasher};
use parse::parse_grid;
use rand::{
    distr::{Distribution, StandardUniform},
    Rng,
};
use ustr::{ustr, Ustr, UstrMap};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    pub kind: Ustr,
    pub value: f64,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            kind: Ustr::default(),
            value: 1.0,
        }
    }
}

impl From<Ustr> for Cell {
    fn from(kind: Ustr) -> Self {
        Self { kind, value: 1.0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Direction {
    #[default]
    North,
    East,
    South,
    West,
}

impl Distribution<Direction> for StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Direction {
        match rng.random_range(0..4) {
            0 => Direction::North,
            1 => Direction::East,
            2 => Direction::South,
            3 => Direction::West,
            _ => unreachable!(),
        }
    }
}

impl Direction {
    pub fn apply(&self, x: usize, y: usize, width: usize, height: usize) -> (usize, usize) {
        match self {
            Self::North => (x, y),
            Self::East => (y, width - x - 1),
            Self::South => (width - x - 1, height - y - 1),
            Self::West => (height - y - 1, x),
        }
    }

    pub fn inverse(&self) -> Self {
        match self {
            Direction::North => Direction::North,
            Direction::East => Direction::West,
            Direction::South => Direction::South,
            Direction::West => Direction::East,
        }
    }

    pub fn apply_size(&self, width: usize, height: usize) -> (usize, usize) {
        match self {
            Direction::North => (width, height),
            Direction::East => (height, width),
            Direction::South => (width, height),
            Direction::West => (height, width),
        }
    }
}

pub struct Grid {
    pub(crate) cells: Vec<Cell>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) caches: HashMap<(usize, usize), PatternCache>,
    pub(crate) inverse: UstrMap<AHashSet<(usize, usize)>>,
}

pub(crate) struct PatternCache {
    pub(crate) pattern_width: usize,
    pub(crate) pattern_height: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) hashes: Vec<u64>,
    pub(crate) inverse: AHashMap<u64, AHashSet<(usize, usize)>>,
}

impl FromStr for Grid {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (i, grid) = parse_grid(s).or(Err(()))?;

        if !i.is_empty() {
            return Err(());
        }

        grid.ok_or(())
    }
}

impl Grid {
    pub fn from_cells(width: usize, height: usize, cells: Vec<Cell>) -> Option<Self> {
        if width * height != cells.len() {
            return None;
        }

        let mut inverse: UstrMap<AHashSet<_>> = UstrMap::default();

        for y in 0..height {
            for x in 0..width {
                let cell = &cells[x + y * width];
                inverse.entry(cell.kind).or_default().insert((x, y));
            }
        }

        Some(Self {
            cells,
            width,
            height,
            caches: Default::default(),
            inverse,
        })
    }

    pub fn new(width: usize, height: usize, cell: impl Into<Cell>) -> Self {
        Self::from_cells(width, height, vec![cell.into(); width * height]).unwrap()
    }

    pub fn empty(width: usize, height: usize) -> Self {
        Self::new(width, height, Cell::default())
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn add_caches(&mut self, width: usize, height: usize) {
        self.add_cache(width, height);
        self.add_cache(height, width);
    }

    pub fn add_cache(&mut self, width: usize, height: usize) {
        if !self.caches.contains_key(&(width, height)) {
            self.caches.insert(
                (width, height),
                PatternCache::new(&self.cells, self.width, self.height, width, height),
            );
        }
    }

    pub fn count(&self, kind: impl Into<Ustr>) -> usize {
        self.inverse
            .get(&kind.into())
            .map(|l| l.len())
            .unwrap_or_default()
    }

    pub fn get(&self, x: usize, y: usize) -> Option<&Cell> {
        if x >= self.width || y >= self.height {
            return None;
        }

        Some(&self.cells[x + y * self.width])
    }

    pub fn set_value(&mut self, x: usize, y: usize, value: f64) -> Option<f64> {
        if x >= self.width || y >= self.height {
            return None;
        }

        Some(mem::replace(
            &mut self.cells[x + y * self.width].value,
            value,
        ))
    }

    pub fn set_value_many(&mut self, points: impl IntoIterator<Item = (usize, usize)>, value: f64) {
        for (x, y) in points.into_iter() {
            self.set_value(x, y, value);
        }
    }

    pub fn set(&mut self, x: usize, y: usize, cell: impl Into<Cell>) -> Option<Cell> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let cell = cell.into();
        let old = mem::replace(&mut self.cells[x + y * self.width], cell);

        if old.kind != cell.kind {
            self.update_cache(x, y, x, y);

            let old_locations = self.inverse.entry(old.kind).or_default();
            old_locations.remove(&(x, y));
            let new_locations = self.inverse.entry(cell.kind).or_default();
            new_locations.insert((x, y));
        }

        Some(old)
    }

    pub fn set_many(
        &mut self,
        points: impl IntoIterator<Item = (usize, usize)>,
        cell: impl Into<Cell>,
    ) {
        let cell = cell.into();

        for (x, y) in points.into_iter() {
            self.set(x, y, cell);
        }
    }

    fn kind_hash(&self, direction: Direction) -> u64 {
        let mut hasher = AHasher::default();

        let (rotated_width, rotated_height) = direction.apply_size(self.width, self.height);
        let inverse = direction.inverse();

        for y in 0..rotated_height {
            for x in 0..rotated_width {
                let (x, y) = inverse.apply(x, y, rotated_width, rotated_height);
                self.cells[x + y * self.width].kind.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    pub fn map(&mut self, kind: Ustr, f: impl Fn(Cell) -> Option<Cell>) {
        if let Some(locations) = self.inverse.get(&kind) {
            locations.iter().for_each(|(x, y)| {
                let cell = &mut self.cells[x + y * self.width];
                if let Some(new) = f(*cell) {
                    *cell = new;
                }
            });
        }
    }

    pub fn place_many(
        &mut self,
        placements: impl IntoIterator<Item = (usize, usize, Direction)>,
        stamp: &Grid,
    ) {
        for (x, y, direction) in placements.into_iter() {
            self.place(x, y, direction, stamp);
        }
    }

    pub fn replace(&mut self, old: impl Into<Ustr>, new: impl Into<Ustr>) -> usize {
        let old = old.into();
        let new = new.into();

        let Some(locations) = self.inverse.remove(&old) else {
            return 0;
        };

        let new_locations = self.inverse.entry(new).or_default();

        let replaced = locations.len();

        let mut min_x = self.width;
        let mut min_y = self.height;
        let mut max_x = 0;
        let mut max_y = 0;

        for (x, y) in locations.into_iter() {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);

            new_locations.insert((x, y));

            self.cells[x + y * self.width].kind = new;
        }

        self.update_cache(min_x, min_y, max_x, max_y);

        replaced
    }

    /// Hill climbs to largest sum of raycast in each cardinal direction.
    ///
    /// Returns (maximum, [neg_x, neg_y, pos_x, pos_y])
    pub fn open_area(&self, mut x: usize, mut y: usize) -> ((usize, usize), [usize; 4]) {
        let Some(cell) = self.get(x, y) else {
            return ((x, y), [0; 4]);
        };

        let kind = cell.kind;

        let mut last = (x, y);

        let mut pos_x = 0;
        let mut pos_y = 0;
        let mut neg_x = 0;
        let mut neg_y = 0;

        for x in (x + 1)..self.width {
            if self.get(x, y).unwrap().kind == kind {
                pos_x += 1;
            } else {
                break;
            }
        }

        for x in (0..x).rev() {
            if self.get(x, y).unwrap().kind == kind {
                neg_x += 1;
            } else {
                break;
            }
        }

        for y in (y + 1)..self.height {
            if self.get(x, y).unwrap().kind == kind {
                pos_y += 1;
            } else {
                break;
            }
        }

        for y in (0..y).rev() {
            if self.get(x, y).unwrap().kind == kind {
                neg_y += 1;
            } else {
                break;
            }
        }

        let mut total = pos_x + pos_y + neg_x + neg_y;
        let mut same_for = 0;

        loop {
            let max = pos_x.max(pos_y).max(neg_x).max(neg_y);

            if pos_x == max || neg_x == max {
                x = if pos_x > neg_x { x + 1 } else { x - 1 };

                pos_y = 0;
                neg_y = 0;

                for y in (y + 1)..self.height {
                    if self.get(x, y).unwrap().kind == kind {
                        pos_y += 1;
                    } else {
                        break;
                    }
                }

                for y in (0..y).rev() {
                    if self.get(x, y).unwrap().kind == kind {
                        neg_y += 1;
                    } else {
                        break;
                    }
                }
            } else if pos_y == max || neg_y == max {
                y = if pos_y > neg_y { y + 1 } else { y - 1 };

                pos_x = 0;
                neg_x = 0;

                for x in (x + 1)..self.width {
                    if self.get(x, y).unwrap().kind == kind {
                        pos_x += 1;
                    } else {
                        break;
                    }
                }

                for x in (0..x).rev() {
                    if self.get(x, y).unwrap().kind == kind {
                        neg_x += 1;
                    } else {
                        break;
                    }
                }
            }

            let new_total = pos_x + pos_y + neg_x + neg_y;

            if total == new_total {
                same_for += 1;
            } else {
                same_for = 0;
            }

            if total > new_total || same_for > 2 {
                return (last, [neg_x, neg_y, pos_x, pos_y]);
            }

            last = (x, y);
            total = new_total;
        }
    }

    pub fn decompose(
        &self,
        rng: &mut impl Rng,
        kind: impl Into<Ustr>,
        first: Option<usize>,
    ) -> Vec<((usize, usize), (usize, usize))> {
        let kind = kind.into();
        let mut rectangles = Vec::new();

        let Some(locations) = self.inverse.get(&kind) else {
            return rectangles;
        };

        let mut accounted_set = AHashSet::new();
        let mut unaccounted: Vec<_> = locations.iter().copied().collect();

        let mut possible_directions = Vec::new();

        let mut added = Vec::new();

        while !unaccounted.is_empty() {
            let (x, y) = unaccounted.swap_remove(rng.random_range(0..unaccounted.len()));

            if accounted_set.contains(&(x, y)) {
                continue;
            }

            let mut min_x = x;
            let mut min_y = y;
            let mut max_x = x;
            let mut max_y = y;

            let mut rectangle = ((min_x, min_y), (max_x, max_y));

            added.clear();
            possible_directions.clear();

            possible_directions.extend([
                Direction::North,
                Direction::East,
                Direction::South,
                Direction::West,
            ]);

            'expand: loop {
                if possible_directions.is_empty() {
                    break;
                }

                let direction_index = rng.random_range(0..possible_directions.len());

                match possible_directions[direction_index] {
                    Direction::East => {
                        max_x = (max_x + 1).min(self.width - 1);
                        added.extend((min_y..=max_y).map(|y| (max_x, y)));
                    }
                    Direction::South => {
                        max_y = (max_y + 1).min(self.height - 1);
                        added.extend((min_x..=max_x).map(|x| (x, max_y)));
                    }
                    Direction::West => {
                        min_x = min_x.saturating_sub(1);
                        added.extend((min_y..=max_y).map(|y| (min_x, y)));
                    }
                    Direction::North => {
                        min_y = min_y.saturating_sub(1);
                        added.extend((min_x..=max_x).map(|x| (x, min_y)));
                    }
                };

                for (x, y) in added.iter() {
                    if accounted_set.contains(&(*x, *y))
                        || self.cells[x + y * self.width].kind != kind
                    {
                        possible_directions.swap_remove(direction_index);
                        ((min_x, min_y), (max_x, max_y)) = rectangle;
                        added.clear();
                        continue 'expand;
                    }
                }

                accounted_set.extend(added.drain(..));

                rectangle = ((min_x, min_y), (max_x, max_y));
            }

            rectangles.push(rectangle);

            if let Some(first) = first {
                if rectangles.len() >= first {
                    return rectangles;
                }
            }
        }

        rectangles
    }

    pub fn place(&mut self, x: usize, y: usize, direction: Direction, stamp: &Grid) {
        let mut changed = false;

        for j in 0..stamp.height {
            for i in 0..stamp.width {
                let (rotated_i, rotated_j) = direction.apply(i, j, stamp.width, stamp.height);

                let x = x + rotated_i;
                let y = y + rotated_j;

                if x >= self.width || y >= self.height {
                    continue;
                }

                let new = stamp.cells[i + j * stamp.width];
                let old = &mut self.cells[x + y * self.width];

                if new.kind.is_empty() {
                    continue;
                }

                if new.kind != old.kind {
                    changed = true;

                    let old_locations = self.inverse.entry(old.kind).or_default();
                    old_locations.remove(&(x, y));
                    let new_locations = self.inverse.entry(new.kind).or_default();
                    new_locations.insert((x, y));
                }

                *old = new;
            }
        }

        if changed {
            let (rotated_width, rotated_height) = direction.apply_size(stamp.width, stamp.height);

            self.update_cache(x, y, x + rotated_width, y + rotated_height);
        }
    }

    pub fn normalize_values(&mut self, kind: impl Into<Ustr>, min: f64, max: f64) {
        let Some(locations) = self.inverse.get(&kind.into()) else {
            return;
        };

        let mut actual_min = f64::INFINITY;
        let mut actual_max = f64::NEG_INFINITY;

        for (x, y) in locations.iter() {
            let value = self.cells[x + y * self.width].value;

            actual_min = actual_min.min(value);
            actual_max = actual_max.max(value);
        }

        for (x, y) in locations.iter() {
            let value = &mut self.cells[x + y * self.width].value;

            *value = (*value - actual_min) / (actual_max - actual_min) * (max - min) + min;
        }
    }

    pub fn all_matches(
        &self,
        pattern: &Grid,
        matches: &mut Vec<(usize, usize, Direction)>,
    ) -> usize {
        let mut total = 0;

        for direction in [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ] {
            total += self.matches(pattern, matches, direction);
        }

        total
    }

    pub fn matches(
        &self,
        pattern: &Grid,
        matches: &mut Vec<(usize, usize, Direction)>,
        direction: Direction,
    ) -> usize {
        let pattern_size = direction.apply_size(pattern.width, pattern.height);

        if let Some(cache) = self.caches.get(&pattern_size) {
            let pattern_hash = pattern.kind_hash(direction);

            if let Some(locations) = cache.inverse.get(&pattern_hash) {
                matches.extend(locations.iter().map(|(x, y)| (*x, *y, direction)));

                locations.len()
            } else {
                0
            }
        } else {
            let mut found = AHashSet::new();

            let (rotated_width, rotated_height) =
                direction.apply_size(pattern.width, pattern.height);
            let inverse = direction.inverse();

            let Some((count, best_kind)) = pattern
                .inverse
                .iter()
                .filter_map(|(k, l)| {
                    if l.is_empty() {
                        None
                    } else {
                        Some((self.inverse.get(k).map(|l| l.len()).unwrap_or_default(), *k))
                    }
                })
                .min()
            else {
                return 0;
            };

            if count == 0 {
                return 0;
            }

            for (original_i, original_j) in pattern.inverse.get(&best_kind).unwrap() {
                let (i, j) =
                    direction.apply(*original_i, *original_j, pattern.width, pattern.height);

                let Some(locations) = self
                    .inverse
                    .get(&pattern.cells[original_i + original_j * pattern.width].kind)
                else {
                    return 0;
                };

                if locations.is_empty() {
                    return 0;
                }
                'locations: for (x, y) in locations {
                    if i > *x
                        || j > *y
                        || (rotated_width + x) >= self.width
                        || (rotated_height + y) >= self.height
                    {
                        continue;
                    }

                    let origin_x = x - i;
                    let origin_y = y - j;

                    for j in 0..rotated_height {
                        for i in 0..rotated_width {
                            let (original_i, original_j) =
                                inverse.apply(i, j, rotated_width, rotated_height);
                            let actual =
                                self.cells[(origin_x + i) + (origin_y + j) * self.width].kind;
                            let expected =
                                pattern.cells[original_i + original_j * pattern.width].kind;

                            if actual != expected {
                                continue 'locations;
                            }
                        }
                    }

                    let location_x = x - i;
                    let location_y = y - j;

                    if !found.contains(&(location_x, location_y, direction)) {
                        found.insert((location_x, location_y, direction));
                    }
                }
            }

            let count = found.len();
            matches.extend(found);
            count
        }
    }

    fn update_cache(&mut self, x_min: usize, y_min: usize, x_max: usize, y_max: usize) {
        for cache in self.caches.values_mut() {
            cache.update(&self.cells, x_min, y_min, x_max, y_max);
        }
    }
}

impl PatternCache {
    pub fn new(
        cells: &[Cell],
        width: usize,
        height: usize,
        pattern_width: usize,
        pattern_height: usize,
    ) -> Self {
        assert!(width >= pattern_width);
        assert!(height >= pattern_height);

        let mut hashes = Vec::with_capacity((width - pattern_width) * (height - pattern_height));
        let mut inverse = AHashMap::new();

        for y in 0..=(height - pattern_height) {
            for x in 0..=(width - pattern_width) {
                let mut hasher = AHasher::default();

                for j in y..(y + pattern_height) {
                    for i in x..(x + pattern_width) {
                        cells[i + j * width].kind.hash(&mut hasher);
                    }
                }

                let hash = hasher.finish();
                hashes.push(hash);

                let locations: &mut AHashSet<_> = inverse.entry(hash).or_default();
                locations.insert((x, y));
            }
        }

        Self {
            pattern_width,
            pattern_height,
            width,
            height,
            hashes,
            inverse,
        }
    }

    pub fn update(
        &mut self,
        cells: &[Cell],
        x_min: usize,
        y_min: usize,
        x_max: usize,
        y_max: usize,
    ) {
        let min_x_affected = x_min - x_min.min(self.pattern_width);
        let min_y_affected = y_min - y_min.min(self.pattern_height);

        let max_x_affected = x_max.min(self.width - self.pattern_width - 1);
        let max_y_affected = y_max.min(self.height - self.pattern_height - 1);

        for y in min_y_affected..=max_y_affected {
            for x in min_x_affected..=max_x_affected {
                // Remove the old hash from the inverse mapping, we will lose the hash so
                // this must be done now
                let hash = &mut self.hashes[x + y * (self.width - self.pattern_width)];
                let locations = self
                    .inverse
                    .get_mut(&hash)
                    .expect("pattern cache out of date");
                locations.remove(&(x, y));

                // Recompute hash based on this location
                let mut hasher = AHasher::default();

                for j in y..(y + self.pattern_height) {
                    for i in x..(x + self.pattern_width) {
                        cells[i + j * self.width].kind.hash(&mut hasher);
                    }
                }

                *hash = hasher.finish();

                let locations = self.inverse.entry(*hash).or_default();
                locations.insert((x, y));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ustr::ustr;

    use super::{Cell, Direction, Grid};

    #[test]
    fn test_matches_uncached() {
        let mut grid = Grid::new(
            5,
            5,
            Cell {
                kind: ustr("A"),
                value: 1.0,
            },
        );

        grid.set(
            2,
            2,
            Cell {
                kind: ustr("B"),
                value: 1.0,
            },
        );

        let mut matches = Vec::new();

        let mut pattern = Grid::new(
            2,
            1,
            Cell {
                kind: ustr("B"),
                value: 1.0,
            },
        );
        pattern.set(
            0,
            0,
            Cell {
                kind: ustr("A"),
                value: 1.0,
            },
        );

        assert_eq!(grid.matches(&pattern, &mut matches, Direction::North), 1);
        assert_eq!(grid.matches(&pattern, &mut matches, Direction::South), 1);
        assert_eq!(grid.matches(&pattern, &mut matches, Direction::East), 1);
        assert_eq!(grid.matches(&pattern, &mut matches, Direction::West), 1);
        assert_eq!(matches.len(), 4);
    }

    #[test]
    fn test_matches_cached() {
        let mut grid = Grid::new(
            5,
            5,
            Cell {
                kind: ustr("A"),
                value: 1.0,
            },
        );

        grid.set(
            2,
            2,
            Cell {
                kind: ustr("B"),
                value: 1.0,
            },
        );

        println!("ADDING CACHES");

        grid.add_cache(1, 2);
        grid.add_cache(2, 1);

        let mut matches = Vec::new();

        let mut pattern = Grid::new(
            2,
            1,
            Cell {
                kind: ustr("B"),
                value: 1.0,
            },
        );
        pattern.set(
            0,
            0,
            Cell {
                kind: ustr("A"),
                value: 1.0,
            },
        );

        println!("FINDING MATCHES");

        assert_eq!(grid.matches(&pattern, &mut matches, Direction::North), 1);
        assert_eq!(grid.matches(&pattern, &mut matches, Direction::South), 1);
        assert_eq!(grid.matches(&pattern, &mut matches, Direction::East), 1);
        assert_eq!(grid.matches(&pattern, &mut matches, Direction::West), 1);
        assert_eq!(matches.len(), 4);
    }
}
