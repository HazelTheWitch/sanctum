use core::f64;
use std::{
    collections::BinaryHeap,
    iter::{from_fn, once},
    usize,
};

use ahash::{AHashMap, AHashSet};
use ordered_float::{Float, NotNan};
use rand::Rng;

use crate::{
    grid::{Cell, Grid},
    line::Line,
};

impl Grid {
    pub fn chiseled_path_between(
        &self,
        rng: &mut impl Rng,
        start: (usize, usize),
        end: (usize, usize),
        deviation: usize,
        quality: f64,
        split_every: usize,
        cost: impl Fn((usize, usize), &Cell) -> Option<f64>,
    ) -> Option<Vec<(usize, usize)>> {
        let mut path: Vec<(usize, usize)> = self.path_between(start, end, &cost)?.collect();

        self.perturb_path_chiseling(rng, &mut path, deviation, quality, split_every, &cost);

        Some(path)
    }

    pub fn perturb_path_chiseling(
        &self,
        rng: &mut impl Rng,
        path: &mut Vec<(usize, usize)>,
        deviation: usize,
        quality: f64,
        split_every: usize,
        cost: &impl Fn((usize, usize), &Cell) -> Option<f64>,
    ) {
        if split_every != 0 {
            let new_path = path
                .chunks(split_every)
                .flat_map(move |chunk| {
                    let mut path = chunk.into_iter().copied().collect();

                    self.perturb_path_chiseling(
                        rng,
                        &mut path,
                        deviation.min(split_every),
                        quality,
                        0,
                        cost,
                    );

                    path
                })
                .collect();
            *path = new_path;
            return;
        }

        let Some(&start) = path.first() else {
            return;
        };

        let Some(&end) = path.last() else {
            return;
        };

        if start == end {
            return;
        }

        let mut open: Vec<(f64, usize, usize)> = Default::default();
        let mut open_set: AHashSet<(usize, usize)> = Default::default();
        let mut allowed: AHashSet<(usize, usize)> = Default::default();

        let mut total_cost = 0.0;

        for (px, py) in path.iter().copied() {
            for x in (px - px.min(deviation))..=(px + deviation).min(self.width - 1) {
                for y in (py - py.min(deviation))..=(py + deviation).min(self.height - 1) {
                    if !open_set.contains(&(x, y)) {
                        if let Some(mut cost) = cost((x, y), self.get(x, y).unwrap()) {
                            cost = cost.max(f64::EPSILON);

                            open_set.insert((x, y));
                            open.push((cost, x, y));
                            total_cost += cost;
                        }
                    }
                }
            }
        }

        let total_cells = open.len();
        let cutoff = ((1.0 - quality) * total_cells as f64).round() as usize;

        while open.len() >= cutoff {
            if open.is_empty() {
                return;
            }

            let mut choice = rng.random_range(0.0..total_cost) - open[0].0;
            let mut index = 0;

            while index < open.len() - 1 && choice > 0.0 {
                index += 1;
                choice -= open[index].0;
            }

            let (c, x, y) = open.swap_remove(index);

            total_cost -= c;

            open_set.remove(&(x, y));

            if path.contains(&(x, y)) {
                if let Some(new_path) = self.path_between(start, end, |p, cell| {
                    if !(open_set.contains(&p) || allowed.contains(&p)) {
                        return None;
                    }

                    cost(p, cell)
                }) {
                    path.clear();
                    path.extend(new_path);

                    continue;
                }

                allowed.insert((x, y));
            }
        }
    }

    pub fn perturb_fractal(
        &self,
        rng: &mut impl Rng,
        mut line: &mut Vec<(usize, usize)>,
        iterations: usize,
        step_by: usize,
        perturb: usize,
        cost: &impl Fn((usize, usize), &Cell) -> Option<f64>,
    ) {
        let mut current_step_by = step_by.pow(iterations as u32 - 1);
        let mut current_perturbation_iterations = perturb.pow(iterations as u32 - 1);

        if line.len() < 2 {
            return;
        }

        for _ in 0..iterations {
            let start = line[0];
            let end = line[line.len() - 1];

            *line = line
                .windows(2)
                .flat_map(|points| {
                    let mut line = Line::new(points[0], points[1]);
                    let first = line.next().unwrap();
                    once(first)
                        .chain(line.step_by(current_step_by))
                        .chain(once(points[1]))
                })
                .collect();

            line.dedup();

            self.perturb_path_gradient(rng, &mut line, current_perturbation_iterations, cost);

            relax_path_recursive(&mut line);

            line[0] = start;
            let length = line.len() - 1;
            line[length] = end;

            current_step_by /= step_by;
            current_perturbation_iterations /= perturb;
        }
    }

    pub fn fractal_between(
        &self,
        rng: &mut impl Rng,
        waypoints: impl IntoIterator<Item = (usize, usize)>,
        iterations: usize,
        step_by: usize,
        perturb: usize,
        cost: &impl Fn((usize, usize), &Cell) -> Option<f64>,
    ) -> Vec<(usize, usize)> {
        let waypoints: Vec<_> = waypoints.into_iter().collect();

        let mut line = Vec::new();

        for window in waypoints.windows(2) {
            let start = window[0];
            let end = window[1];

            let mut section = vec![start, end];

            self.perturb_fractal(rng, &mut section, iterations, step_by, perturb, cost);

            line.extend(section);
        }

        line = line
            .windows(2)
            .flat_map(|points| Line::new(points[0], points[1]))
            .collect();

        line.dedup();

        line
    }

    pub fn perturb_path_gradient(
        &self,
        rng: &mut impl Rng,
        waypoints: &mut Vec<(usize, usize)>,
        iterations: usize,
        cost: impl Fn((usize, usize), &Cell) -> Option<f64>,
    ) {
        let mut scratch = Vec::with_capacity(4);

        for _ in 0..iterations {
            for (wx, wy) in waypoints.iter_mut() {
                let mut total_cost = 0.0;

                for (nx, ny) in neighbors(*wx, *wy, self.width, self.height) {
                    if let Some(cost) = cost((nx, ny), &self.cells[nx + ny * self.width]) {
                        if cost < 0.0 || !cost.is_finite() {
                            continue;
                        }

                        scratch.push(((nx, ny), cost));
                        total_cost += cost;
                    }
                }

                if scratch.len() == 0 {
                    continue;
                }

                let mut choice = rng.random_range(0.0..total_cost) - scratch[0].1;
                let mut index = 0;

                while index < scratch.len() - 1 && choice > 0.0 {
                    index += 1;
                    choice -= scratch[index].1;
                }

                let (cx, cy) = scratch[index].0;
                *wx = cx;
                *wy = cy;

                scratch.clear();
            }
        }
    }

    pub fn path_between(
        &self,
        start: (usize, usize),
        end: (usize, usize),
        cost: impl Fn((usize, usize), &Cell) -> Option<f64>,
    ) -> Option<impl Iterator<Item = (usize, usize)>> {
        let mut came_from = AHashMap::new();

        let h = |(x, y)| {
            ((end.0.abs_diff(x) as f64).powi(2) + (end.1.abs_diff(y) as f64).powi(2)).sqrt()
        };

        let h_start = NotNan::new(h(start)).ok()?;

        let mut open = BinaryHeap::new();
        open.push(CostLocation {
            location: start,
            cost: h_start,
        });

        let mut g_cache = AHashMap::new();
        g_cache.insert(start, NotNan::new(0.0).unwrap());

        loop {
            let Some(next) = open.pop() else {
                break;
            };

            let next = next.location;

            if next == end {
                return Some(reconstruct_path(came_from, next));
            }

            for (nx, ny) in neighbors(next.0, next.1, self.width, self.height) {
                let Some(cost) = cost((nx, ny), &self.cells[nx + ny * self.width]) else {
                    continue;
                };

                if cost < 0.0 {
                    continue;
                }

                let tentative_g = NotNan::new(
                    g_cache
                        .get(&next)
                        .unwrap_or(&NotNan::new(f64::INFINITY).unwrap())
                        + cost,
                )
                .ok()?;

                if tentative_g
                    < *g_cache
                        .get(&(nx, ny))
                        .unwrap_or(&NotNan::new(f64::INFINITY).unwrap())
                {
                    came_from.insert((nx, ny), next);
                    g_cache.insert((nx, ny), tentative_g);

                    let f = NotNan::new(tentative_g + h((nx, ny))).ok()?;

                    open.push(CostLocation {
                        location: (nx, ny),
                        cost: f,
                    });
                }
            }
        }

        return None;
    }
}

pub fn relax_path(waypoints: &mut Vec<(usize, usize)>) -> bool {
    waypoints.dedup();

    let mut changed = false;

    for i in (1..(waypoints.len() - 1)).rev() {
        let (ax, ay) = waypoints[i - 1];
        let (bx, by) = waypoints[i];
        let (cx, cy) = waypoints[i + 1];

        let (dx1, dy1) = (bx as i64 - ax as i64, by as i64 - ay as i64);
        let (dx2, dy2) = (cx as i64 - bx as i64, cy as i64 - by as i64);

        let dot = dx1 * dx2 + dy1 * dy2;

        if dot <= 0 {
            let new0 = ((ax + bx) / 2, (ay + by) / 2);
            let new1 = ((bx + cx) / 2, (by + cy) / 2);

            if new0 == (ax, ay) || new0 == (bx, by) || new1 == (bx, by) || new1 == (cx, cy) {
                continue;
            }

            if new0 != new1 {
                waypoints[i] = ((ax + bx) / 2, (ay + by) / 2);
                waypoints.insert(i + 1, ((bx + cx) / 2, (by + cy) / 2));
            } else {
                waypoints[i] = new0;
            }
            changed = true;
        }
    }

    changed
}

pub fn relax_path_recursive(waypoints: &mut Vec<(usize, usize)>) -> bool {
    let mut changed = false;

    while relax_path(waypoints) {
        changed = true;
    }

    changed
}

struct CostLocation {
    pub cost: NotNan<f64>,
    pub location: (usize, usize),
}

impl PartialEq for CostLocation {
    fn eq(&self, other: &Self) -> bool {
        self.cost.eq(&other.cost)
    }
}

impl Eq for CostLocation {}

impl PartialOrd for CostLocation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cost.partial_cmp(&other.cost)?.reverse())
    }
}

impl Ord for CostLocation {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost.cmp(&other.cost).reverse()
    }
}

fn reconstruct_path(
    mut came_from: AHashMap<(usize, usize), (usize, usize)>,
    mut end: (usize, usize),
) -> impl Iterator<Item = (usize, usize)> {
    once(end).chain(from_fn(move || {
        end = came_from.remove(&end)?;

        Some(end)
    }))
}

pub fn neighbors(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = (usize, usize)> {
    [(1, 0), (0, 1), (-1, 0), (0, -1)]
        .into_iter()
        .filter_map(move |(dx, dy)| {
            let x = x as i64 + dx;
            let y = y as i64 + dy;

            if x < 0 || y < 0 || x >= width as i64 || y >= height as i64 {
                return None;
            }

            Some((x as usize, y as usize))
        })
}
