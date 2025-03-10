use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::pathfinding::neighbors;

pub struct RandomWalk {
    position: (usize, usize),
    scratch: Vec<(usize, usize)>,
    bounds: ((usize, usize), (usize, usize)),
    rng: StdRng,
}

impl RandomWalk {
    pub fn new(
        start: (usize, usize),
        bounds: ((usize, usize), (usize, usize)),
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            position: start,
            scratch: Vec::with_capacity(4),
            bounds,
            rng: StdRng::from_rng(rng),
        }
    }
}

impl Iterator for RandomWalk {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let (x, y) = self.position;

        let ((min_x, min_y), (max_x, max_y)) = self.bounds;

        self.scratch.extend(
            neighbors(x, y, max_x, max_y)
                .filter(|(x, y)| x >= &min_x && y >= &min_y && x < &max_x && y < &max_y),
        );

        self.position = self
            .scratch
            .swap_remove(self.rng.random_range(0..self.scratch.len()));

        self.scratch.clear();

        Some((x, y))
    }
}
