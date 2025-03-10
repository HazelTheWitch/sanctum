use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::pathfinding::neighbors;

pub struct RandomWalk {
    position: (usize, usize),
    scratch: Vec<((usize, usize), usize)>,
    bounds: ((usize, usize), (usize, usize)),
    smoothing_radius: usize,
    rng: StdRng,
}

impl RandomWalk {
    pub fn new(
        start: (usize, usize),
        bounds: ((usize, usize), (usize, usize)),
        smoothing_radius: usize,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            position: start,
            scratch: Vec::with_capacity(4),
            bounds,
            smoothing_radius,
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
                .filter(|(x, y)| x >= &min_x && y >= &min_y && x < &max_x && y < &max_y)
                .map(|(x, y)| {
                    (
                        (x, y),
                        usize::min(
                            x.abs_diff(min_x).min(x.abs_diff(max_x)),
                            y.abs_diff(min_y).min(y.abs_diff(max_y)),
                        )
                        .min(self.smoothing_radius)
                            + 1,
                    )
                }),
        );

        let total_weight = self.scratch.iter().map(|(_, w)| w).sum();

        let mut choice = self.rng.random_range(0..total_weight);
        let mut index = 0;

        while choice >= self.scratch[index].1 {
            choice -= self.scratch[index].1;
            index += 1;
        }

        self.position = self.scratch[index].0;

        self.scratch.clear();

        Some((x, y))
    }
}
