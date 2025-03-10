use core::f64;

use noise::NoiseFn;
use ustr::Ustr;

use crate::grid::Grid;

impl Grid {
    pub fn apply_noise(
        &mut self,
        kind: impl Into<Ustr>,
        scale: (f64, f64),
        noise: &impl NoiseFn<f64, 2>,
    ) {
        let kind = kind.into();

        let Some(locations) = self.inverse.remove(&kind) else {
            return;
        };

        let (sx, sy) = scale;

        for (x, y) in &locations {
            let fx = *x as f64 / sx;
            let fy = *y as f64 / sy;
            let value = noise.get([fx, fy]);

            self.set_value(*x, *y, value);
        }

        self.inverse.insert(kind, locations);
    }
}
