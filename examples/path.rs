use std::{iter::from_fn, str::FromStr, usize};

use image::imageops::{resize, FilterType};
use noise::{Constant, HybridMulti, Max, MultiFractal, ScaleBias, Simplex};
use rand::{rngs::StdRng, Rng, SeedableRng};
use sanctum::{
    grid::{Cell, Direction, Grid},
    image::Palette,
};
use ustr::ustr;

fn main() {
    let size = 256;
    let deviation = 32;

    let mut grid = Grid::new(
        size,
        size,
        Cell {
            kind: ustr("B"),
            value: 1.0,
        },
    );

    let mut rng = StdRng::seed_from_u64(16);

    grid.apply_noise(
        ustr("B"),
        (size as f64 / 0.5, size as f64 / 0.5),
        &HybridMulti::new(rng.random())
            .set_sources(
                from_fn(|| Some(Simplex::new(rng.random())))
                    .take(4)
                    .collect(),
            )
            .set_octaves(4)
            .set_persistence(0.8),
    );

    println!("Created Mountains");

    grid.map(ustr("B"), |c| {
        if c.value > 0.4 {
            Some(ustr("D").into())
        } else {
            None
        }
    });

    for y in 0..grid.height() {
        for x in 0..grid.width() {
            if let Some(cell) = grid.get(x, y) {
                if cell.kind == ustr("B") && rng.random_bool(1.0 / 25.0) {
                    grid.set(x, y, ustr("E"));
                }
            }
        }
    }

    let stamp = Grid::new(4, 4, ustr("B"));

    grid.place_many(
        grid.path_between((0, 0), (size - 1, size - 1), |_, c: &Cell| {
            match c.kind.as_str() {
                "E" | "D" => Some(100.0),
                _ => Some(1.0),
            }
        })
        .unwrap()
        .into_iter()
        .map(|(x, y)| (x, y, Direction::North)),
        &stamp,
    );

    println!("Dug Path");

    grid.set(0, 0, ustr("R"));
    grid.set(size - 1, size - 1, ustr("G"));

    let cost = |_, c: &Cell| match c.kind.as_str() {
        "E" | "D" => None,
        _ => Some(c.value),
    };

    grid.normalize_values("B", 0.5, 1.5);

    println!("Initalized Costs");

    let mut path: Vec<_> = grid
        .path_between((0, 0), (size - 1, size - 1), cost)
        .expect("accidentally blocked path")
        .collect();

    grid.set_many(path.iter().copied(), ustr("U"));

    println!("Calculated Optimal Path");
    grid.perturb_path_chiseling(&mut rng, &mut path, deviation, 1.0, 32, &cost);

    grid.set_many(path, ustr("F"));

    println!("Chiseled path");

    let pattern = Grid::from_str("FB").unwrap();

    let mut matches = Vec::new();

    grid.all_matches(&pattern, &mut matches);

    matches.retain(|_| rng.random_bool(0.25));

    grid.place_many(matches, &Grid::new(2, 2, ustr("F")));

    grid.export(&Palette::pico8()).save("path.png").unwrap();
}
