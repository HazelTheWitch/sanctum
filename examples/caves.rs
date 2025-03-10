use std::{str::FromStr, time::Instant};

use rand::{rng, Rng};
use sanctum::{
    grid::{Cell, Direction, Grid},
    image::Palette,
    walk::RandomWalk,
};
use ustr::ustr;

fn main() {
    let size = 256;

    let origins = [
        (size / 4, size / 3),
        (size / 2, size / 2),
        (size - 32, size - 32),
    ];

    let mut grid = Grid::new(size, size, ustr("B"));

    grid.add_caches(3, 3);
    grid.add_caches(3, 2);
    grid.add_caches(1, 2);

    let mut rng = rng();

    let mut count = grid.count("W");

    for origin in origins {
        for (x, y) in RandomWalk::new(origin, ((0, 0), (size, size)), 10, &mut rng) {
            grid.set(x, y, ustr("W"));

            if grid.count("W") >= count + (size * size / 5) / origins.len() {
                break;
            }
        }
        count = grid.count("W");
    }

    grid.export(&Palette::pico8())
        .save("caves_pre.png")
        .unwrap();

    let lone_wall = Grid::from_str("WWW/WBW/WWW").unwrap();
    let empty = Grid::new(3, 3, ustr("W"));

    let mut matches = Vec::new();
    grid.matches(&lone_wall, &mut matches, Direction::North);
    grid.place_many(matches, &empty);

    let mut origins: Vec<_> = origins.into_iter().collect();
    for (x, y) in &origins {
        grid.set(*x, *y, ustr("G"));
    }

    for _ in 0..10 {
        let rectangles = grid.decompose(&mut rng, "W", Some(10));

        if let Some(((x, y), distances)) = rectangles
            .into_iter()
            .map(|((min_x, min_y), (max_x, max_y))| {
                grid.open_area((min_x + max_x) / 2, (min_y + max_y) / 2)
            })
            .max_by_key(|(_, distances)| distances.iter().sum::<usize>())
        {
            grid.set(x, y, ustr("G"));

            origins.push((x, y));

            let pattern = Grid::from_str("GW").unwrap();
            let stamp = Grid::from_str("GG").unwrap();

            for _ in 0..(distances.into_iter().max().unwrap_or(0)) {
                let mut matches = Vec::new();
                let mut all_matches = Vec::new();
                grid.all_matches(&pattern, &mut matches);
                all_matches.extend(matches.drain(..).filter(|_| rng.random_bool(0.3)));
                grid.place_many(all_matches, &stamp);
            }

            grid.replace("G", "R");
        };
    }

    let cost = |_, c: &Cell| match c.kind.as_str() {
        "R" | "W" => Some(c.value),
        "B" => Some(c.value * 32.0),
        _ => None,
    };

    for (i, origin) in origins.iter().enumerate() {
        for other in origins.iter().skip(i + 1) {
            if let Some(path) =
                grid.chiseled_path_between(&mut rng, *origin, *other, 4, 0.5, 20, &cost)
            {
                grid.set_many(path, ustr("R"));
            }
        }
    }

    let pattern = Grid::from_str("RRR/RBR").unwrap();
    let stamp = Grid::new(3, 2, ustr("R"));

    let mut matches = Vec::new();

    while grid.all_matches(&pattern, &mut matches) > 0 {
        grid.place_many(matches.drain(..), &stamp);
    }

    grid.replace("W", "B");
    grid.replace("R", "W");

    let pattern = Grid::from_str("WBW").unwrap();
    let stamp = Grid::new(3, 1, ustr("W"));

    let mut matches = Vec::new();

    for _ in 0..3 {
        grid.all_matches(&pattern, &mut matches);
        grid.place_many(matches.drain(..).filter(|_| rng.random_bool(0.4)), &stamp);
    }

    grid.export(&Palette::pico8()).save("caves.png").unwrap();
}
