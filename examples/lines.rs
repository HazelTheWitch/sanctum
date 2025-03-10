use rand::rng;
use sanctum::{
    grid::{Cell, Grid},
    image::Palette,
    line::Line,
};
use ustr::ustr;

fn main() {
    let size = 128;

    let mut grid = Grid::new(size, size, ustr("B"));

    let line = |x| vec![(x, 8), (x, size - 8)];

    let mut fractal = line(size / 4);
    grid.set_many(Line::new(fractal[0], fractal[1]), ustr("U"));

    let mut rng = rng();

    let cost = |_, c: &Cell| Some(c.value);

    grid.set_many(
        grid.fractal_between(&mut rng, fractal, 3, 4, 6, &cost),
        ustr("Y"),
    );

    let chiseled = line(size / 2);
    grid.set_many(Line::new(chiseled[0], chiseled[1]), ustr("U"));

    let chiseled = grid
        .chiseled_path_between(
            &mut rng,
            chiseled[0],
            chiseled[1],
            size / 8,
            1.0,
            0,
            |_, c: &Cell| Some(c.value),
        )
        .unwrap();
    grid.set_many(chiseled, ustr("E"));

    let chiseled = line(size * 3 / 4);
    grid.set_many(Line::new(chiseled[0], chiseled[1]), ustr("U"));

    let chiseled = grid
        .chiseled_path_between(
            &mut rng,
            chiseled[0],
            chiseled[1],
            size / 8,
            0.4,
            0,
            |_, c: &Cell| Some(c.value),
        )
        .unwrap();
    grid.set_many(chiseled, ustr("R"));

    grid.export(&Palette::pico8()).save("lines.png").unwrap();
}
