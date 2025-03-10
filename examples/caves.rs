use rand::rng;
use sanctum::{grid::Grid, image::Palette, walk::RandomWalk};
use ustr::ustr;

fn main() {
    let size = 256;
    let mut grid = Grid::new(size, size, ustr("B"));

    let mut rng = rng();

    for (x, y) in RandomWalk::new(
        (size / 2, size / 2),
        ((size / 4, size / 4), (3 * size / 4, 3 * size / 4)),
        &mut rng,
    ) {
        grid.set(x, y, ustr("W"));

        if grid.count("W") >= size * size / 8 {
            break;
        }
    }

    grid.export(&Palette::pico8()).save("caves.png").unwrap();
}
