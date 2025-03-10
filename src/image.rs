use image::RgbImage;
use ustr::{ustr, UstrMap};

use crate::grid::Grid;

#[derive(Default)]
pub struct Palette {
    pub colors: UstrMap<[u8; 3]>,
}

impl Palette {
    pub fn pico8() -> Self {
        let mut palette = Self::default();

        palette.colors.insert(ustr("B"), [0, 0, 0]);
        palette.colors.insert(ustr("I"), [29, 43, 83]);
        palette.colors.insert(ustr("P"), [126, 37, 83]);
        palette.colors.insert(ustr("E"), [0, 135, 81]);
        palette.colors.insert(ustr("N"), [171, 82, 54]);
        palette.colors.insert(ustr("D"), [95, 87, 79]);
        palette.colors.insert(ustr("A"), [194, 195, 199]);
        palette.colors.insert(ustr("W"), [255, 241, 232]);
        palette.colors.insert(ustr("R"), [255, 0, 77]);
        palette.colors.insert(ustr("O"), [255, 163, 0]);
        palette.colors.insert(ustr("Y"), [255, 236, 39]);
        palette.colors.insert(ustr("G"), [0, 228, 54]);
        palette.colors.insert(ustr("U"), [41, 173, 255]);
        palette.colors.insert(ustr("S"), [131, 118, 156]);
        palette.colors.insert(ustr("K"), [255, 119, 168]);
        palette.colors.insert(ustr("F"), [255, 204, 170]);

        palette
    }
}

impl Grid {
    pub fn export(&self, palette: &Palette) -> RgbImage {
        let mut image = RgbImage::new(self.width as u32, self.height as u32);

        for y in 0..self.height {
            for x in 0..self.width {
                let kind = self.cells[x + y * self.width].kind;
                if let Some(color) = palette.colors.get(&kind) {
                    image.put_pixel(x as u32, y as u32, image::Rgb(*color));
                }
            }
        }

        image
    }
}
