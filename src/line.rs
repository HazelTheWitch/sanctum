#[derive(Debug)]
pub struct Line {
    start: (i64, i64),
    end: (i64, i64),
    delta: (i64, i64),
    sign: (i64, i64),
    error: i64,
    completed: bool,
}

impl Line {
    pub fn new(start: (usize, usize), end: (usize, usize)) -> Self {
        let (x0, y0) = start;
        let (x1, y1) = end;

        let (x0, y0) = (x0 as i64, y0 as i64);
        let (x1, y1) = (x1 as i64, y1 as i64);

        let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
        Self {
            start: (x0, y0),
            end: (x1, y1),
            delta: (dx, dy),
            sign: (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 }),
            error: dx + dy,
            completed: false,
        }
    }
}

impl Iterator for Line {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.completed {
            return None;
        }

        let x = self.start.0 as usize;
        let y = self.start.1 as usize;

        let e2 = 2 * self.error;

        if e2 >= self.delta.1 {
            if self.start.0 == self.end.0 {
                self.completed = true;
            }
            self.error += self.delta.1;
            self.start.0 += self.sign.0;
        }
        if e2 <= self.delta.0 {
            if self.start.1 == self.end.1 {
                self.completed = true;
            }
            self.error += self.delta.0;
            self.start.1 += self.sign.1;
        }

        Some((x, y))
    }
}
