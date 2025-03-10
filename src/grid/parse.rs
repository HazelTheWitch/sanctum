use nom::{
    branch::alt,
    bytes::complete::{is_not, tag},
    character::complete::{char, none_of},
    multi::{many1, separated_list1},
    sequence::delimited,
    IResult, Parser,
};
use ustr::ustr;

use super::{Cell, Grid};

fn parse_cell(i: &str) -> IResult<&str, Cell> {
    alt((
        delimited(char('('), is_not(")"), char(')')).map(|c| Cell::from(ustr(c))),
        none_of("/").map(|c| Cell::from(ustr(&c.to_string()))),
    ))
    .parse(i)
}

fn parse_row(i: &str) -> IResult<&str, Vec<Cell>> {
    many1(parse_cell).parse(i)
}

pub fn parse_grid(i: &str) -> IResult<&str, Option<Grid>> {
    let (i, rows) = separated_list1(tag("/"), parse_row).parse(i)?;

    let height = rows.len();

    if height == 0 {
        return Ok((i, None));
    }

    let width = rows[0].len();

    if rows.iter().any(|r| r.len() != width) {
        return Ok((i, None));
    }

    Ok((
        i,
        Grid::from_cells(width, height, rows.into_iter().flatten().collect()),
    ))
}
