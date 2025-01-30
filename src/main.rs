use std::ops::Index;

fn main() {
    let program = Program([
        PerState([
            PerInput { next_state: 1, direction: Direction::Right },
            PerInput { next_state: 2, direction: Direction::Left },
        ]),
        PerState([
            PerInput { next_state: 3, direction: Direction::Right },
            PerInput { next_state: 4, direction: Direction::Left },
        ]),
        PerState([
            PerInput { next_state: 5, direction: Direction::Right },
            PerInput { next_state: 0, direction: Direction::Left },
        ]),
        PerState([
            PerInput { next_state: 1, direction: Direction::Right },
            PerInput { next_state: 2, direction: Direction::Left },
        ]),
        PerState([
            PerInput { next_state: 3, direction: Direction::Right },
            PerInput { next_state: 4, direction: Direction::Left },
        ]),
    ]);
}

#[derive(Default)]
struct Tape {
    nonnegative: Vec<u8>,
    negative: Vec<u8>,
}

impl Index<i32> for Tape {
    type Output = u8;

    fn index(&mut self, index: i32) -> &u8 {
        let (index, vec) =
        if index >= 0 {
            (index as usize, &mut self.nonnegative)
        } else {
            ((-index - 1) as usize, &mut self.negative)
        };
            
        if vec.len() <= index {
            vec.resize(index + 1, 0u8);
        }

        &vec[index]

    }
}



struct Machine {
    tape: Tape,
    program: Program,
    state: u8,
}

struct Program (
    // array of 5 PerState structs
    [PerState; 5],
);

struct PerState (
    // array of 2 PerInput structs
    [PerInput; 2],
);

struct PerInput {
    next_state: u8,
    direction: Direction
};

enum Direction {
    Left,
    Right,
}
