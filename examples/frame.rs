use busy_beaver_blaze::{LogStepIterator, PowerOfTwo, SpaceTimeMachine, BB5_CHAMP, BB6_CONTENDER};
use std::str::FromStr;
use std::{fs, path::Path};

#[derive(Debug)]
enum Resolution {
    Tiny,   // 320x180
    TwoK,   // 1920x1080
    FourK,  // 3840x2160
    EightK, // 7680x4320
}

impl FromStr for Resolution {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(Resolution::Tiny),
            "2k" => Ok(Resolution::TwoK),
            "4k" => Ok(Resolution::FourK),
            "8k" => Ok(Resolution::EightK),
            _ => Err(format!("Unknown resolution: {}. Use 2k, 4k, or 8k", s)),
        }
    }
}

impl Resolution {
    fn dimensions(&self) -> (u32, u32) {
        match self {
            Resolution::Tiny => (320, 180),
            Resolution::TwoK => (1920, 1080),
            Resolution::FourK => (3840, 2160),
            Resolution::EightK => (7680, 4320),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();

    let machine_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bb5_champ".to_string());

    let resolution = std::env::args()
        .nth(2)
        .map(|arg| Resolution::from_str(&arg))
        .transpose()?
        .unwrap_or(Resolution::Tiny); // cmk2K
    let (goal_x, goal_y) = resolution.dimensions();

    let x_smoothness = std::env::args()
        .nth(3)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1);
    let y_smoothness = std::env::args()
        .nth(4)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1);
    let run_id = std::env::args()
        .nth(5)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(0);

    let (up_x, up_y) = (goal_x, goal_y);
    let end_step = 18_349_821;
    let num_frames = 2;

    let (max_x_2, max_y_2) = (
        PowerOfTwo::from_exp(x_smoothness).log2(),
        PowerOfTwo::from_exp(y_smoothness).log2(),
    );

    let mut space_time_machine = match machine_name.as_str() {
        "bb5_champ" => SpaceTimeMachine::from_str(BB5_CHAMP, up_x, up_y, max_x_2, max_y_2)?,
        "bb6_contender" => SpaceTimeMachine::from_str(BB6_CONTENDER, up_x, up_y, max_x_2, max_y_2)?,
        "bb6_contender2" => {
            SpaceTimeMachine::from_str(BB6_CONTENDER, up_x, up_y, max_x_2, max_y_2)?
        }
        "bb5_1RB1RE_0RC1RA_1RD0LD_1LC1LB_0RA---" => SpaceTimeMachine::from_str(
            "1RB1RE_0RC1RA_1RD0LD_1LC1LB_0RA---",
            up_x,
            up_y,
            max_x_2,
            max_y_2,
        )?,
        _ => Err(format!("Unknown machine: {}", machine_name))?,
    };

    let log_iter = LogStepIterator::new(end_step, num_frames);

    for goal_step_index in log_iter {
        if goal_step_index == 0 {
            continue;
        }
        let actual_step_index = space_time_machine.step_count() - 1;
        if goal_step_index > actual_step_index
            && !space_time_machine.nth_js(goal_step_index - actual_step_index - 1)
        {
            break;
        }
        let f = format!(
            r"m:\deldir\bb\frame\{run_id}\{machine_name}_{resolution:?}_{x_smoothness}_{y_smoothness}_{goal_step_index}_{:?}.png",
            start.elapsed()
        );
        let out_info = Path::new(f.as_str());
        fs::create_dir_all(out_info.parent().unwrap())?;

        let png_data = space_time_machine.png_data();
        let base = image::load_from_memory(&png_data)?;
        base.save(out_info)?;
    }

    println!("Elapsed: {:?}", start.elapsed());
    Ok(())
}
