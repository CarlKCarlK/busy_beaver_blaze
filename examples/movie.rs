use ab_glyph::{FontArc, PxScale};
use busy_beaver_blaze::{BB5_CHAMP, BB6_CONTENDER, LogStepIterator, SpaceTimeMachine};
use core::str::FromStr;
use image::Rgba;
use image::{DynamicImage, imageops::FilterType};
use imageproc::drawing::draw_text_mut;
use std::{
    fs,
    path::{Path, PathBuf},
};
use thousands::Separable;

#[derive(Debug)]
enum Resolution {
    Tiny,   // 320x180
    TwoK,   // 1920x1080
    FourK,  // 3840x2160
    EightK, // 7680x4320
}

impl FromStr for Resolution {
    type Err = String;

    #[allow(clippy::min_ident_chars)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(Self::Tiny),
            "2k" => Ok(Self::TwoK),
            "4k" => Ok(Self::FourK),
            "8k" => Ok(Self::EightK),
            _ => Err(format!("Unknown resolution: {s}. Use 2k, 4k, or 8k")),
        }
    }
}

impl Resolution {
    const fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Tiny => (320, 180),
            Self::TwoK => (1920, 1080),
            Self::FourK => (3840, 2160),
            Self::EightK => (7680, 4320),
        }
    }
}

#[allow(clippy::shadow_unrelated, clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn core::error::Error>> {
    let start = std::time::Instant::now();

    let machine_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bb5_champ".to_owned());

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

    let buffer1_count = 0; // cmk0000000

    let (up_x, up_y) = (goal_x, goal_y);
    let (mut space_time_machine, end_step, num_frames, (output_dir, run_id)) = match machine_name
        .as_str()
    {
        "bb5_champ" => {
            let machine = SpaceTimeMachine::from_str(
                BB5_CHAMP,
                up_x,
                up_y,
                x_smoothness,
                y_smoothness,
                buffer1_count,
            )?;
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\bb5_champ")?;
            (machine, 47_176_870, 1000, dir_info)
        }
        "bb6_contender" => {
            let machine = SpaceTimeMachine::from_str(
                BB6_CONTENDER,
                up_x,
                up_y,
                x_smoothness,
                y_smoothness,
                buffer1_count,
            )?;
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\bb6_contender")?;
            (machine, 1_000_000_000_000u64, 2000, dir_info)
        }
        "bb6_contender2" => {
            let machine = SpaceTimeMachine::from_str(
                BB6_CONTENDER,
                up_x,
                up_y,
                x_smoothness,
                y_smoothness,
                buffer1_count,
            )?;
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\bb6_contender2")?;
            (machine, 1_000_000_000u64, 1000, dir_info)
        }
        "bb5_1RB1RE_0RC1RA_1RD0LD_1LC1LB_0RA---" => {
            let machine = SpaceTimeMachine::from_str(
                "1RB1RE_0RC1RA_1RD0LD_1LC1LB_0RA---",
                up_x,
                up_y,
                x_smoothness,
                y_smoothness,
                buffer1_count,
            )?;
            let dir_info =
                create_sequential_subdir(r"m:\deldir\bb\bb5_1RB1RE_0RC1RA_1RD0LD_1LC1LB_0RA---")?;
            (machine, 1_000_000_000u64, 1000, dir_info)
        }
        _ => Err(format!("Unknown machine: {machine_name}"))?,
    };

    println!(
        "Using machine: {} with output in {:?}",
        machine_name, &output_dir
    );
    println!("Using resolution: {resolution:?} ({goal_x}x{goal_y})");

    let log_iter = LogStepIterator::new(end_step, num_frames);

    for (frame_index, goal_step_index) in log_iter.enumerate() {
        let actual_step_index = space_time_machine.step_count() - 1;
        if goal_step_index > actual_step_index
            && !space_time_machine.nth_js(goal_step_index - actual_step_index - 1)
        {
            break;
        }
        let actual_step_index = space_time_machine.step_count() - 1;
        println!(
            "run_id: {}, Frame {}, time so far {:?}",
            run_id,
            frame_index,
            start.elapsed()
        );

        save_frame(
            &mut space_time_machine,
            &output_dir,
            run_id,
            frame_index as u32,
            actual_step_index + 1,
            goal_x,
            goal_y,
        )?;
    }

    println!("Elapsed: {:?}", start.elapsed());
    Ok(())
}

fn save_frame(
    machine: &mut SpaceTimeMachine,
    output_dir: &Path,
    run_id: u32,
    frame: u32,
    step: u64,
    goal_x: u32,
    goal_y: u32,
) -> Result<(), Box<dyn core::error::Error>> {
    let base_file_name = output_dir.join(format!("base/{run_id}_{frame:07}.png"));
    let resized_file_name = output_dir.join(format!("resized/{run_id}_{frame:07}.png"));
    let metadata_file_name = output_dir.join(format!("metadata/{run_id}_{frame:07}.txt"));
    // create the 3 subdirectories if they don't exist
    fs::create_dir_all(base_file_name.parent().unwrap())?;
    fs::create_dir_all(resized_file_name.parent().unwrap())?;
    fs::create_dir_all(metadata_file_name.parent().unwrap())?;

    #[cfg(target_os = "linux")]
    let font_data = include_bytes!(
        r"/mnt/c/Program Files/WindowsApps/Microsoft.WindowsTerminal_1.21.10351.0_x64__8wekyb3d8bbwe/CascadiaMono.ttf"
    );
    #[cfg(target_os = "windows")]
    let font_data = include_bytes!(
        r"C:\Program Files\WindowsApps\Microsoft.WindowsTerminal_1.21.10351.0_x64__8wekyb3d8bbwe\CascadiaMono.ttf"
    );

    let font = FontArc::try_from_slice(font_data).map_err(|_| "Failed to load font")?;
    let scale = PxScale::from(50.0);

    let png_data = machine.png_data();
    let base = image::load_from_memory(&png_data)?;
    base.save(&base_file_name)?;

    // Resize and anti-alias the image
    let x_fraction = base.width() as f32 / goal_x as f32;
    let mut resized = if x_fraction < 0.25 {
        base.resize_exact(goal_x, goal_y, FilterType::Nearest)
    } else {
        let blurred = image::imageops::blur(&base, 1.0);
        DynamicImage::ImageRgba8(blurred).resize_exact(
            goal_x,
            goal_y,
            if x_fraction < 1.0 {
                FilterType::Lanczos3
            } else {
                FilterType::Nearest
            },
        )
    };

    // Calculate text position for lower right corner
    let text = format!("{:>75}", step.separate_with_commas());
    let text_height = 50.0; // Approximate height based on font size
    let y_position = goal_y as f32 - text_height - 10.0; // 10 pixels padding from bottom

    // save text to metadata text file
    let metadata = &text;
    fs::write(&metadata_file_name, metadata)?;

    draw_text_mut(
        &mut resized,
        Rgba([110, 110, 110, 255]), // Color
        25,                         // X position (keep same padding from edge)
        y_position as i32,          // Y position now near bottom
        scale,
        &font,
        &text,
    );
    resized.save(&resized_file_name)?;

    Ok(())
}

fn create_sequential_subdir(top_dir: &str) -> std::io::Result<(PathBuf, u32)> {
    // create top_dir if it doesn't exist
    fs::create_dir_all(top_dir)?;

    // Read all entries in the top directory
    let entries = fs::read_dir(top_dir)?;

    // Find the highest numbered subdirectory
    let mut max_num = 0;
    for entry in entries.flatten() {
        if entry.path().is_dir() {
            if let Some(num) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u32>().ok())
            {
                max_num = max_num.max(num);
            }
        }
    }

    // Create a new subdirectory with the next sequential number
    let new_dir_num = max_num + 1;
    let new_dir_path = Path::new(top_dir).join(new_dir_num.to_string());

    fs::create_dir_all(&new_dir_path)?; // Handle error appropriately

    Ok((new_dir_path, new_dir_num))
}
