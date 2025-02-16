use ab_glyph::{FontArc, PxScale};
use busy_beaver_blaze::{LogStepIterator, SpaceTimeMachine, BB5_CHAMP, BB6_CONTENDER};
use image::{imageops::FilterType, Rgb};
use imageproc::drawing::draw_text_mut;
use std::str::FromStr;
use std::{
    fs,
    path::{Path, PathBuf},
};
use thousands::Separable;

#[derive(Debug)]
enum Resolution {
    TwoK,   // 1920x1080
    FourK,  // 3840x2160
    EightK, // 7680x4320
}

impl FromStr for Resolution {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
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
            Resolution::TwoK => (1920, 1080),
            Resolution::FourK => (3840, 2160),
            Resolution::EightK => (7680, 4320),
        }
    }
}

fn main() -> Result<(), String> {
    let machine_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bb5_champ".to_string());

    let resolution = std::env::args()
        .nth(2)
        .map(|arg| Resolution::from_str(&arg))
        .transpose()?
        .unwrap_or(Resolution::TwoK);
    let (goal_x, goal_y) = resolution.dimensions();

    let (up_x, up_y) = (goal_x, goal_y);
    let (mut space_time_machine, end_step, num_frames, (output_dir, run_id)) =
        match machine_name.as_str() {
            "bb5_champ" => {
                let machine = SpaceTimeMachine::from_str(BB5_CHAMP, up_x, up_y)?;
                let dir_info =
                    create_sequential_subdir(r"m:\deldir\bb5_champ").map_err(|e| e.to_string())?;
                (machine, 47_176_870, 1000, dir_info)
            }
            "bb6_contender" => {
                let machine = SpaceTimeMachine::from_str(BB6_CONTENDER, up_x, up_y)?;
                let dir_info = create_sequential_subdir(r"m:\deldir\bb6_contender")
                    .map_err(|e| e.to_string())?;
                (machine, 1_000_000_000_000u64, 2000, dir_info)
            }
            "bb6_contender2" => {
                let machine = SpaceTimeMachine::from_str(BB6_CONTENDER, up_x, up_y)?;
                let dir_info = create_sequential_subdir(r"m:\deldir\bb6_contender2")
                    .map_err(|e| e.to_string())?;
                (machine, 1_000_000_000u64, 1000, dir_info)
            }
            "bb5_1RB1RE_0RC1RA_1RD0LD_1LC1LB_0RA---" => {
                let machine =
                    SpaceTimeMachine::from_str("1RB1RE_0RC1RA_1RD0LD_1LC1LB_0RA---", up_x, up_y)?;
                let dir_info =
                    create_sequential_subdir(r"m:\deldir\bb5_1RB1RE_0RC1RA_1RD0LD_1LC1LB_0RA---")
                        .map_err(|e| e.to_string())?;
                (machine, 1_000_000_000u64, 1000, dir_info)
            }
            _ => Err(format!("Unknown machine: {}", machine_name))?,
        };

    println!(
        "Using machine: {} with output in {:?}",
        machine_name, &output_dir
    );
    println!("Using resolution: {:?} ({}x{})", resolution, goal_x, goal_y);

    let log_iter = LogStepIterator::new(end_step, num_frames);

    for (frame_index, goal_step_index) in log_iter.enumerate() {
        let actual_step_index = space_time_machine.step_count() - 1;
        if goal_step_index > actual_step_index
            && !space_time_machine.nth_js(goal_step_index - actual_step_index - 1)
        {
            break;
        }
        let actual_step_index = space_time_machine.step_count() - 1;
        println!("run_id: {}, Frame {}", run_id, frame_index);

        save_frame(
            &space_time_machine,
            &output_dir,
            run_id,
            frame_index as u32,
            actual_step_index + 1,
            goal_x,
            goal_y,
        )?;
    }

    Ok(())
}

fn save_frame(
    machine: &SpaceTimeMachine,
    output_dir: &Path,
    run_id: u32,
    frame: u32,
    step: u64,
    goal_x: u32,
    goal_y: u32,
) -> Result<(), String> {
    let file_name = output_dir.join(format!("run{run_id}_{frame:07}.png"));

    let font_data = include_bytes!(
        r"C:\Program Files\WindowsApps\Microsoft.WindowsTerminal_1.21.10351.0_x64__8wekyb3d8bbwe\CascadiaMono.ttf"
    );
    let font = FontArc::try_from_slice(font_data).map_err(|_| "Failed to load font")?;
    let scale = PxScale::from(50.0);

    let png_data = machine.png_data();
    let img = image::load_from_memory(&png_data).map_err(|e| e.to_string())?;
    let file_name2 = output_dir.join(format!("base{run_id}_{frame:07}.png"));
    img.save(&file_name2).map_err(|e| e.to_string())?;

    // Check if we are downscaling
    let mut resized = //if img.width() > goal_x && img.height() > goal_y {
        // Downscaling branch: Apply blur before resizing
        // let downscale_factor = (img.width().max(img.height()) as f32) / goal_x.max(goal_y) as f32;
        // let sigma = 1000.0; // downscale_factor * 0.5; // Adaptive blur based on downscaling ratio

        //let blurred = image::imageops::blur(&img, sigma);
        image::DynamicImage::ImageRgba8(image::imageops::blur(&img, 1.0))
            .resize_exact(goal_x, goal_y, FilterType::Lanczos3)
            .into_rgb16()
    // } else {
    //     // Upscaling or same-size branch: Use `Nearest` without blur
    //     img.resize_exact(goal_x, goal_y, FilterType::Nearest)
    //         .into_rgb16()
    // };
    ;

    // Calculate text position for lower right corner
    let text = format!("{:>75}", step.separate_with_commas());
    let text_height = 50.0; // Approximate height based on font size
    let y_position = goal_y as f32 - text_height - 10.0; // 10 pixels padding from bottom

    draw_text_mut(
        &mut resized,
        Rgb([110, 110, 110]),
        25,                // X position (keep same padding from edge)
        y_position as i32, // Y position now near bottom
        scale,
        &font,
        &text,
    );
    resized.save(&file_name).map_err(|e| e.to_string())?;

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
