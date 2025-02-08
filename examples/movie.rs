use ab_glyph::{FontArc, PxScale};
use busy_beaver_blaze::{LogStepIterator, SpaceTimeMachine, BB6_CONTENDER};
use image::{imageops::FilterType, Rgb};
use imageproc::drawing::draw_text_mut;
use std::{
    fs,
    path::{Path, PathBuf},
};
use thousands::Separable;

fn main() -> Result<(), String> {
    let goal_x: u32 = 1920;
    let goal_y: u32 = 1080;
    let mut space_time_machine = SpaceTimeMachine::from_str(BB6_CONTENDER, goal_x, goal_y)?;

    let num_frames = 2000;
    let end_step = 1_000_000_000_000u64; // billion, trillion, etc

    let log_iter = LogStepIterator::new(end_step, num_frames);

    let (output_dir, run_id) =
        create_sequential_subdir(r"m:\deldir\bb6_contender").map_err(|e| e.to_string())?;

    for (frame_index, goal_step_index) in log_iter.enumerate() {
        let actual_step_index = space_time_machine.step_count() - 1;
        if goal_step_index > actual_step_index
            && !space_time_machine.nth_js(goal_step_index - actual_step_index - 1)
        {
            break;
        }
        let actual_step_index = space_time_machine.step_count() - 1;
        println!(
            "run_id: {}, Frame {}: goal {}, actual {}",
            run_id, frame_index, goal_step_index, actual_step_index
        );

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
    let mut resized = img
        .resize_exact(goal_x, goal_y, FilterType::Nearest)
        .into_rgb8();

    // Calculate text position for lower right corner
    let text = format!("{:>75}", step.separate_with_commas());
    let text_height = 50.0; // Approximate height based on font size
    let y_position = goal_y as f32 - text_height - 10.0; // 10 pixels padding from bottom

    draw_text_mut(
        &mut resized,
        Rgb([211, 211, 211]),
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
