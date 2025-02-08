use ab_glyph::{FontArc, PxScale, ScaleFont};
use busy_beaver_blaze::{SpaceTimeMachine, BB6_CONTENDER};
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
    let start_step = 0f64;
    let end_step = 1e12f64; // billion, trillion, etc

    // Calculate logarithmic step positions
    let log_start = start_step.max(1.0).ln();
    let log_end = end_step.ln();
    let log_step = (log_end - log_start) / (num_frames - 1) as f64;

    let (output_dir, run_id) =
        create_sequential_subdir(r"m:\deldir\bb6_contender").map_err(|e| e.to_string())?;
    let mut prev_steps = 0u64;

    // Capture initial frame
    save_frame(
        &space_time_machine,
        &output_dir,
        run_id,
        0,
        prev_steps + 1,
        goal_x,
        goal_y,
    )?;

    // Capture remaining frames with logarithmic spacing
    for frame in 1..num_frames {
        let target_steps = (log_start + frame as f64 * log_step).exp().floor() as u64;
        let steps_to_take = target_steps - prev_steps;

        if !space_time_machine.nth_js(steps_to_take) {
            println!("Machine halted at step {}", prev_steps + steps_to_take);
            break;
        }

        println!(
            "Run_id: {}, Step: {}, Writing frame {:?}",
            run_id,
            space_time_machine.step_count().separate_with_commas(),
            frame
        );

        save_frame(
            &space_time_machine,
            &output_dir,
            run_id,
            frame,
            prev_steps + 1,
            goal_x,
            goal_y,
        )?;
        prev_steps = target_steps;
    }

    Ok(())
}

fn save_frame(
    machine: &SpaceTimeMachine,
    output_dir: &Path,
    run_id: u32,
    frame: usize,
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
