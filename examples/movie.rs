use busy_beaver_blaze::{SpaceTimeMachine, BB6_CONTENDER};
use image::{ImageBuffer, Rgb};
use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
};

fn main() -> Result<(), String> {
    let goal_x: u32 = 1920;
    let goal_y: u32 = 1080;
    let mut space_time_machine = SpaceTimeMachine::from_str(BB6_CONTENDER, goal_x, goal_y)?;

    let num_frames = 1000;
    let start_step = 0f64;
    let end_step = 1e12f64; // 1 trillion

    // Calculate logarithmic step positions
    let log_start = start_step.max(1.0).ln();
    let log_end = end_step.ln();
    let log_step = (log_end - log_start) / (num_frames - 1) as f64;

    let output_dir = create_sequential_subdir(r"m:\deldir\bb6_contender").unwrap();
    let mut prev_steps = 0u64;

    // Capture initial frame
    save_frame(&space_time_machine, &output_dir, 0, goal_x, goal_y)?;

    // Capture remaining frames with logarithmic spacing
    for frame in 1..num_frames {
        let target_steps = (log_start + frame as f64 * log_step).exp().floor() as u64;
        let steps_to_take = target_steps - prev_steps;

        if !space_time_machine.nth_js(steps_to_take) {
            println!("Machine halted at step {}", prev_steps + steps_to_take);
            break;
        }

        println!(
            "Step: {}, Writing frame {:?}",
            space_time_machine.step_count().thousands_sep(),
            frame
        );

        save_frame(&space_time_machine, &output_dir, frame, goal_x, goal_y)?;
        prev_steps = target_steps;
    }

    Ok(())
}

fn save_frame(
    machine: &SpaceTimeMachine,
    output_dir: &Path,
    frame: usize,
    goal_x: u32,
    goal_y: u32,
) -> Result<(), String> {
    let file_name = output_dir.join(format!("{:07}.png", frame));

    let png_data = machine.png_data();
    let img = image::load_from_memory(&png_data).map_err(|e| e.to_string())?;
    let resized = img.resize_exact(goal_x, goal_y, image::imageops::FilterType::Nearest);
    resized.save(&file_name).map_err(|e| e.to_string())?;

    Ok(())
}

fn create_sequential_subdir(top_dir: &str) -> std::io::Result<PathBuf> {
    // Read all entries in the top directory
    let entries = fs::read_dir(top_dir)?;

    // Find the highest numbered subdirectory
    let mut max_num = 0;
    for entry in entries {
        if let Ok(entry) = entry {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Ok(num) = name.parse::<u32>() {
                        if num > max_num {
                            max_num = num;
                        }
                    }
                }
            }
        }
    }

    // Create a new subdirectory with the next sequential number
    let new_dir_num = max_num + 1;
    let new_dir_path = Path::new(top_dir).join(new_dir_num.to_string());

    fs::create_dir_all(&new_dir_path)?; // Handle error appropriately

    Ok(new_dir_path)
}
