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

    let n = 10_000_000;

    let top_dir = r"m:\deldir\bb6_contender";
    let output_dir = create_sequential_subdir(top_dir).unwrap();

    let mut image_number = 0;

    while space_time_machine.nth_js(n) {
        let file_name = output_dir.join(format!("{:07}.png", image_number));
        println!("Writing {:?}", file_name);

        // Get PNG data and convert to image
        let png_data = space_time_machine.png_data();
        let img = image::load_from_memory(&png_data).map_err(|e| e.to_string())?;

        // Resize to goal dimensions
        let resized = img.resize_exact(goal_x, goal_y, image::imageops::FilterType::Nearest);

        // Save the resized image
        resized.save(&file_name).map_err(|e| e.to_string())?;

        image_number += 1;
    }

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
