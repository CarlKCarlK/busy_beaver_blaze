use core::error::Error;
use image::{DynamicImage, imageops::FilterType};
use std::{fs, path::Path, path::PathBuf};

fn main() -> Result<(), Box<dyn Error>> {
    //let input_pattern = r"M:\deldir\bb5_champ\42\base42_*.png";
    let input_pattern = r"M:\deldir\bb6_contender\36\base36_*.png";
    let dir_info = create_sequential_subdir(r"m:\deldir\bb6_contender_blur")?;

    let (goal_x, goal_y) = (1920, 1080);

    // print all files that match the pattern
    for entry in glob::glob(input_pattern)? {
        let path = entry?;
        let base = image::open(&path)?;
        let output_path = dir_info.0.join(path.file_name().unwrap());

        let x_fraction = base.width() as f32 / goal_x as f32;
        let resized = if x_fraction < 0.25 {
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
        resized.save(&output_path)?;
        println!("Resized and saved {output_path:?}");
    }

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
