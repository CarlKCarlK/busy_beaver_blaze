use ab_glyph::{FontArc, PxScale};
use busy_beaver_blaze::{BB5_CHAMP, LogStepIterator, PixelPolicy, PngDataIterator};
use image::Rgba;
use image::{DynamicImage, imageops::FilterType};
use imageproc::drawing::draw_text_mut;
use itertools::Itertools;
use sanitize_filename::sanitize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use thousands::Separable;

pub const RESOLUTION_TINY: (u32, u32) = (320, 180); // Tiny (320x180)
pub const RESOLUTION_2K: (u32, u32) = (1920, 1080); // 2K (1920x1080, Full HD)
pub const RESOLUTION_4K: (u32, u32) = (3840, 2160); // 4K (3840x2160, Ultra HD)
pub const RESOLUTION_8K: (u32, u32) = (7680, 4320); // 8K (7680x4320, Ultra HD)
#[derive(Clone)]
struct Movie<'a> {
    title: &'a str,
    program: &'a str,
    pixel_policy: PixelPolicy,
    colors: Vec<[u8; 3]>,
    end_step: u64,
    frame_count: u32,
}

const DEFAULT_MOVIE: Movie = Movie {
    title: "DEFAULT MOVIE",
    program: BB5_CHAMP,
    pixel_policy: PixelPolicy::Binning,
    colors: Vec::new(),
    end_step: 50_000_000,
    frame_count: 500,
};

#[allow(clippy::shadow_unrelated, clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn core::error::Error>> {
    let start = std::time::Instant::now();
    let top_directory = PathBuf::from(r"m:\deldir\bb\movie_list\");
    let (goal_x, goal_y) = RESOLUTION_2K;

    let movie_list = vec![
        Movie {
            title: "Zig Zag",
            program: "1RB---0LC_2LC2RC1LB_0RA2RB0LB",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Mud Piles",
            program: "1RB0LB0RC_2LC2LA1RA_1RA1LC---",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Screen Door",
            program: "1RB2RB1LC_1LA2RB0RB_2LB---0LA",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Receding Hills To The Left",
            program: "1RB1LC---_0LC2RB1LB_2LA0RC1RC",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Receding Hills To The Right",
            program: "1RB2LA0LA_2LC---2RA_0RA2RC1LC",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Hill With Left Shading",
            program: "1RB2LA2RA_1LC1LB0RA_2RA0LB---",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Hill With Zig-Zag",
            program: "1RB2LB---_1RC2RB1LC_0LA0RB1LB",
            ..DEFAULT_MOVIE
        },
    ];

    println!("Using resolution: ({goal_x}x{goal_y})");
    let (run_dir, run_id) = create_sequential_subdir(&top_directory)?;

    for Movie {
        title,
        program,
        pixel_policy,
        colors,
        end_step,
        frame_count,
    } in movie_list
    {
        let sanitized_title = sanitize(title);
        let output_dir = run_dir.join(sanitized_title.as_str());

        println!(
            "Using machine: {} with output in {}",
            title,
            output_dir.display()
        );

        let part_count = 32;

        let log_iter = LogStepIterator::new(end_step, frame_count).collect_vec();
        let png_data_iterator = PngDataIterator::new(
            end_step,
            part_count,
            program,
            &colors,
            goal_x,
            goal_y,
            pixel_policy,
            &log_iter,
        );

        for (frame_index, (step_index, png_data_layers)) in png_data_iterator.enumerate() {
            println!(
                "run_id: {}, Frame {}, Step {}, time so far {:?}",
                run_id,
                frame_index,
                step_index + 1,
                start.elapsed()
            );

            save_frame(
                &png_data_layers,
                &output_dir,
                sanitized_title.as_str(),
                frame_index as u32,
                step_index + 1,
                goal_x,
                goal_y,
            )?;
        }
        println!(
            "Elapsed: {:?}, output_dir: {}",
            start.elapsed(),
            output_dir.display()
        );
    }

    Ok(())
}

fn save_frame(
    png_data: &[u8],
    output_dir: &Path,
    prefix: &str,
    frame: u32,
    step: u64,
    goal_x: u32,
    goal_y: u32,
) -> Result<(), Box<dyn core::error::Error>> {
    let base_file_name = output_dir.join(format!("base/{prefix}_{frame:07}.png"));
    let resized_file_name = output_dir.join(format!("resized/{prefix}_{frame:07}.png"));
    let metadata_file_name = output_dir.join(format!("metadata/{prefix}_{frame:07}.txt"));
    // Create the 3 subdirectories if they don't exist
    fs::create_dir_all(base_file_name.parent().unwrap())?;
    fs::create_dir_all(resized_file_name.parent().unwrap())?;
    fs::create_dir_all(metadata_file_name.parent().unwrap())?;

    #[cfg(target_os = "linux")]
    let font_data = include_bytes!(r"/mnt/c/Windows/Fonts/CascadiaMono.ttf");
    #[cfg(target_os = "windows")]
    let font_data = include_bytes!(r"C:\Windows\Fonts\CascadiaMono.ttf");

    let font = FontArc::try_from_slice(font_data).map_err(|_| "Failed to load font")?;

    // Compute a scale factor based on a base resolution of 1920x1080.
    // Here, we use the vertical dimension (1080) as the reference.
    let scale_factor = goal_y as f32 / 1080.0;
    let base_font_size = 50.0; // Font size that works for 1080p
    let scale = PxScale::from(base_font_size * scale_factor);

    // Use relative padding (using the same scale factor)
    let horizontal_padding = (25.0 * scale_factor).round() as u32;
    let vertical_padding = (10.0 * scale_factor).round() as u32;

    // Load the base image from memory and save it
    let base = image::load_from_memory(png_data)?;
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

    // Prepare the text. Here we assume `step.separate_with_commas()` returns a String.
    let text = step.separate_with_commas();
    // Save the metadata
    fs::write(&metadata_file_name, &text)?;

    // Calculate text dimensions using imageproc's text_size helper.
    let (text_width, text_height) = imageproc::drawing::text_size(scale, &font, &text);

    // Position the text in the bottom right corner.
    // x_position: from the right edge, back off horizontal_padding and the text width.
    // y_position: from the bottom edge, back off vertical_padding and the text height.
    let x_position = goal_x - horizontal_padding - text_width;
    let y_position = goal_y - vertical_padding - text_height - (text_height >> 1);

    // Draw the text onto the resized image
    draw_text_mut(
        &mut resized,
        Rgba([110, 110, 110, 255]), // Text color
        x_position as i32,
        y_position as i32,
        scale,
        &font,
        &text,
    );
    resized.save(&resized_file_name)?;

    Ok(())
}

// cmk appears elsewhere
fn create_sequential_subdir(top_dir: &Path) -> std::io::Result<(PathBuf, u32)> {
    // create top_dir if it doesn't exist
    fs::create_dir_all(top_dir)?;

    // Read all entries in the top directory
    let entries = fs::read_dir(top_dir)?;

    // Find the highest numbered subdirectory
    let mut max_num = 0;
    for entry in entries.flatten() {
        if entry.path().is_dir()
            && let Some(num) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u32>().ok())
        {
            max_num = max_num.max(num);
        }
    }

    // Create a new subdirectory with the next sequential number
    let new_dir_num = max_num + 1;
    let new_dir_path = top_dir.join(new_dir_num.to_string());

    fs::create_dir_all(&new_dir_path)?; // Handle error appropriately

    Ok((new_dir_path, new_dir_num))
}
