use ab_glyph::PxScale;
use busy_beaver_blaze::{BB5_CHAMP, BB6_CONTENDER, LogStepIterator, PixelPolicy, PngDataIterator};
use image::{DynamicImage, imageops::FilterType};
use image::{Rgba, RgbaImage};
use imageproc::drawing::draw_text_mut;
use itertools::Itertools;
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
    caption: fn(&str, &str) -> String,
    pixel_policy: PixelPolicy,
    colors: Vec<[u8; 3]>,
    early_stop: u64,
    frame_start: u32,
    frame_end: u32,
    part_count: usize,
}

const DEFAULT_MOVIE: Movie = Movie {
    title: "DEFAULT MOVIE",
    program: BB5_CHAMP,
    caption: |title, program| format!("{title} {program}"),
    pixel_policy: PixelPolicy::Binning,
    colors: Vec::new(),
    early_stop: 100_000_000,
    frame_start: 250,
    frame_end: 500,
    part_count: 32,
};

#[allow(unused)]
fn bb_2_5_list<'a>() -> Vec<Movie<'a>> {
    vec![
        Movie {
            title: "Neon Hill",
            program: "1RB1LA1RB2RB2LA_2LB3RB4RB---0LA",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Waterfall",
            program: "1RB4LA1LB2LA0RB_2LB3RB4LA---1RA",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Jagged Mountain",
            program: "1RB---4LB1RA4RA_2LB2LA3RA4LB0RB",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Hills of Color",
            program: "1RB3LA3LA2RA3RA_2LB2RA---4RB1LB",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Stairs without Railings",
            program: "1RB3RA3LB0LA3RB_2LA3RB4RA1LA---",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Infinite Hawaiian Ice",
            program: "1RB2LA1RA2LB---_0LA4RB3RB1LA2RA",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Stairs with Railing",
            program: "1RB3LA2LB0RA---_2LA4LA3RB0LB0RB",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Two Face",
            program: "1RB3LA0LB1RA0RB_2LB2LA1RB4RA---",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Lumpy",
            program: "1RB3LB4LA0LB---_2LA0LA1RB0RA3RA",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Monochrome Wings",
            program: "1RB3RB---3LA1RA_2LA3RA4LB0LB1LB",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Yellow Brick Road",
            program: "1RB3RB1LB2RA---_2LA2RB1LA4LB0RA",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Tents",
            program: "1RB2RB---0LB3LA_2LA2LB3RB4RB1LB",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Rotating Wheel",
            program: "1RB0LB2LA4LB3LA_2LA---3RA4RB2RB",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Closing Shells",
            program: "1RB3LA---4RB0LB_2LA3LB4LA1RB3RA",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Patched Tower",
            program: "1RB2LA3LA4RA0LA_1LA3RB1RB1LB---",
            ..DEFAULT_MOVIE
        },
        Movie {
            title: "Fish Scales",
            program: "1RB2LA1LA4RA2LA_0LA3RB3LB2RB---",
            ..DEFAULT_MOVIE
        },
    ]
}

#[allow(clippy::shadow_unrelated, clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn core::error::Error>> {
    let start = std::time::Instant::now();

    let top_directory = PathBuf::from(r"m:\deldir\bb\movie_list\bb6_contender");
    let (goal_x, goal_y) = RESOLUTION_4K;
    let movie_list = vec![Movie {
        title: "",
        colors: Vec::new(),
        program: BB6_CONTENDER,
        caption: |_title, _program| String::new(),
        pixel_policy: PixelPolicy::Binning,
        //early_stop: 1_000_000_000_000,
        early_stop: 10_000_000_000,
        frame_start: 0,
        frame_end: 1000,
        part_count: 1,
    }];

    // let top_directory = PathBuf::from(r"m:\deldir\bb\movie_list\bb_2_5");
    // let (goal_x, goal_y) = RESOLUTION_2K;
    // let movie_list = bb_2_5_list();

    // let top_directory = PathBuf::from(r"m:\deldir\bb\movie_list\bb5_champ");
    // let (goal_x, goal_y) = RESOLUTION_2K;
    // let movie_list = vec![Movie {
    //     title: "",
    //     colors: Vec::new(),
    //     program: BB5_CHAMP,
    //     caption: |_title, _program| String::new(),
    //     pixel_policy: PixelPolicy::Binning,
    //     early_stop: 47_176_870,
    //     frame_start: 0,
    //     frame_end: 1000,
    // }];

    println!("Using resolution: ({goal_x}x{goal_y})");
    let (output_dir, run_id) = create_sequential_subdir(&top_directory)?;
    let _ = fs::create_dir_all(&output_dir);
    let file_prefix = run_id.to_string();

    let mut last_frame: Option<DynamicImage> = None;

    let mut frame_index_range = 0u32..;
    for Movie {
        title,
        program,
        caption,
        pixel_policy,
        colors,
        early_stop,
        frame_start,
        frame_end,
        part_count,
    } in movie_list
    {
        println!(
            "Using machine: {} with output in {}",
            title,
            &output_dir.display()
        );

        let log_iter = LogStepIterator::new(early_stop, frame_end).collect_vec();
        let png_data_iterator = PngDataIterator::new(
            early_stop,
            part_count,
            program,
            &colors,
            goal_x,
            goal_y,
            pixel_policy,
            &log_iter,
        );

        for (inner_frame_index, (step_index, png_data_layers)) in
            png_data_iterator.skip(frame_start as usize).enumerate()
        {
            println!(
                "run_id: {}, Frame {}, Step {}, time so far {:?}",
                run_id,
                inner_frame_index,
                step_index + 1,
                start.elapsed()
            );

            let resized = create_frame(
                &png_data_layers,
                caption(title, program).as_str(),
                step_index + 1,
                goal_x,
                goal_y,
            )?;

            if let Some(last_frame) = last_frame
                && inner_frame_index == 0
            {
                const TRANSITION_DURATION: usize = 10;
                // create TRANSITION_DURATION png frames that interpolate between last_frame and resized
                for i in 0..TRANSITION_DURATION {
                    let fraction = (i + 1) as f32 / (TRANSITION_DURATION + 1) as f32;
                    let interpolated = blend_images(&last_frame, &resized, fraction);
                    let frame_index = frame_index_range.next().unwrap();
                    let interpolated_file_name =
                        output_dir.join(format!("{file_prefix}_{frame_index:07}.png").as_str());
                    interpolated.save(&interpolated_file_name)?;
                }
            }

            let frame_index = frame_index_range.next().unwrap();
            let resized_file_name =
                output_dir.join(format!("{file_prefix}_{frame_index:07}.png").as_str());
            resized.save(&resized_file_name)?;

            last_frame = Some(resized);
        }
        println!(
            "Elapsed: {:?}, output_dir: {}",
            start.elapsed(),
            output_dir.display()
        );
    }

    Ok(())
}

fn create_frame(
    png_data: &[u8],
    caption: &str,
    step: u64,
    goal_x: u32,
    goal_y: u32,
) -> Result<DynamicImage, Box<dyn core::error::Error>> {
    let font = busy_beaver_blaze::test_utils::get_portable_font()?;

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
    let text = format!("{} {}", step.separate_with_commas(), caption);

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

    Ok(resized)
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

#[must_use]
fn blend_images(img1: &DynamicImage, img2: &DynamicImage, fraction: f32) -> DynamicImage {
    let img1 = img1.to_rgba8();
    let img2 = img2.to_rgba8();
    assert_eq!(
        img1.dimensions(),
        img2.dimensions(),
        "Images must be the same size"
    );

    let (width, height) = img1.dimensions();
    let mut blended = RgbaImage::new(width, height);

    for (x, y, pixel) in blended.enumerate_pixels_mut() {
        let p1 = img1.get_pixel(x, y).0;
        let p2 = img2.get_pixel(x, y).0;
        let blended_pixel = [
            (1.0 - fraction).mul_add(p1[0] as f32, fraction * p2[0] as f32) as u8,
            (1.0 - fraction).mul_add(p1[1] as f32, fraction * p2[1] as f32) as u8,
            (1.0 - fraction).mul_add(p1[2] as f32, fraction * p2[2] as f32) as u8,
            (1.0 - fraction).mul_add(p1[3] as f32, fraction * p2[3] as f32) as u8,
        ];
        *pixel = image::Rgba(blended_pixel);
    }

    DynamicImage::ImageRgba8(blended)
}
