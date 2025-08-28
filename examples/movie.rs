use ab_glyph::{FontArc, PxScale};
use busy_beaver_blaze::{
    BB_2_5_CHAMP_AUG25, BB_3_3_355317, BB5_CHAMP, BB6_CONTENDER, BIGFOOT33, BIGFOOT72, BRADY,
    LogStepIterator, Machine, PixelPolicy, PngDataIterator,
};
use core::str::FromStr;
use image::Rgba;
use image::{DynamicImage, imageops::FilterType};
use imageproc::drawing::draw_text_mut;
use itertools::Itertools;
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
        .map(|str| Resolution::from_str(&str))
        .transpose()?
        .unwrap_or(Resolution::Tiny); // cmk2K
    let (goal_x, goal_y) = resolution.dimensions();

    let binning = std::env::args()
        .nth(3)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(true);

    // Optional 4th argument: comma-separated list of colors (CSS names or #hex)
    let colors_arg = std::env::args().nth(4);

    // let (up_x, up_y) = (goal_x, goal_y);
    let (program_string, end_step, num_frames, (output_dir, run_id)) = match machine_name.as_str() {
        "bb5_champ" => {
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\bb5_champ")?;
            (BB5_CHAMP, 47_176_870, 1000, dir_info)
        }
        "bb5_champ2" => {
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\bb5_champ2")?;
            (BB5_CHAMP, 100_000_000, 250, dir_info)
        }
        "bb6_contender" => {
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\bb6_contender")?;
            (BB6_CONTENDER, 1_000_000_000_000u64, 2000, dir_info)
        }
        "bb6_contender10T" => {
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\bb6_contender")?;
            (BB6_CONTENDER, 10_000_000_000_000u64, 4000, dir_info)
        }
        "bb6_contender100T" => {
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\bb6_contender")?;
            (BB6_CONTENDER, 100_000_000_000_000u64, 4000, dir_info)
        }
        "bb6_contender2" => {
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\bb6_contender2")?;
            (BB6_CONTENDER, 1_000_000_000u64, 1000, dir_info)
        }
        "bb5_1RB1RE_0RC1RA_1RD0LD_1LC1LB_0RA---" => {
            let dir_info =
                create_sequential_subdir(r"m:\deldir\bb\bb5_1RB1RE_0RC1RA_1RD0LD_1LC1LB_0RA---")?;
            (
                "1RB1RE_0RC1RA_1RD0LD_1LC1LB_0RA---",
                1_000_000_000u64,
                1000,
                dir_info,
            )
        }
        "BB_3_3_355317" => {
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\BB_3_3_355317")?;
            (BB_3_3_355317, 1_000_000_000u64, 1000, dir_info)
        }
        "Bigfoot33" => {
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\Bigfoot33")?;
            (BIGFOOT33, 1_000_000_000u64, 1000, dir_info)
        }
        "Bigfoot72" => {
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\Bigfoot72")?;
            (BIGFOOT72, 1_000_000_000u64, 1000, dir_info)
        }
        "Brady" => {
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\Brady")?;
            (BRADY, 1_000_000_000u64, 1000, dir_info)
        }

        "BB_2_5_CHAMP_AUG25" => {
            let dir_info = create_sequential_subdir(r"m:\deldir\bb\BB_2_5_CHAMP_AUG25")?;
            (BB_2_5_CHAMP_AUG25, 1_000_000_000u64, 1000, dir_info)
        }
        _ => Err(format!("Unknown machine: {machine_name}"))?,
    };

    // cmk0 review this code
    // Determine symbol count by parsing the machine once (cheap) to validate colors length
    let machine_tmp = Machine::from_string(program_string)?;
    let symbol_count = machine_tmp.symbol_count() as usize;

    let colors_vec: Vec<[u8; 3]> = if let Some(colors_csv) = colors_arg {
        let parsed_colors_res: Result<Vec<[u8; 3]>, String> = colors_csv
            .split(',')
            .map(str::trim)
            .filter(|segment| !segment.is_empty())
            .map(|name| {
                csscolorparser::parse(name)
                    .map(|color| {
                        let rgba = color.to_rgba8();
                        [rgba[0], rgba[1], rgba[2]]
                    })
                    .map_err(|err| format!("Invalid color '{name}': {err}"))
            })
            .collect();
        let parsed_colors = parsed_colors_res.map_err(|err| format!("Color parse error: {err}"))?;
        if parsed_colors.len() != symbol_count {
            return Err(format!(
                "Number of colors ({}) must equal number of symbols ({symbol_count})",
                parsed_colors.len()
            )
            .into());
        }
        parsed_colors
    } else {
        // cmk000 shouldn't reuse the 0th color. See lib.rs's list of default colors
        // Default palette starting with white, yellow, orange; repeat to fill
        let default_palette: &[[u8; 3]] = &[
            [255, 255, 255], // white
            [255, 165, 0],   // orange
            [255, 255, 0],   // yellow
            [255, 0, 255],   // magenta
            [0, 255, 255],   // cyan
            [0, 128, 0],     // green
            [0, 0, 255],     // blue
            [75, 0, 130],    // indigo
            [238, 130, 238], // violet
            [255, 0, 0],     // red
            [0, 0, 0],       // black
        ];
        default_palette
            .iter()
            .copied()
            .cycle()
            .take(symbol_count)
            .collect()
    };

    let colors: Vec<[u8; 3]> = colors_vec;
    println!(
        "Using machine: {} with output in {}",
        machine_name,
        &output_dir.display()
    );
    println!("Using resolution: {resolution:?} ({goal_x}x{goal_y})");

    let part_count = 32;

    let log_iter = LogStepIterator::new(end_step, num_frames).collect_vec();
    let png_data_iterator = PngDataIterator::new(
        end_step,
        part_count,
        program_string,
        &colors,
        goal_x,
        goal_y,
        PixelPolicy::from(binning),
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
            run_id,
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
    Ok(())
}

fn save_frame(
    png_data: &[u8],
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

fn create_sequential_subdir(top_dir: &str) -> std::io::Result<(PathBuf, u32)> {
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
    let new_dir_path = Path::new(top_dir).join(new_dir_num.to_string());

    fs::create_dir_all(&new_dir_path)?; // Handle error appropriately

    Ok((new_dir_path, new_dir_num))
}
