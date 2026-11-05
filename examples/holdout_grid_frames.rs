//! Cache 1,800 genuinely distinct simulation snapshots per machine at tile size.
use busy_beaver_blaze::SpaceByTimeMachine;
use image::{GrayImage, imageops::FilterType};
use rayon::prelude::*;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    assert!(
        matches!(arguments.len(), 3 | 4 | 6),
        "usage: holdout_grid_frames INPUT.txt CACHE_ROOT [MAX_STEPS [WIDTH HEIGHT]]"
    );
    let max_steps: u64 = arguments.get(3).map_or(Ok(10000), |value| value.parse())?;
    let tile_width: u32 = arguments.get(4).map_or(Ok(59), |value| value.parse())?;
    let tile_height: u32 = arguments.get(5).map_or(Ok(46), |value| value.parse())?;
    assert!((2..=4096).contains(&tile_width) && (2..=4096).contains(&tile_height));
    assert!(
        max_steps >= 1800,
        "need at least 1800 steps for distinct targets"
    );
    let programs = fs::read_to_string(&arguments[1])?;
    let programs: Vec<&str> = programs.lines().collect();
    assert_eq!(programs.len(), 608);
    let root = PathBuf::from(&arguments[2]);
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()?;
    let completed = AtomicUsize::new(0);
    programs
        .par_iter()
        .try_for_each(|program| -> Result<(), String> {
            let folder = root.join(program).join(format!(
                "{tile_width}x{tile_height}/steps{max_steps}_frames1800_linear_binning_gray_v1"
            ));
            let path = folder.join("frames.gray");
            let expected = 1800 * u64::from(tile_width) * u64::from(tile_height);
            if fs::metadata(&path).is_ok_and(|metadata| metadata.len() == expected) {
                completed.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
            fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
            let mut machine =
                SpaceByTimeMachine::from_str(program, tile_width, tile_height, true, 0)?;
            let mut frames = Vec::with_capacity(expected as usize);
            for frame_index in 0..1800_u64 {
                let target = 1 + frame_index * (max_steps - 1) / 1799;
                while machine.step_count() < target {
                    assert!(
                        machine.next().is_some(),
                        "machine halted before step {target}: {program}"
                    );
                }
                let (_, width, height, layers) =
                    machine.png_data_and_packed_data(&[[0, 0, 0], [255, 255, 255]]);
                let image = GrayImage::from_raw(width, height, layers[0].to_vec())
                    .expect("valid packed grayscale");
                let tile =
                    image::imageops::resize(&image, tile_width, tile_height, FilterType::Nearest);
                frames.extend_from_slice(tile.as_raw());
            }
            assert_eq!(frames.len(), expected as usize);
            let temporary = folder.join("frames.tmp.gray");
            fs::write(&temporary, frames).map_err(|error| error.to_string())?;
            fs::rename(temporary, path).map_err(|error| error.to_string())?;
            let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if count % 25 == 0 || count == 608 {
                eprintln!("Animated tile caches: {count}/608");
            }
            Ok(())
        })?;
    Ok(())
}
