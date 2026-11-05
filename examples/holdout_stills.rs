//! Cache high-resolution final diagrams for the holdout poster.
use busy_beaver_blaze::SpaceByTimeMachine;
use rayon::prelude::*;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    assert_eq!(
        arguments.len(),
        6,
        "usage: holdout_stills INPUT.txt CACHE_ROOT STEPS WIDTH HEIGHT"
    );
    let programs = fs::read_to_string(&arguments[1])?;
    let root = PathBuf::from(&arguments[2]);
    let steps: u64 = arguments[3].parse()?;
    let width: u32 = arguments[4].parse()?;
    let height: u32 = arguments[5].parse()?;
    assert!(steps > 0 && width >= 2 && height >= 2);
    let programs: Vec<&str> = programs.lines().collect();
    assert_eq!(programs.len(), 608);
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()?;
    let completed = AtomicUsize::new(0);
    programs
        .par_iter()
        .try_for_each(|program| -> Result<(), String> {
            let folder = root
                .join(program)
                .join(format!("{width}x{height}/steps{steps}_binning_teal_v1"));
            let path = folder.join(format!("step_{steps}.png"));
            if !path.exists() {
                fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
                let mut machine = SpaceByTimeMachine::from_str(program, width, height, true, 0)?;
                while machine.step_count() < steps {
                    assert!(machine.next().is_some(), "machine halted early: {program}");
                }
                let temporary = path.with_extension("tmp.png");
                fs::write(&temporary, machine.to_png(&[12, 17, 24, 77, 218, 210])?)
                    .map_err(|error| error.to_string())?;
                fs::rename(temporary, path).map_err(|error| error.to_string())?;
            }
            let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if count % 50 == 0 || count == 608 {
                eprintln!("Poster renders: {count}/608");
            }
            Ok(())
        })?;
    Ok(())
}
