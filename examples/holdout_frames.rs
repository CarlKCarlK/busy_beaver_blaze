//! Cache three space-time frames per machine for the quick holdout montage.
use busy_beaver_blaze::SpaceByTimeMachine;
use std::{fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    assert_eq!(
        arguments.len(),
        3,
        "usage: holdout_frames INPUT.txt CACHE_ROOT"
    );
    let programs = fs::read_to_string(&arguments[1])?;
    let root = PathBuf::from(&arguments[2]);
    for (index, program) in programs.lines().enumerate() {
        let folder = root
            .join(program)
            .join("1920x984/steps10000_frames3_binning_v1");
        let targets = [3333_u64, 6666, 10000];
        if targets
            .iter()
            .all(|step| folder.join(format!("step_{step:05}.png")).exists())
        {
            continue;
        }
        fs::create_dir_all(&folder)?;
        let mut machine = SpaceByTimeMachine::from_str(program, 1920, 984, true, 0)?;
        for target in targets {
            while machine.step_count() < target {
                assert!(
                    machine.next().is_some(),
                    "machine {} halted early",
                    index + 1
                );
            }
            let path = folder.join(format!("step_{target:05}.png"));
            let temporary = path.with_extension("tmp.png");
            fs::write(&temporary, machine.to_png(&[12, 17, 24, 77, 218, 210])?)?;
            fs::rename(temporary, path)?;
        }
        if index % 25 == 0 || index == 607 {
            eprintln!("Rendered {}/608", index + 1);
        }
    }
    Ok(())
}
