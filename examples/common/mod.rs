use ab_glyph::FontArc;
use std::fs;
use std::path::{Path, PathBuf};

// Portable font for examples and dev tooling. Avoids platform-specific font loading.
// File path is relative to this source file.
const PORTABLE_FONT_BYTES: &[u8] = include_bytes!("../fonts/FiraCode-Regular.ttf");

/// Monospace font bundled with the repository for consistent rendering.
///
/// Errors if the embedded font cannot be parsed.
pub fn portable_font() -> Result<FontArc, &'static str> {
    FontArc::try_from_slice(PORTABLE_FONT_BYTES).map_err(|_| "Failed to load embedded font")
}

/// Creates a new numeric subdirectory under `top_dir` named with the next
/// sequential integer (starting from 1). Returns the new path and the number.
pub fn create_sequential_subdir(top_dir: impl AsRef<Path>) -> std::io::Result<(PathBuf, u32)> {
    let top_dir = top_dir.as_ref();
    fs::create_dir_all(top_dir)?;

    let mut max_num = 0u32;
    for entry in fs::read_dir(top_dir)? {
        if let Ok(entry) = entry {
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
    }

    let new_dir_num = max_num + 1;
    let new_dir_path = top_dir.join(new_dir_num.to_string());
    fs::create_dir_all(&new_dir_path)?;
    Ok((new_dir_path, new_dir_num))
}
