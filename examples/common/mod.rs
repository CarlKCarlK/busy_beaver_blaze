use ab_glyph::FontArc;

// Portable font for examples and dev tooling. Avoids platform-specific font loading.
// File path is relative to this source file.
const PORTABLE_FONT_BYTES: &[u8] = include_bytes!("../fonts/FiraCode-Regular.ttf");

/// Monospace font bundled with the repository for consistent rendering.
///
/// Errors if the embedded font cannot be parsed.
pub fn portable_font() -> Result<FontArc, &'static str> {
    FontArc::try_from_slice(PORTABLE_FONT_BYTES).map_err(|_| "Failed to load embedded font")
}

