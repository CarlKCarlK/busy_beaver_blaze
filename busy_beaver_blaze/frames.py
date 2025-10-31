"""Frame generation utilities for Busy Beaver visualizations.

This module provides Python-side utilities for generating and processing
Turing machine visualization frames. It complements the Rust PngDataIterator
with image post-processing capabilities.
"""

import math
from io import BytesIO
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    from matplotlib.font_manager import FontProperties, findfont
except ImportError as e:
    raise ImportError(
        "PIL (Pillow) and matplotlib are required for frame processing. "
        "Install with: pip install pillow matplotlib"
    ) from e

# Resolution constants matching Rust examples/movie_list.rs
RESOLUTION_TINY = (320, 180)    # Tiny (320x180)
RESOLUTION_2K = (1920, 1080)    # 2K (1920x1080, Full HD)
RESOLUTION_4K = (3840, 2160)    # 4K (3840x2160, Ultra HD)
RESOLUTION_8K = (7680, 4320)    # 8K (7680x4320, Ultra HD)


def log_step_iterator(max_steps: int, frame_count: int) -> list[int]:
    """Generate logarithmically-spaced step indices.
    
    This is a Python reimplementation of Rust's LogStepIterator.
    Returns `frame_count` steps between 0 and `max_steps-1`, spaced
    logarithmically for better visualization of exponential growth.
    
    Args:
        max_steps: Maximum step count (exclusive upper bound)
        frame_count: Number of frames to generate
        
    Returns:
        List of step indices, logarithmically spaced
        
    Example:
        >>> steps = log_step_iterator(1000000, 100)
        >>> len(steps)
        100
        >>> steps[0]
        0
        >>> steps[-1]
        999999
    """
    if frame_count == 0:
        return []
    if frame_count == 1:
        return [max_steps - 1]
    
    result = []
    for current_frame in range(frame_count):
        t = current_frame / (frame_count - 1)
        
        if t == 1.0:
            value = max_steps - 1
        else:
            # f(t) = exp(ln(max_steps) * t) - 1
            log_value = math.expm1(math.log(max_steps) * t)
            # Use floor so lower integer is used until f(t) reaches next integer
            value = min(int(math.floor(log_value)), max_steps - 1)
        
        result.append(value)
    
    return result


def resize_png(
    png_bytes: bytes,
    target_resolution: tuple[int, int],
) -> bytes:
    """Resize PNG to exact target dimensions with appropriate filtering.
    
    The Rust engine can produce PNGs up to 2× larger than requested for efficiency
    (adaptive sampling). This function scales them down to the exact target size,
    using appropriate resampling based on the scale factor.
    
    This logic is shared by create_frame() and the interactive visualizer.
    
    Args:
        png_bytes: Raw PNG image data
        target_resolution: Target (width, height) in pixels
        
    Returns:
        Resized PNG as bytes
        
    Example:
        >>> png_bytes = machine.to_png()
        >>> resized = resize_png(png_bytes, (1920, 1080))
    """
    width, height = target_resolution
    
    # Load image from memory
    base = Image.open(BytesIO(png_bytes))
    
    # If already exact size, return as-is (optimization)
    if base.width == width and base.height == height:
        return png_bytes
    
    # Resize with anti-aliasing (same logic as create_frame)
    x_fraction = base.width / width
    if x_fraction < 0.25:
        # Very small source - use nearest neighbor
        resized = base.resize((width, height), Image.NEAREST)
    else:
        # Blur and resize with lanczos for quality
        blurred = base.filter(ImageFilter.GaussianBlur(radius=1.0))
        if x_fraction < 1.0:
            resized = blurred.resize((width, height), Image.LANCZOS)
        else:
            resized = blurred.resize((width, height), Image.NEAREST)
    
    # Save back to bytes
    output = BytesIO()
    resized.save(output, format='PNG')
    return output.getvalue()


def create_frame(
    png_bytes: bytes,
    caption: str,
    step_index: int,
    resolution: tuple[int, int],
    font_size: Optional[int] = None,
) -> Image.Image:
    """Create a visualization frame from PNG bytes with text overlay.
    
    This function replicates the behavior of Rust's create_frame() from
    examples/movie_list.rs. It loads the PNG, resizes it with anti-aliasing,
    and adds a text caption in the bottom-right corner.
    
    Args:
        png_bytes: Raw PNG image data from PngDataIterator
        caption: Text to display (appended to step count)
        step_index: Current step number (0-based)
        resolution: Target (width, height) in pixels
        font_size: Font size override (auto-scaled by default based on height)
        
    Returns:
        PIL Image ready to save or further process
        
    Example:
        >>> from busy_beaver_blaze import PngDataIterator, BB5_CHAMP, RESOLUTION_2K
        >>> iterator = PngDataIterator([0, 1000, 10000], program=BB5_CHAMP)
        >>> step_idx, png_data = next(iterator)
        >>> img = create_frame(png_data, "BB5 Champion", step_idx, RESOLUTION_2K)
        >>> img.save("frame_0000.png")
    """
    width, height = resolution
    
    # Resize to exact target resolution
    png_bytes = resize_png(png_bytes, resolution)
    
    # Load resized image
    base = Image.open(BytesIO(png_bytes))
    
    # Compute scale factor based on 1920x1080 reference (vertical dimension)
    scale_factor = height / 1080.0
    
    # Convert to RGBA for text drawing
    if base.mode != 'RGBA':
        base = base.convert('RGBA')
    
    # Prepare text with thousand separators
    step_display = f"{step_index + 1:,}"  # +1 to match Rust (step_index + 1)
    text = f"{step_display} {caption}".strip()
    
    # Load FiraCode font (same as Rust) or fallback to monospace
    if font_size is None:
        base_font_size = 50.0  # Font size for 1080p (matches Rust)
        font_size = int(base_font_size * scale_factor)
    
    # Try to load FiraCode-Regular.ttf from examples/fonts (matching Rust)
    font = None
    from pathlib import Path
    # Navigate from busy_beaver_blaze package to repository root
    repo_root = Path(__file__).parent.parent
    fira_code_path = repo_root / "examples" / "fonts" / "FiraCode-Regular.ttf"
    
    if fira_code_path.exists():
        try:
            font = ImageFont.truetype(str(fira_code_path), font_size)
        except Exception:
            pass
    
    if font is None:
        try:
            # Try to find monospace font on system
            font_path = findfont(FontProperties(family='monospace'))
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            # Last resort
            font = ImageFont.load_default()
    
    # Calculate text dimensions
    draw = ImageDraw.Draw(base)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    
    # Use a reference string without descenders for consistent height positioning
    # This prevents the text from jumping when commas (with descenders) appear
    ref_bbox = draw.textbbox((0, 0), "0123456789", font=font)
    text_height = ref_bbox[3] - ref_bbox[1]
    
    # Position in bottom-right with padding (matching Rust behavior)
    horizontal_padding = int(25.0 * scale_factor)
    vertical_padding = int(10.0 * scale_factor)
    
    x_position = width - horizontal_padding - text_width
    # Add extra vertical offset to account for descenders and ensure bottom margin
    y_position = height - vertical_padding - text_height - int(10.0 * scale_factor)
    
    # Draw text (matching Rust: gray color, no stroke)
    draw.text(
        (x_position, y_position),
        text,
        font=font,
        fill=(110, 110, 110, 255)
    )
    
    return base


def blend_images(img1: Image.Image, img2: Image.Image, fraction: float) -> Image.Image:
    """Blend two images with linear interpolation.
    
    Creates smooth transitions between frames by blending pixels.
    This replicates Rust's blend_images() from examples/movie_list.rs.
    
    Args:
        img1: First image
        img2: Second image
        fraction: Blend fraction (0.0 = all img1, 1.0 = all img2)
        
    Returns:
        Blended image
        
    Raises:
        ValueError: If images have different dimensions
        
    Example:
        >>> # Create 10-frame transition
        >>> for i in range(10):
        ...     frac = (i + 1) / 11.0
        ...     blended = blend_images(frame1, frame2, frac)
        ...     blended.save(f"transition_{i:03d}.png")
    """
    if img1.size != img2.size:
        raise ValueError(
            f"Images must be same size, got {img1.size} and {img2.size}"
        )
    
    # Convert to RGBA
    img1_rgba = img1.convert('RGBA')
    img2_rgba = img2.convert('RGBA')
    
    # Use PIL's blend for efficiency
    return Image.blend(img1_rgba, img2_rgba, fraction)
