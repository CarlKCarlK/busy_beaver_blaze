"""Tests for PyO3 PngDataIterator bindings."""
# cmk run these tests

import pytest

# Skip all tests if Rust bindings not available
pytest.importorskip("busy_beaver_blaze._busy_beaver_blaze")

from busy_beaver_blaze import (
    PngDataIterator,
    BB5_CHAMP,
    BB6_CONTENDER,
    log_step_iterator,
)


class TestLogStepIterator:
    """Test the pure Python log_step_iterator function."""
    
    def test_single_frame(self):
        """Test with a single frame."""
        steps = log_step_iterator(1000, 1)
        assert len(steps) == 1
        assert steps[0] == 999
    
    def test_multiple_frames(self):
        """Test with multiple frames."""
        steps = log_step_iterator(1000, 10)
        assert len(steps) == 10
        assert steps[0] == 0
        assert steps[-1] == 999
        # Should be monotonically increasing
        assert all(steps[i] <= steps[i+1] for i in range(len(steps)-1))
    
    def test_empty(self):
        """Test with zero frames."""
        steps = log_step_iterator(1000, 0)
        assert len(steps) == 0


class TestPngDataIterator:
    """Test the Rust PngDataIterator Python bindings."""
    
    def test_create_iterator(self):
        """Test that we can create an iterator."""
        frame_steps = [0, 10, 100]
        iterator = PngDataIterator(
            early_stop=1000,
            program=BB5_CHAMP,
            width=320,
            height=180,
            pixel_policy="binning",
            frame_steps=frame_steps,
            colors=[],
            part_count=1,
        )
        assert iterator is not None
    
    def test_iterator_protocol(self):
        """Test that iterator yields step_index and png_bytes."""
        frame_steps = [0, 10]
        iterator = PngDataIterator(
            early_stop=100,
            program=BB5_CHAMP,
            width=320,
            height=180,
            pixel_policy="binning",
            frame_steps=frame_steps,
            colors=[],
            part_count=1,
        )
        
        results = list(iterator)
        assert len(results) == 2
        
        # Check first frame
        step_index, png_bytes = results[0]
        assert step_index == 0
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        # PNG files start with magic bytes
        assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n'
    
    def test_pixel_policy_binning(self):
        """Test with binning pixel policy."""
        iterator = PngDataIterator(
            early_stop=100,
            program=BB5_CHAMP,
            width=320,
            height=180,
            pixel_policy="binning",
            frame_steps=[0],
            colors=[],
            part_count=1,
        )
        results = list(iterator)
        assert len(results) == 1
    
    def test_pixel_policy_sampling(self):
        """Test with sampling pixel policy."""
        iterator = PngDataIterator(
            early_stop=100,
            program=BB5_CHAMP,
            width=320,
            height=180,
            pixel_policy="sampling",
            frame_steps=[0],
            colors=[],
            part_count=1,
        )
        results = list(iterator)
        assert len(results) == 1
    
    def test_invalid_pixel_policy(self):
        """Test that invalid pixel policy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid pixel_policy"):
            PngDataIterator(
                early_stop=100,
                program=BB5_CHAMP,
                width=320,
                height=180,
                pixel_policy="invalid",
                frame_steps=[0],
                colors=[],
                part_count=1,
            )
    
    def test_hex_colors_rrggbb(self):
        """Test hex color parsing with #RRGGBB format."""
        iterator = PngDataIterator(
            early_stop=100,
            program=BB5_CHAMP,
            width=320,
            height=180,
            pixel_policy="binning",
            frame_steps=[0],
            colors=["#FF0000", "#00FF00", "#0000FF"],
            part_count=1,
        )
        results = list(iterator)
        assert len(results) == 1
    
    def test_hex_colors_rgb(self):
        """Test hex color parsing with #RGB format."""
        iterator = PngDataIterator(
            early_stop=100,
            program=BB5_CHAMP,
            width=320,
            height=180,
            pixel_policy="binning",
            frame_steps=[0],
            colors=["#F00", "#0F0", "#00F"],
            part_count=1,
        )
        results = list(iterator)
        assert len(results) == 1
    
    def test_hex_colors_no_hash(self):
        """Test hex color parsing without # prefix."""
        iterator = PngDataIterator(
            early_stop=100,
            program=BB5_CHAMP,
            width=320,
            height=180,
            pixel_policy="binning",
            frame_steps=[0],
            colors=["FF0000", "00FF00"],
            part_count=1,
        )
        results = list(iterator)
        assert len(results) == 1
    
    def test_invalid_hex_color(self):
        """Test that invalid hex color raises ValueError."""
        with pytest.raises(ValueError, match="Invalid .* component in hex color"):
            PngDataIterator(
                early_stop=100,
                program=BB5_CHAMP,
                width=320,
                height=180,
                pixel_policy="binning",
                frame_steps=[0],
                colors=["#GGGGGG"],  # Invalid hex
                part_count=1,
            )
    
    def test_empty_colors(self):
        """Test that empty color list uses defaults."""
        iterator = PngDataIterator(
            early_stop=100,
            program=BB5_CHAMP,
            width=320,
            height=180,
            pixel_policy="binning",
            frame_steps=[0],
            colors=[],
            part_count=1,
        )
        results = list(iterator)
        assert len(results) == 1
    
    def test_bb6_contender(self):
        """Test with BB6 contender program."""
        iterator = PngDataIterator(
            early_stop=1000,
            program=BB6_CONTENDER,
            width=320,
            height=180,
            pixel_policy="binning",
            frame_steps=[0, 100],
            colors=[],
            part_count=1,
        )
        results = list(iterator)
        assert len(results) == 2
    
    def test_auto_part_count(self):
        """Test that part_count=0 auto-detects CPU count."""
        iterator = PngDataIterator(
            early_stop=100,
            program=BB5_CHAMP,
            width=320,
            height=180,
            pixel_policy="binning",
            frame_steps=[0],
            colors=[],
            part_count=0,  # Auto-detect
        )
        results = list(iterator)
        assert len(results) == 1
    
    def test_multiple_parts(self):
        """Test with multiple parallel workers."""
        frame_steps = log_step_iterator(10000, 20)
        iterator = PngDataIterator(
            early_stop=10000,
            program=BB5_CHAMP,
            width=320,
            height=180,
            pixel_policy="binning",
            frame_steps=frame_steps,
            colors=[],
            part_count=4,
        )
        results = list(iterator)
        assert len(results) == 20
        
        # Verify step indices match requested steps
        actual_steps = [step_idx for step_idx, _ in results]
        assert actual_steps == frame_steps
