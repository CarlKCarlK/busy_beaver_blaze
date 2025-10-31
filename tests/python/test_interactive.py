"""Test SpaceByTimeMachine interactive API."""

import pytest


def test_space_by_time_machine_basic():
    """Test basic SpaceByTimeMachine creation and stepping."""
    try:
        from busy_beaver_blaze import SpaceByTimeMachine, BB5_CHAMP
    except ImportError:
        pytest.skip("Rust extension not built")
    
    # Create machine
    machine = SpaceByTimeMachine(
        program=BB5_CHAMP,
        resolution=(100, 100),
        binning=True,
        skip=0
    )
    
    # Initial state
    assert machine.step_count() == 1  # Step 0 counted as step 1
    assert not machine.is_halted()
    
    # Step for a bit
    can_continue = machine.step_for_secs(0.01, early_stop=100)
    assert can_continue or machine.step_count() >= 100
    
    # Check state updated
    assert machine.step_count() > 1
    assert machine.count_nonblanks() > 0


def test_space_by_time_machine_png_rendering():
    """Test PNG rendering at various points."""
    try:
        from busy_beaver_blaze import SpaceByTimeMachine, BB5_CHAMP
    except ImportError:
        pytest.skip("Rust extension not built")
    
    machine = SpaceByTimeMachine(
        program=BB5_CHAMP,
        resolution=(100, 100),
        binning=True
    )
    
    # Render initial state
    png1 = machine.to_png()
    assert isinstance(png1, bytes)
    assert png1.startswith(b'\x89PNG')  # PNG magic bytes
    
    # Step and render again
    machine.step_for_secs(0.01, early_stop=50)
    png2 = machine.to_png()
    assert isinstance(png2, bytes)
    assert png2.startswith(b'\x89PNG')
    
    # Should be different (machine state changed)
    assert png1 != png2


def test_space_by_time_machine_custom_colors():
    """Test custom color palette."""
    try:
        from busy_beaver_blaze import SpaceByTimeMachine, BB5_CHAMP
    except ImportError:
        pytest.skip("Rust extension not built")
    
    # Create with custom colors
    machine = SpaceByTimeMachine(
        program=BB5_CHAMP,
        resolution=(100, 100),
        binning=True,
        colors=["#FFFFFF", "#FF0000", "#00FF00", "#0000FF", "#FFFF00"]
    )
    
    # Run to a specific point
    machine.step_for_secs(0.1, early_stop=100)
    png1 = machine.to_png()
    
    # Change colors and re-render
    machine.set_colors(["#000000", "#00FFFF", "#FF00FF", "#808080", "#C0C0C0"])
    png2 = machine.to_png()
    
    # Both should be valid PNGs
    assert png1.startswith(b'\x89PNG')
    assert png2.startswith(b'\x89PNG')
    
    # Should be different (different colors)
    assert png1 != png2


def test_space_by_time_machine_early_stop():
    """Test early_stop parameter."""
    try:
        from busy_beaver_blaze import SpaceByTimeMachine, BB5_CHAMP
    except ImportError:
        pytest.skip("Rust extension not built")
    
    machine = SpaceByTimeMachine(
        program=BB5_CHAMP,
        resolution=(50, 50),
        binning=True
    )
    
    # Run with low early_stop - may reach it in one call or multiple
    while machine.step_count() < 10:
        can_continue = machine.step_for_secs(0.01, early_stop=10)
        if not can_continue:
            break
    
    # Should be at or past early_stop
    assert machine.step_count() >= 10


def test_space_by_time_machine_with_skip():
    """Test skipping initial steps."""
    try:
        from busy_beaver_blaze import SpaceByTimeMachine, BB5_CHAMP
    except ImportError:
        pytest.skip("Rust extension not built")
    
    # Create with skip
    machine = SpaceByTimeMachine(
        program=BB5_CHAMP,
        resolution=(50, 50),
        binning=True,
        skip=100
    )
    
    # Should start at step 101 (1-indexed)
    assert machine.step_count() == 101
    
    # Should have non-zero tape content
    assert machine.count_nonblanks() > 0


def test_live_visualizer_import():
    """Test that LiveVisualizer can be imported (if IPython available)."""
    try:
        from busy_beaver_blaze.interactive import LiveVisualizer, visualize_live
        assert LiveVisualizer is not None
        assert visualize_live is not None
    except ImportError:
        # IPython not available in test environment - that's OK
        pytest.skip("IPython not available")
