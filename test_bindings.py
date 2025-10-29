"""Quick test to verify PyO3 bindings work."""

import sys
import os

# Add the target directory to Python path to import the built extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "target-nightly", "release"))

try:
    from busy_beaver_blaze import _busy_beaver_blaze
    print("✓ Successfully imported _busy_beaver_blaze")
    
    # Check that the class is available
    print(f"✓ PyPngDataIterator available: {hasattr(_busy_beaver_blaze, 'PyPngDataIterator')}")
    
    # Check constants
    print(f"✓ BB5_CHAMP available: {hasattr(_busy_beaver_blaze, 'BB5_CHAMP')}")
    print(f"✓ BB6_CONTENDER available: {hasattr(_busy_beaver_blaze, 'BB6_CONTENDER')}")
    
    # Try to create an iterator
    PyPngDataIterator = _busy_beaver_blaze.PyPngDataIterator
    BB5_CHAMP = _busy_beaver_blaze.BB5_CHAMP
    
    print(f"\nBB5_CHAMP program:\n{BB5_CHAMP[:100]}...")
    
    iterator = PyPngDataIterator(
        early_stop=100,
        program=BB5_CHAMP,
        width=320,
        height=180,
        pixel_policy="binning",
        frame_steps=[0, 10],
        colors=[],
        part_count=1
    )
    print("✓ Created PyPngDataIterator instance")
    
    # Try to iterate
    results = list(iterator)
    print(f"✓ Got {len(results)} frames")
    
    if results:
        step_idx, png_bytes = results[0]
        print(f"✓ First frame: step={step_idx}, size={len(png_bytes)} bytes")
        print(f"✓ PNG magic bytes: {png_bytes[:8]}")
        assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n', "Not a valid PNG file"
        print("✓ Valid PNG file")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
