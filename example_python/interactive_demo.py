"""Demo script showing SpaceByTimeMachine interactive API.

This demonstrates the JavaScript/WASM-style API for Python notebooks,
allowing indefinite running with periodic rendering and graceful interruption.
"""

from busy_beaver_blaze import SpaceByTimeMachine, BB5_CHAMP

def main():
    print("Creating SpaceByTimeMachine for BB5 Champion...")
    machine = SpaceByTimeMachine(
        program=BB5_CHAMP,
        resolution=(800, 600),
        binning=True,
        skip=0
    )
    
    print(f"Initial state: Step {machine.step_count():,}")
    print("\nRunning for 100,000 steps with periodic snapshots...")
    print("(Press Ctrl+C to stop early)\n")
    
    snapshot_interval = 10_000
    next_snapshot = snapshot_interval
    
    try:
        while True:
            # Step for a bit (0.1 seconds of computation)
            can_continue = machine.step_for_secs(
                0.1,                         # Run for 0.1 seconds
                early_stop=100_000,          # Stop at 100k steps
                loops_per_time_check=10_000  # Check time every 10k loops
            )
            
            current_step = machine.step_count()
            
            # Show progress at intervals
            if current_step >= next_snapshot:
                png_bytes = machine.to_png()
                print(f"✓ Step {current_step:,}: "
                      f"Nonblanks={machine.count_nonblanks():,}, "
                      f"PNG size={len(png_bytes):,} bytes")
                next_snapshot += snapshot_interval
            
            # Check if we should stop
            if not can_continue:
                # Final snapshot
                png_bytes = machine.to_png()
                print(f"\n✓ Final state at step {current_step:,}:")
                print(f"  - Nonblanks: {machine.count_nonblanks():,}")
                print(f"  - Halted: {machine.is_halted()}")
                print(f"  - PNG size: {len(png_bytes):,} bytes")
                break
    
    except KeyboardInterrupt:
        current_step = machine.step_count()
        png_bytes = machine.to_png()
        print(f"\n⏹ Interrupted at step {current_step:,}:")
        print(f"  - Nonblanks: {machine.count_nonblanks():,}")
        print(f"  - PNG size: {len(png_bytes):,} bytes")
        print("\nMachine state preserved - could resume later!")

if __name__ == "__main__":
    main()
