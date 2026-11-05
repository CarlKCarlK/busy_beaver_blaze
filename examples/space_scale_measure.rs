//! Rendering times and temporary allocator peaks, independently of the simulator benchmark.
use busy_beaver_blaze::{BB5_CHAMP, PyramidMachine};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

struct CountingAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

fn record_allocation(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
}

// SAFETY: Every operation delegates the same pointer, size and alignment to System.
// Counters observe allocation sizes and never dereference allocation pointers.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller provides the valid Layout required by GlobalAlloc.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: Forward the original allocation pointer and Layout unchanged.
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: Forward the valid allocation pointer, old Layout and requested size.
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            record_allocation(new_size);
        }
        pointer
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() {
    for (name, program) in [
        ("BB5", BB5_CHAMP.to_owned()),
        (
            "26 layers",
            (0..26)
                .map(|state_index| {
                    let symbol = char::from_digit(state_index + 1, 36).unwrap();
                    let next_state = char::from(b'A' + ((state_index + 1) % 26) as u8);
                    format!("{symbol}R{next_state}").repeat(27)
                })
                .collect::<Vec<_>>()
                .join("_"),
        ),
    ] {
        let mut machine = PyramidMachine::new(&program, 360, 432, "linear-b").unwrap();
        for _step_index in 0..1_000_000 {
            assert!(machine.step());
        }
        // Normalize the reference's row storage before comparing retained bytes.
        black_box(machine.linear_png(&[255, 255, 255, 255, 165, 0]).unwrap());
        let retained = machine.retained_bytes();
        let checkpoints = machine.checkpoint_count();
        println!("{name}: checkpoints={checkpoints} retained_bytes={retained}");
        for time_scale in ["linear-a", "linear-b", "sqrt", "log"] {
            if time_scale != "linear-a" {
                machine.set_time_scale(time_scale).unwrap();
            }
            for space_scale in ["linear", "row"] {
                machine.set_space_scale(space_scale).unwrap();
                let base = LIVE.load(Ordering::Relaxed);
                PEAK.store(base, Ordering::Relaxed);
                let allocations = ALLOCATIONS.load(Ordering::Relaxed);
                let start = Instant::now();
                for _frame in 0..5 {
                    let png = if time_scale == "linear-a" {
                        machine.linear_png(&[255, 255, 255, 255, 165, 0])
                    } else {
                        machine.to_png(&[255, 255, 255, 255, 165, 0])
                    };
                    black_box(png.unwrap());
                }
                let elapsed_ms = start.elapsed().as_secs_f64() * 200.0;
                let peak = PEAK.load(Ordering::Relaxed);
                assert!(peak >= base);
                let temporary_bytes = peak - base;
                let allocations = (ALLOCATIONS.load(Ordering::Relaxed) - allocations) / 5;
                assert_eq!(retained, machine.retained_bytes());
                assert_eq!(checkpoints, machine.checkpoint_count());
                println!(
                    "Y={time_scale} X={space_scale} png_ms={elapsed_ms:.3} temporary_peak_bytes={temporary_bytes} allocations_per_frame={allocations}"
                );
            }
        }
    }
}
