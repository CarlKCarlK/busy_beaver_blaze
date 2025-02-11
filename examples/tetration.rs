use std::sync::atomic::AtomicU64;

use num_bigint::BigUint;
use num_traits::{identities::Zero, ConstZero};

// atomic::AtomicUsize;
static RESULT: AtomicU64 = AtomicU64::new(0);

#[inline]
fn work_item() {
    RESULT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn multiply_skimp_work(a: u32, mut b: BigUint) -> BigUint {
    debug_assert!(a > 0);
    let mut result = BigUint::ZERO;
    while !b.is_zero() {
        b -= 1u32;
        result += 1u32;
        for _ in 1..a {
            work_item();
            result += 1u32;
        }
    }
    result
}

#[inline]
fn power_i(a: u32, mut b: BigUint) -> BigUint {
    let mut running_total = BigUint::from(1u32);
    work_item();
    if a == 0 {
        // Some leave 0^0 as undefined, but we'll define it as 1
        return running_total;
    }
    while !b.is_zero() {
        b -= 1u32;
        print!("{}*{} = ", a, running_total);
        running_total = multiply_skimp_work(a, running_total);
        println!("rt{}", running_total);
    }
    running_total
}

// #[inline]
// fn tetration_i(a: u32, b: u32) {
//     if b == 0 {
//         work_item();
//         return;
//     }
//     let tetration_a_b_less_1 = tetration_fast(a, b - 1);
//     power_i(a, tetration_a_b_less_1);
// }

fn main() -> Result<(), String> {
    let base = 2;
    // Test increment
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    work_item();
    println!(
        "Increment:  work_item_count = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test multiply_i
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);

    let x = 3u32;
    let running_total = multiply_skimp_work(base, BigUint::from(x));
    println!(
        "Multiply_i {base}x{x}={}:  work_item_count = {}",
        running_total,
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test power_i
    for x in 0u32..=10 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        power_i(base, BigUint::from(x));
        println!(
            "Power_i {base}^{x}:  work_item_count = {}",
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    // // Test tetration_i
    // for x in 0u32..=4 {
    //     RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    //     tetration_i(base, x);
    //     println!(
    //         "Tetration {base}^^{x}:  work_item_count = {}",
    //         RESULT.load(std::sync::atomic::Ordering::Relaxed)
    //     );
    // }

    Ok(())
}
