use std::sync::atomic::AtomicU64;

use num_bigint::BigUint;
use num_traits::identities::Zero;

// atomic::AtomicUsize;
static RESULT: AtomicU64 = AtomicU64::new(0);

#[inline]
fn work_item() {
    RESULT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn product(a: u32, mut b: BigUint, skimp_b: bool, mut skimp_one_more: bool) -> BigUint {
    debug_assert!(a > 0);
    let mut result = BigUint::ZERO;
    while !b.is_zero() {
        b -= 1u32;

        // a=0
        result += 1u32;
        if !skimp_b {
            work_item();
        }
        for _ in 1..a {
            if !skimp_one_more {
                work_item();
            } else {
                skimp_one_more = false;
            }
            result += 1u32;
        }
    }
    result
}

#[inline]
fn power(a: u32, mut b: BigUint, skimp_work: bool) -> BigUint {
    let mut result = BigUint::from(1u32);
    work_item();
    if a == 0 {
        return result; // Rust says 0^0 is 1
    }
    while !b.is_zero() {
        b -= 1u32;
        result = product(a, result, true, skimp_work);
    }
    result
}

#[inline]
fn tetration(a: u32, b: u32) -> BigUint {
    debug_assert!(a > 0);
    let mut result = BigUint::from(1u32);
    work_item();

    for _ in 0..b {
        result = power(a, result, true);
    }

    result
}

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
    let running_total = product(base, BigUint::from(x), false, false);
    println!(
        "Multiply_i {base}x{x}={}:  work_item_count = {}",
        running_total,
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test power_i
    for x in 0u32..=10 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        let result = power(base, BigUint::from(x), false);
        println!(
            "Power_i {base}^{x}: {result} work_item_count = {}",
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    // Test tetration_i
    for x in 0u32..=4 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        let result = tetration(base, x);
        println!(
            "Tetration {base}^^{x}={result}:  work_item_count = {}",
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    Ok(())
}
