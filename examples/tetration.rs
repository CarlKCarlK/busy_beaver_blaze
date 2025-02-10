use std::sync::atomic::AtomicU64;

use num_bigint::BigInt;
use num_traits::identities::Zero;

// atomic::AtomicUsize;
static RESULT: AtomicU64 = AtomicU64::new(0);

#[inline]
fn increment() {
    RESULT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn plus_r(base: u32) {
    if base == 0 {
        return;
    }
    increment();
    plus_r(base - 1);
}

#[inline]
fn plus_i(base: u32) {
    for _ in 0..base {
        increment();
    }
}

fn multiply_r(base: u32, x: BigInt) {
    if x.is_zero() {
        return;
    }
    plus_i(base);
    multiply_r(base, x - 1);
}

#[inline]
fn multiply_i(base: u32, mut x: BigInt) {
    while !x.is_zero() {
        x -= 1;
        plus_i(base);
    }
}

fn power_r(base: u32, x: BigInt) {
    if x.is_zero() {
        increment();
        return;
    }
    power_r(base, x - 1);
    multiply_r(
        base - 1,
        RESULT.load(std::sync::atomic::Ordering::Relaxed).into(),
    );
}

#[inline]
fn power_i(base: u32, mut x: BigInt) {
    increment();
    while !x.is_zero() {
        x -= 1;
        multiply_i(
            base - 1,
            RESULT.load(std::sync::atomic::Ordering::Relaxed).into(),
        );
    }
}

// fn tetration_r(base: u32, x: BigInt) -> BigInt {
//     if x.is_zero() {
//         return BigInt::from(1);
//     }
//     power_r(base, tetration_r(base, x - 1))
// }

// #[inline]
// fn tetration_i(base: u32, mut x: BigInt) -> BigInt {
//     let mut result = BigInt::from(1);
//     while !x.is_zero() {
//         x -= 1;
//         result = power_i(base, result);
//     }
//     result
// }

fn main() -> Result<(), String> {
    let base = 2;

    // Test increment
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    println!(
        "Increment: before = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );
    increment();
    println!(
        "Increment: after = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test plus_i
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    println!(
        "Plus: before = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );
    plus_i(base);
    println!(
        "Plus_i +{base}: after = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test plus_r
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    println!(
        "Plus: before = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );
    plus_r(base);
    println!(
        "Plus_r +{base}: after = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test multiply_i
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    println!(
        "Multiply: before = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );
    let x = BigInt::from(3);
    multiply_i(base, x.clone());
    println!(
        "Multiply_i {base}x{x}: after = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test multiply_r
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    println!(
        "Multiply: before = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );
    let x = BigInt::from(3);
    multiply_r(base, x.clone());
    println!(
        "Multiply_r {base}x{x}: after = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test power_r
    for x in 0..5 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        println!(
            "Power: before = {}",
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
        power_r(base, BigInt::from(x));
        println!(
            "Power_r {base}^{x}: after = {}",
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    // Test power_i
    for x in 0..5 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        println!(
            "Power: before = {}",
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
        power_i(base, BigInt::from(x));
        println!(
            "Power_i {base}^{x}: after = {}",
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
    // let x = BigInt::from(3);
    // println!("\nPower:");
    // TOTAL_WORK.store(0, Ordering::Relaxed);
    // println!("  Iterative: {base}^{x} = {}", power_i(base, x.clone()));
    // println!("Total increments: {}", TOTAL_WORK.load(Ordering::Relaxed));
    // TOTAL_WORK.store(0, Ordering::Relaxed);
    // println!("  Recursive: {base}^{x} = {}", power_r(base, x.clone()));
    // println!("Total increments: {}", TOTAL_WORK.load(Ordering::Relaxed));

    // let x = BigInt::from(3);
    // let base = 2;
    // println!("\nTetration:");
    // TOTAL_WORK.store(0, Ordering::Relaxed);
    // println!(
    //     "  Recursive: {base}↑↑{x} = {}",
    //     tetration_r(base, x.clone())
    // );
    // TOTAL_WORK.store(0, Ordering::Relaxed);
    // println!(
    //     "  Iterative: {base}↑↑{x} = {}",
    //     tetration_i(base, x.clone())
    // );

    Ok(())
}
