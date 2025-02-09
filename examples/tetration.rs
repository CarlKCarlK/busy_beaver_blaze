use num_bigint::BigInt;
use num_traits::identities::Zero;
use std::sync::atomic::{AtomicU64, Ordering};

static TOTAL_WORK: AtomicU64 = AtomicU64::new(0);

fn increment(x: BigInt) -> BigInt {
    let count = TOTAL_WORK.fetch_add(1, Ordering::Relaxed);
    if count % 1 == 0 {
        println!("\tIncrements index: {}", count);
    }
    x + 1
}

fn plus_r(base: u32, x: BigInt) -> BigInt {
    if base == 0 {
        return x;
    }
    plus_r(base - 1, increment(x))
}

#[inline]
fn plus_i(base: u32, x: BigInt) -> BigInt {
    let mut result = x;
    for _ in 0..base {
        result = increment(result);
    }
    result
}
fn multiply_r(base: u32, x: BigInt) -> BigInt {
    if x.is_zero() {
        return BigInt::from(0);
    }
    let increment_by = plus_r(base, BigInt::from(0)); // Calculate once
    multiply_r(base, x - 1) + increment_by // Reuse the pre-calculated value
}

#[inline]
fn multiply_i(base: u32, mut x: BigInt) -> BigInt {
    let mut result = BigInt::from(0);
    while !x.is_zero() {
        x -= 1;
        result = plus_i(base, result);
    }
    result
}

fn power_r(base: u32, x: BigInt) -> BigInt {
    if x.is_zero() {
        return BigInt::from(1);
    }
    multiply_r(base, power_r(base, x - 1))
}

#[inline]
fn power_i(base: u32, mut x: BigInt) -> BigInt {
    let mut result = BigInt::from(1);
    while !x.is_zero() {
        x -= 1;
        println!("--");
        result = multiply_i(base, BigInt::from(1));
    }
    result
}

fn tetration_r(base: u32, x: BigInt) -> BigInt {
    if x.is_zero() {
        return BigInt::from(1);
    }
    power_r(base, tetration_r(base, x - 1))
}

#[inline]
fn tetration_i(base: u32, mut x: BigInt) -> BigInt {
    let mut result = BigInt::from(1);
    while !x.is_zero() {
        x -= 1;
        result = power_i(base, result);
    }
    result
}

fn main() -> Result<(), String> {
    let base = 2;

    // Test increment
    let x = BigInt::from(5);
    println!("Increment:");
    TOTAL_WORK.store(0, Ordering::Relaxed);
    println!("  Recursive: {x}++ = {}", increment(x.clone()));
    println!("Total increments: {}", TOTAL_WORK.load(Ordering::Relaxed));

    // Test both recursive and iterative versions
    let x = BigInt::from(5);
    println!("Plus:");
    TOTAL_WORK.store(0, Ordering::Relaxed);
    println!("  Iterative: {base}+{x} = {}", plus_i(base, x.clone()));
    println!("Total increments: {}", TOTAL_WORK.load(Ordering::Relaxed));
    TOTAL_WORK.store(0, Ordering::Relaxed);
    println!("  Recursive: {base}+{x} = {}", plus_r(base, x.clone()));
    println!("Total increments: {}", TOTAL_WORK.load(Ordering::Relaxed));

    let x = BigInt::from(3);
    println!("\nMultiply:");
    TOTAL_WORK.store(0, Ordering::Relaxed);
    println!("  Iterative: {base}×{x} = {}", multiply_i(base, x.clone()));
    println!("Total increments: {}", TOTAL_WORK.load(Ordering::Relaxed));
    TOTAL_WORK.store(0, Ordering::Relaxed);
    println!("  Recursive: {base}×{x} = {}", multiply_r(base, x.clone()));
    println!("Total increments: {}", TOTAL_WORK.load(Ordering::Relaxed));

    let x = BigInt::from(3);
    println!("\nPower:");
    TOTAL_WORK.store(0, Ordering::Relaxed);
    println!("  Iterative: {base}^{x} = {}", power_i(base, x.clone()));
    println!("Total increments: {}", TOTAL_WORK.load(Ordering::Relaxed));
    TOTAL_WORK.store(0, Ordering::Relaxed);
    println!("  Recursive: {base}^{x} = {}", power_r(base, x.clone()));
    println!("Total increments: {}", TOTAL_WORK.load(Ordering::Relaxed));

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
