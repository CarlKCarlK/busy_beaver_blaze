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
fn increment(running_total: &mut BigUint, skip_work_count: &mut BigUint) {
    *running_total += 1u32;
    if skip_work_count.is_zero() {
        work_item();
    } else {
        *skip_work_count -= 1u32;
    }
}

// fn plus_r(base: u32) {
//     if base == 0 {
//         return;
//     }
//     increment();
//     plus_r(base - 1);
// }

#[inline]
fn plus_i(running_total: &mut BigUint, base: u32, skip_work_count: &mut BigUint) {
    for _ in 0..base {
        increment(running_total, skip_work_count);
    }
}

// fn multiply_r(base: u32, x: BigInt) {
//     if x.is_zero() {
//         return;
//     }
//     plus_i(base);
//     multiply_r(base, x - 1);
// }

#[inline]
fn add_multiply(
    running_total: &mut BigUint,
    a: u32,
    mut b: BigUint,
    skip_work_count: &mut BigUint,
) {
    while !b.is_zero() {
        b -= 1u32;
        plus_i(running_total, a, skip_work_count);
    }
}

// fn power_r(base: u32, x: BigInt) {
//     if x.is_zero() {
//         increment();
//         return;
//     }
//     power_r(base, x - 1);
//     multiply_r(
//         base - 1,
//         RESULT.load(std::sync::atomic::Ordering::Relaxed).into(),
//     );
// }

#[inline]
fn power_i(a: u32, mut b: BigUint, skip_work_count: &mut BigUint) -> BigUint {
    let mut running_total = BigUint::ZERO;
    increment(&mut running_total, skip_work_count);
    if a == 0 {
        // Some leave 0^0 as undefined, but we'll define it as 1
        return running_total;
    }
    let a_less_one = a - 1;
    while !b.is_zero() {
        b -= 1u32;
        let clone = running_total.clone();
        add_multiply(&mut running_total, a_less_one, clone, skip_work_count);
    }
    running_total
}

#[inline]
fn tetration_r(a: u32, b: BigUint) -> BigUint {
    let mut running_total = BigUint::ZERO;
    if b.is_zero() {
        let mut zero = BigUint::zero();
        increment(&mut running_total, &mut zero);
        return running_total;
    }
    let start = RESULT.load(std::sync::atomic::Ordering::Relaxed);
    let exp = tetration_r(a, b - 1u32);
    let change = RESULT.load(std::sync::atomic::Ordering::Relaxed) - start;
    power_i(a, exp, &mut BigUint::from(change))
}

// #[inline]
// fn tetration_i(base: u32, mut x: BigInt) {
//     increment();
//     while !x.is_zero() {
//         x -= 1;
//         power_i(
//             base,
//             RESULT.load(std::sync::atomic::Ordering::Relaxed).into(),
//         );
//     }
// }

fn main() -> Result<(), String> {
    let base = 3;
    let mut skip_work_count = BigUint::ZERO;

    // Test increment
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut x = BigUint::ZERO;
    increment(&mut x, &mut skip_work_count);
    println!(
        "Increment: after = {}, work_item_count = {}",
        x,
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test plus_i
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    let start = 3u32;

    let mut x = BigUint::from(start);
    plus_i(&mut x, base, &mut skip_work_count);
    println!(
        "Plus_i {start}+{base}: after = {}, work_item_count = {}",
        x,
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test multiply_i
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut running_total = BigUint::ZERO;

    let x = 3u32;
    add_multiply(
        &mut running_total,
        base,
        BigUint::from(x),
        &mut skip_work_count,
    );
    println!(
        "Multiply_i {base}x{x}: after = {}, work_item_count = {}",
        running_total,
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test power_i
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);

    let x = 3u32;
    let running_total = power_i(base, BigUint::from(x), &mut skip_work_count);
    println!(
        "Power_i {base}^{x}: after = {}, work_item_count = {}",
        running_total,
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test tetration_i
    for x in 0u32..=4 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        let running_total = tetration_r(base, BigUint::from(x));
        println!(
            "Tetration_r {base}^^{x}: after = {}, work_item_count = {}",
            running_total,
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
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
