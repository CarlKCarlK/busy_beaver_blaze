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
fn power_fast(a: u32, mut b: BigUint) -> BigUint {
    let mut running_total = BigUint::from(1u32);
    if a == 0 {
        // Some leave 0^0 as undefined, but we'll define it as 1
        return running_total;
    }
    let a_less_one = a - 1;
    while !b.is_zero() {
        b -= 1u32;
        running_total += &running_total * a_less_one;
    }
    running_total
}

#[inline]
fn tetration_fast(a: u32, b: u32) -> BigUint {
    let mut running_total = BigUint::from(1u32);
    for _ in 0..b {
        running_total = power_fast(a, running_total);
    }
    running_total
}

#[inline]
fn tetration_i(a: u32, b: u32) -> BigUint {
    let mut running_total = BigUint::ZERO;
    let mut zero = BigUint::ZERO;
    if b == 0 {
        increment(&mut running_total, &mut zero);
        return running_total;
    }
    let tetration_a_b_less_1 = tetration_fast(a, b - 1);
    power_i(a, tetration_a_b_less_1, &mut zero)
}

fn main() -> Result<(), String> {
    let base = 2;
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

    // test power_fast
    for x in 0u32..=4 {
        let power = power_fast(base, x.into());
        println!("Power_fast {base}^{x} = {}", power,);
    }

    // test tetration_fast
    for x in 0u32..=5 {
        let tetration = tetration_fast(base, x);
        println!("Tetration_fast {base}^^{x} = {}", tetration,);
    }

    // Test tetration_i
    for x in 0u32..=4 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        let running_total = tetration_i(base, x);
        println!(
            "Tetration {base}^^{x}: after = {}, work_item_count = {}",
            running_total,
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    Ok(())
}
