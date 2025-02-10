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
fn multiply(a: u32, mut b: BigUint) {
    while !b.is_zero() {
        b -= 1u32;
        for _ in 0..a {
            work_item();
        }
    }
}

#[inline]
fn power_i(a: u32, mut b: BigUint) {
    let mut running_total = BigUint::from(1u32);
    work_item();
    if a == 0 {
        // Some leave 0^0 as undefined, but we'll define it as 1
        return;
    }
    let a_less_one = a - 1;
    while !b.is_zero() {
        b -= 1u32;
        let clone = running_total.clone();
        multiply(a_less_one, clone);
        running_total += &running_total * a_less_one;
    }
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
fn tetration_i(a: u32, b: u32) {
    if b == 0 {
        work_item();
        return;
    }
    let tetration_a_b_less_1 = tetration_fast(a, b - 1);
    power_i(a, tetration_a_b_less_1);
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
    multiply(base, BigUint::from(x));
    println!(
        "Multiply_i {base}x{x}:  work_item_count = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test power_i
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);

    let x = 3u32;
    power_i(base, BigUint::from(x));
    println!(
        "Power_i {base}^{x}:  work_item_count = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // test power_fast
    for x in 0u32..=4 {
        let power = power_fast(base, x.into());
        println!("Power_fast {base}^{x} = {}", power,);
    }

    // test tetration_fast
    for x in 0u32..=4 {
        let tetration = tetration_fast(base, x);
        println!("Tetration_fast {base}^^{x} = {}", tetration,);
    }

    // Test tetration_i
    for x in 0u32..=4 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        tetration_i(base, x);
        println!(
            "Tetration {base}^^{x}:  work_item_count = {}",
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    Ok(())
}
