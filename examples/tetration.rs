use std::sync::atomic::AtomicU64;

use num_bigint::BigUint;
use num_traits::identities::Zero;

// atomic::AtomicUsize;
static RESULT: AtomicU64 = AtomicU64::new(0);

#[inline]
fn work_item() {
    RESULT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[derive(PartialEq, Copy, Clone, Debug)]
enum ProductSkips {
    None,
    Column,
    ColumnPlusOne,
}

impl From<PowerSkips> for ProductSkips {
    fn from(skip: PowerSkips) -> Self {
        match skip {
            PowerSkips::PlusOne => ProductSkips::ColumnPlusOne,
            PowerSkips::None => ProductSkips::Column,
        }
    }
}

#[inline]
fn product(a: u32, mut b: BigUint, mut product_skips: ProductSkips) -> BigUint {
    debug_assert!(a > 0); // cmk
    let mut result = BigUint::ZERO;
    while !b.is_zero() {
        b -= 1u32;

        // a=0
        result += 1u32;
        if product_skips == ProductSkips::None {
            work_item();
        }
        for _ in 1..a {
            if product_skips == ProductSkips::ColumnPlusOne {
                product_skips = ProductSkips::Column;
            } else {
                work_item();
            }
            result += 1u32;
        }
    }
    result
}

#[derive(Copy, Clone, Debug)]
enum PowerSkips {
    None,
    PlusOne,
}

#[inline]
fn power(a: u32, mut b: BigUint, power_skips: PowerSkips) -> BigUint {
    let mut result = BigUint::from(1u32);
    work_item();
    if a == 0 {
        return result; // Rust says 0^0 is 1
    }
    let product_skips = ProductSkips::from(power_skips);
    while !b.is_zero() {
        b -= 1u32;
        result = product(a, result, product_skips);
    }
    result
}

#[inline]
fn tetration(a: u32, b: u32) -> BigUint {
    debug_assert!(a > 0);
    let mut result = BigUint::from(1u32);
    work_item();

    for _ in 0..b {
        result = power(a, result, PowerSkips::PlusOne);
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
    let running_total = product(base, BigUint::from(x), ProductSkips::None);
    println!(
        "Multiply_i {base}x{x}={}:  work_item_count = {}",
        running_total,
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test power_i
    for x in 0u32..=10 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        let result = power(base, BigUint::from(x), PowerSkips::None);
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

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_traits::ToPrimitive;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_increment() {
        RESULT.store(0, Ordering::Relaxed);
        work_item();
        assert_eq!(RESULT.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_multiply() {
        let base = 2;
        let x = 3u32;
        RESULT.store(0, Ordering::Relaxed);
        let result = product(base, BigUint::from(x), ProductSkips::None)
            .to_u64()
            .unwrap();
        assert_eq!(result, (base * x).into());
        assert_eq!(RESULT.load(Ordering::Relaxed), result);
    }

    #[test]
    fn test_power() {
        let base = 2;
        for x in 0u32..=10 {
            RESULT.store(0, Ordering::Relaxed);
            let result: u64 = power(base, BigUint::from(x), PowerSkips::None)
                .to_u64()
                .unwrap();
            assert_eq!(result, base.pow(x).into());
            assert_eq!(RESULT.load(Ordering::Relaxed), result);
        }
    }

    #[test]
    fn test_tetration() {
        let base: u32 = 2;
        let mut expecteds: [u64; 5] = [1, 2, 4, 16, 65536];
        for (x, expected) in (0u32..=4).zip(expecteds.iter()) {
            RESULT.store(0, Ordering::Relaxed);
            let result = tetration(base, x).to_u64().unwrap();
            assert_eq!(RESULT.load(Ordering::Relaxed), result);
        }
    }
}
