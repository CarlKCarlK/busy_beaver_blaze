use std::{default, mem, ops::Range, sync::atomic::AtomicU64};

use num_bigint::BigUint;
use num_traits::identities::Zero;

// atomic::AtomicUsize;
static RESULT: AtomicU64 = AtomicU64::new(0);

#[inline]
fn work_item() {
    RESULT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn simple_tetration(base: u32, height: u32) -> BigUint {
    let mut tetration = BigUint::from(1u32);
    for _ in 0..height {
        let mut power = BigUint::from(1u32);
        while !tetration.is_zero() {
            tetration -= 1u32;
            let mut product = BigUint::zero();
            while !power.is_zero() {
                power -= 1u32;
                // let mut sum = BigUint::zero();
                for _ in 0..base {
                    product += 1u32;
                    work_item();
                }
                // product = sum;
            }
            power = product;
        }
        tetration = power;
    }
    tetration
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

struct Product {
    a: u32,
    a_range: Range<u32>,
    b: BigUint,
    product_skips: ProductSkips,
    result: BigUint,
}

impl Product {
    fn new(a: u32, b: BigUint, product_skips: ProductSkips) -> Self {
        debug_assert!(a > 0); // cmk
        Product {
            a,
            a_range: 0..a,
            b,
            product_skips,
            result: BigUint::ZERO,
        }
    }

    fn into_result(self) -> BigUint {
        self.result
    }
}

impl Default for Product {
    fn default() -> Self {
        Self::new(1, BigUint::from(1u32), ProductSkips::None)
    }
}

impl Iterator for Product {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.b.is_zero() {
                return None;
            }
            let Some(a) = self.a_range.next() else {
                self.a_range = 0..self.a;
                self.b -= 1u32;
                continue;
            };
            self.result += 1u32;
            match self.product_skips {
                ProductSkips::None => return Some(()),
                ProductSkips::Column if a > 0 => return Some(()),
                ProductSkips::ColumnPlusOne if a > 0 => {
                    self.product_skips = ProductSkips::Column;
                }
                ProductSkips::Column | ProductSkips::ColumnPlusOne => {
                    debug_assert!(a == 0, "real assert")
                    // do nothing but loop
                }
            }
        }
    }
}

#[inline]
fn product(a: u32, b: BigUint, product_skips: ProductSkips) -> BigUint {
    debug_assert!(a > 0);
    let mut iter = Product::new(a, b, product_skips);
    for _ in iter.by_ref() {
        work_item();
    }
    iter.into_result()
}

#[inline]
fn product_old(a: u32, mut b: BigUint, mut product_skips: ProductSkips) -> BigUint {
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

struct Power {
    is_first: bool,
    a: u32,
    b: BigUint,
    product_skips: ProductSkips,
    product_iter: Product,
}

impl Power {
    fn new(a: u32, b: BigUint, power_skips: PowerSkips) -> Self {
        assert!(a > 0); // cmk
        let product_skips = ProductSkips::from(power_skips);
        let mut product = Product::default();
        product.next().unwrap(); // cmk
        Power {
            is_first: true,
            a,
            b,
            product_skips,
            product_iter: product,
        }
    }

    fn into_result(self) -> BigUint {
        self.product_iter.into_result()
    }
}

impl Iterator for Power {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_first {
            self.is_first = false;
            return Some(());
        }
        loop {
            if self.b.is_zero() {
                return None;
            }
            let Some(b) = self.product_iter.by_ref().next() else {
                let result = mem::take(&mut self.product_iter).into_result();
                self.product_iter = Product::new(self.a, result, self.product_skips);
                self.b -= 1u32;
                continue;
            };
            return Some(());
        }
    }
}

#[inline]
fn power_new(a: u32, b: BigUint, power_skips: PowerSkips) -> BigUint {
    let mut iter = Power::new(a, b, power_skips);
    for _ in iter.by_ref() {
        work_item();
    }
    iter.into_result()
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

// cmk!!!! BUGBUG can't run tests in parallel because of Global
fn main() -> Result<(), String> {
    let base = 2;
    for x in 0..5 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        let simple = simple_tetration(base, x);
        work_item();
        println!(
            "simple({x}): {simple} work_item_count = {}",
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

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
    use num_traits::ToPrimitive;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_increment() {
        RESULT.store(0, Ordering::Relaxed);
        work_item();
        assert_eq!(RESULT.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_product() {
        let base = 2;
        let x = 3u32;
        RESULT.store(0, Ordering::Relaxed);
        let result = product_old(base, BigUint::from(x), ProductSkips::None)
            .to_u64()
            .unwrap();
        assert_eq!(result, (base * x).into());
        assert_eq!(RESULT.load(Ordering::Relaxed), result);
    }

    #[test]
    fn test_product_new() {
        let base = 2;
        for x in 0u32..=10 {
            RESULT.store(0, Ordering::Relaxed);
            let result = product(base, BigUint::from(x), ProductSkips::None)
                .to_u64()
                .unwrap();
            println!(
                "{base}x{x}={result}:  work_item_count = {}",
                RESULT.load(Ordering::Relaxed)
            );
            assert_eq!(result, (base * x).into());
            assert_eq!(RESULT.load(Ordering::Relaxed), result);
        }
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
    fn test_power_new() {
        let base = 2;
        for x in 0u32..=10 {
            RESULT.store(0, Ordering::Relaxed);
            let result: u64 = power_new(base, BigUint::from(x), PowerSkips::None)
                .to_u64()
                .unwrap();
            println!(
                "{base}^{x}={result}:  work_item_count = {}",
                RESULT.load(Ordering::Relaxed)
            );
            assert_eq!(result, base.pow(x).into());
            assert_eq!(RESULT.load(Ordering::Relaxed), result);
        }
    }

    #[test]
    fn test_tetration() {
        let base: u32 = 2;
        let expecteds: [u64; 5] = [1, 2, 4, 16, 65536];
        for (x, expected) in (0u32..=4).zip(expecteds.iter()) {
            RESULT.store(0, Ordering::Relaxed);
            let result = tetration(base, x).to_u64().unwrap();
            assert_eq!(result, *expected);
            assert_eq!(RESULT.load(Ordering::Relaxed), result);
        }
    }
}
