use std::sync::atomic::AtomicU64;

use num_bigint::BigUint;
use num_traits::identities::Zero;
use std::ops::SubAssign;

trait Decrementable: Zero + SubAssign<u32> {}

// Blanket implementation for any type that meets the requirements
impl<T: Zero + SubAssign<u32>> Decrementable for T {}

struct CountDown<T: Decrementable> {
    current: T,
}

impl<T: Decrementable> Iterator for CountDown<T> {
    type Item = ();

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_zero() {
            None
        } else {
            self.current -= 1u32;
            Some(())
        }
    }
}

trait IntoCountDown {
    fn into_count_down(self) -> CountDown<Self>
    where
        Self: Sized + Decrementable;
}

impl<T: Decrementable> IntoCountDown for T {
    #[inline]
    fn into_count_down(self) -> CountDown<T> {
        CountDown { current: self }
    }
}

struct CountDownMutRef<'a, T: Decrementable> {
    current: &'a mut T,
}

impl<T: Decrementable> Iterator for CountDownMutRef<'_, T> {
    type Item = ();

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_zero() {
            None
        } else {
            *self.current -= 1u32;
            Some(())
        }
    }
}

trait CountDownMut {
    fn count_down(&mut self) -> CountDownMutRef<Self>
    where
        Self: Sized + Decrementable;
}

impl<T: Decrementable> CountDownMut for T {
    #[inline]
    fn count_down(&mut self) -> CountDownMutRef<T> {
        CountDownMutRef { current: self }
    }
}

// atomic::AtomicUsize;
static RESULT: AtomicU64 = AtomicU64::new(0);

#[inline]
fn work_item_a() {
    RESULT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn tetration_s(a: u32, b: u32) -> BigUint {
    debug_assert!(a > 0);
    // let f = format!("| {a}^^{b} = ");
    // println!("{f}?");
    let mut result = BigUint::from(1u32);
    for _ in 0..b {
        result = power_s(a, result);
    }
    // println!("{f}{result}");
    result
}

#[inline]
fn tetration_f(a: u32, b: u32) -> BigUint {
    debug_assert!(a > 0);
    (0..b).fold(BigUint::from(1u32), |acc, _| power_f(a, acc))
}

#[inline]
fn tetration_f_simple(a: u32, b: u32) -> BigUint {
    debug_assert!(a > 0);
    (0..b).fold(BigUint::from(1u32), |tetration, _| {
        tetration
            .into_count_down()
            .fold(BigUint::from(1u32), |power, _| {
                power.into_count_down().fold(BigUint::ZERO, |product, _| {
                    (0..a).fold(product, |sum, _| {
                        let mut increment = sum;
                        increment += 1u32;
                        increment
                    })
                })
            })
    })
}

#[inline]
fn tetration_f_vector_simple(a: u32, b: u32) -> Vec<()> {
    debug_assert!(a > 0);
    let zero = vec![];
    let one = vec![()];
    (0..b).fold(one.clone(), |tetration, _| {
        tetration.into_iter().fold(one.clone(), |power, _| {
            power.into_iter().fold(zero.clone(), |product, _| {
                (0..a).fold(product, |sum, _| {
                    let mut increment = sum;
                    increment.push(());
                    increment
                })
            })
        })
    })
}

#[inline]
fn power_s(a: u32, b: BigUint) -> BigUint {
    debug_assert!(a > 0);
    // let f = format!("|\t{a}^{b} = ");
    // println!("{f}?");
    let mut result = BigUint::from(1u32);
    for _ in b.into_count_down() {
        result = product_s(a, result);
    }
    // println!("{f}{result}");
    result
}

#[inline]
fn power_f(a: u32, b: BigUint) -> BigUint {
    debug_assert!(a > 0);
    b.into_count_down()
        .fold(BigUint::from(1u32), |acc, _| product_f(a, acc))
}

#[inline]
fn product_s(a: u32, b: BigUint) -> BigUint {
    debug_assert!(a > 0); // cmk
                          // let f = format!("|\t\t{a}*{b} = ");
    let mut result = BigUint::ZERO;
    for _ in b.into_count_down() {
        result = add_ownership(a, result);
    }
    // println!("{f}{result}");
    result
}

#[inline]
fn product_f(a: u32, b: BigUint) -> BigUint {
    debug_assert!(a > 0);
    b.into_count_down()
        .fold(BigUint::ZERO, |acc, _| add_functional(a, acc))
}

#[inline]
fn add(a: u32, acc: &mut BigUint) {
    for _ in 0..a {
        increment(acc);
    }
}

#[inline]
fn increment(acc: &mut BigUint) {
    *acc += 1u32;
}

#[inline]
fn multiply(a: u32, acc0: &mut BigUint) {
    assert!(a > 0, "a must be greater than 0");

    let mut acc1 = BigUint::ZERO;
    increment(&mut acc1);

    for _ in acc0.count_down() {
        add(a, &mut acc1);
    }
    *acc0 = acc1;
}

#[inline]
fn add_ownership(a: u32, b: BigUint) -> BigUint {
    let mut result = b;
    for _ in 0..a {
        result = increment_ownership(result);
    }
    result
}

#[inline]
fn increment_ownership(a: BigUint) -> BigUint {
    let mut result = a;
    result += 1u32;
    result
}

#[inline]
fn add_functional(a: u32, b: BigUint) -> BigUint {
    (0..a).fold(b, |acc, _| increment_functional(acc))
}

#[inline]
fn increment_functional(mut a: BigUint) -> BigUint {
    a += 1u32;
    a
}

// no cloning of BigUint. Lots of ownership passing.
fn simple_tetration<F>(base: u32, height: u32, mut work_item: F) -> BigUint
where
    F: FnMut(),
{
    let mut tetration = BigUint::from(1u32);
    for _ in 0..height {
        let mut power = BigUint::from(1u32);
        for _ in tetration.into_count_down() {
            let mut product = BigUint::zero();
            for _ in power.into_count_down() {
                let mut sum = product;
                for _ in 0..base {
                    let mut increment = sum;
                    increment += 1u32;
                    work_item();
                    sum = increment;
                }
                product = sum;
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

#[inline]
fn product<F>(a: u32, b: BigUint, mut product_skips: ProductSkips, mut work_item: F) -> BigUint
where
    F: FnMut(),
{
    debug_assert!(a > 0); // cmk
    let mut result = BigUint::ZERO;
    for _ in b.into_count_down() {
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
fn power<F>(a: u32, b: BigUint, power_skips: PowerSkips, mut work_item: F) -> BigUint
where
    F: FnMut(),
{
    let mut result = BigUint::from(1u32);
    work_item();
    if a == 0 {
        return result; // Rust says 0^0 is 1
    }
    let product_skips = ProductSkips::from(power_skips);
    for _ in b.into_count_down() {
        result = product(a, result, product_skips, &mut work_item);
    }
    result
}

#[inline]
fn tetration<F>(a: u32, b: u32, mut work_item: F) -> BigUint
where
    F: FnMut(),
{
    debug_assert!(a > 0);
    let mut result = BigUint::from(1u32);
    work_item();

    for _ in 0..b {
        result = power(a, result, PowerSkips::PlusOne, &mut work_item);
    }

    result
}

fn slow_enough() {
    for a in 0..u128::MAX {
        for b in 0..u128::MAX {
            for c in 0..u128::MAX {
                for d in 0..u128::MAX {
                    if d % 1_000_000_000 == 0 {
                        println!("{} {} {} {}", a, b, c, d);
                    }
                }
            }
        }
    }
}

// cmk!!!! BUG BUG can't run tests in parallel because of Global
fn main() -> Result<(), String> {
    let mut b = BigUint::ZERO;
    add(2, &mut b);
    assert_eq!(b, BigUint::from(2u32));

    let mut b = BigUint::from(3u32);
    multiply(2, &mut b);
    assert_eq!(b, BigUint::from(6u32));

    let c = add_ownership(2, BigUint::ZERO);
    assert_eq!(c, BigUint::from(2u32));

    let c = add_functional(2, BigUint::ZERO);
    assert_eq!(c, BigUint::from(2u32));

    slow_enough();
    // // add
    // let s = add_s(2, BigUint::from(3u32));
    // println!("add_s(2, 3) = {}", s);
    // // product
    // let p = product_s(2, BigUint::from(3u32));
    // println!("product_s(2, 3) = {}", p);
    // // power
    // let p = power_s(2, BigUint::from(4u32));
    // println!("power_s(2, 4) = {}", p);

    let t = tetration_f_vector_simple(2, 4).len();
    println!("tetration_f_vector(2, 4) = {}", t);

    let t = tetration_f_simple(2, 4);
    println!("tetration_f_simple(2, 4) = {}", t);

    let t = tetration_f(2, 4);
    println!("tetration_f(2, 4) = {}", t);

    let t = tetration_s(2, 4);
    println!("tetration_s(2, 4) = {}", t);

    let base = 2;
    for x in 0..5 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);

        let simple = simple_tetration(base, x, work_item_a);
        println!(
            "simple({x}): {simple} work_item_count = {}",
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    // Test increment
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
    work_item_a();
    println!(
        "Increment:  work_item_count = {}",
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test multiply_i
    RESULT.store(0, std::sync::atomic::Ordering::Relaxed);

    let x = 3u32;
    let running_total = product(base, BigUint::from(x), ProductSkips::None, work_item_a);
    println!(
        "Multiply_i {base}x{x}={}:  work_item_count = {}",
        running_total,
        RESULT.load(std::sync::atomic::Ordering::Relaxed)
    );

    // Test power_i
    for x in 0u32..=10 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        let result = power(base, BigUint::from(x), PowerSkips::None, work_item_a);
        println!(
            "Power_i {base}^{x}: {result} work_item_count = {}",
            RESULT.load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    // Test tetration_i
    for x in 0u32..=4 {
        RESULT.store(0, std::sync::atomic::Ordering::Relaxed);
        let result = tetration(base, x, work_item_a);
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
        work_item_a();
        assert_eq!(RESULT.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_product() {
        let base = 2;
        let x = 3u32;
        RESULT.store(0, Ordering::Relaxed);
        let result = product(base, BigUint::from(x), ProductSkips::None, work_item_a)
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
            let result: u64 = power(base, BigUint::from(x), PowerSkips::None, work_item_a)
                .to_u64()
                .unwrap();
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
            let result = tetration(base, x, work_item_a).to_u64().unwrap();
            assert_eq!(result, *expected);
            assert_eq!(RESULT.load(Ordering::Relaxed), result);
        }
    }
}
