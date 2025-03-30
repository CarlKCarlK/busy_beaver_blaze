#![allow(clippy::min_ident_chars)]
use core::ops::SubAssign;
use num_bigint::BigUint;
use num_traits::identities::Zero;

trait Decrementable: Zero + SubAssign<u32> {}

// Blanket implementation for any type that meets the requirements
impl<T: Zero + SubAssign<u32>> Decrementable for T {}

struct CountDownMutRef<'a, T: Decrementable> {
    current: &'a mut T,
}

#[allow(clippy::missing_trait_methods)]
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

fn sum(a: u32, sum: &mut BigUint) {
    for _ in 0..a {
        *sum += 1u32;
    }
}

fn product(a: u32, product: &mut BigUint) {
    let mut sum = BigUint::ZERO;
    for () in product.count_down() {
        for _ in 0..a {
            sum += 1u32;
        }
    }
    *product = sum;
}

fn power(a: u32, power: &mut BigUint) {
    assert!(
        a > 0 || *power > BigUint::ZERO,
        "a must be greater than 0 or power greater than 0"
    );

    let mut product = BigUint::ZERO;
    product += 1u32;
    for () in power.count_down() {
        let mut sum = BigUint::ZERO;
        for () in product.count_down() {
            for _ in 0..a {
                sum += 1u32;
            }
        }
        product = sum;
    }
    *power = product;
}

fn tetrate(a: u32, tetration: &mut BigUint) {
    assert!(a > 0, "a must be greater than 0");

    let mut power = BigUint::ZERO;
    power += 1u32;
    for () in tetration.count_down() {
        let mut product = BigUint::ZERO;
        product += 1u32;
        for () in power.count_down() {
            let mut sum = BigUint::ZERO;
            for () in product.count_down() {
                for _ in 0..a {
                    sum += 1u32;
                }
            }
            product = sum;
        }
        power = product;
    }
    *tetration = power;
}

#[allow(clippy::shadow_reuse)]
#[allow(clippy::shadow_unrelated)]
fn main() {
    let a = 2;
    let mut b = BigUint::from(4u32);
    print!("{a} + {b}\t= ");
    sum(a, &mut b);
    assert_eq!(b, BigUint::from(6u32));
    println!("{b}");

    let a = 2;
    let mut b = BigUint::from(4u32);
    print!("{a} * {b}\t= ");
    product(a, &mut b);
    assert_eq!(b, BigUint::from(8u32));
    println!("{b}");

    let a = 2;
    let mut b = BigUint::from(4u32);
    print!("{a}^{b}\t= ");
    power(a, &mut b);
    assert_eq!(b, BigUint::from(16u32));
    println!("{b}");

    let a = 2;
    let mut b = BigUint::from(4u32);
    print!("{a}↑↑{b}\t= ");
    tetrate(a, &mut b);
    assert_eq!(b, BigUint::from(65536u32));
    println!("{b}");
}
