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

fn sum(a: u32, sum_acc: &mut BigUint) {
    for _ in 0..a {
        *sum_acc += 1u32;
    }
}

fn product(a: u32, product_acc: &mut BigUint) {
    let mut sum_acc = BigUint::ZERO;
    for () in product_acc.count_down() {
        for _ in 0..a {
            sum_acc += 1u32;
        }
    }
    *product_acc = sum_acc;
}

fn power(a: u32, power_acc: &mut BigUint) {
    assert!(
        a > 0 || *power_acc > BigUint::ZERO,
        "a must be greater than 0 or power greater than 0"
    );

    let mut product_acc = BigUint::ZERO;
    product_acc += 1u32;
    for () in power_acc.count_down() {
        let mut sum_acc = BigUint::ZERO;
        for () in product_acc.count_down() {
            for _ in 0..a {
                sum_acc += 1u32;
            }
        }
        product_acc = sum_acc;
    }
    *power_acc = product_acc;
}

fn tetration(a: u32, tetration_acc: &mut BigUint) {
    assert!(a > 0, "a must be greater than 0");

    let mut power_acc = BigUint::ZERO;
    power_acc += 1u32;
    for () in tetration_acc.count_down() {
        let mut product_acc = BigUint::ZERO;
        product_acc += 1u32;
        for () in power_acc.count_down() {
            let mut sum_acc = BigUint::ZERO;
            for () in product_acc.count_down() {
                for _ in 0..a {
                    sum_acc += 1u32;
                }
            }
            product_acc = sum_acc;
        }
        power_acc = product_acc;
    }
    *tetration_acc = power_acc;
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
    tetration(a, &mut b);
    assert_eq!(b, BigUint::from(65536u32));
    println!("{b}");
}
