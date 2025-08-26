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
    fn count_down(&mut self) -> CountDownMutRef<'_, Self>
    where
        Self: Sized + Decrementable;
}

impl<T: Decrementable> CountDownMut for T {
    #[inline]
    fn count_down(&mut self) -> CountDownMutRef<'_, T> {
        CountDownMutRef { current: self }
    }
}

fn increment(increment_acc: &mut BigUint) {
    *increment_acc += 1u32;
}

fn add(a: u32, add_acc: &mut BigUint) {
    for _ in 0..a {
        // We in-line `increment` manually, to keep our work explicit.
        *add_acc += 1u32;
    }
}

fn multiply(a: u32, multiply_acc: &mut BigUint) {
    let mut add_acc = BigUint::ZERO;
    for () in multiply_acc.count_down() {
        for _ in 0..a {
            add_acc += 1u32;
        }
    }
    *multiply_acc = add_acc;
}

fn exponentiate(a: u32, exponentiate_acc: &mut BigUint) {
    assert!(
        a > 0 || *exponentiate_acc != BigUint::ZERO,
        "0^0 is undefined"
    );

    let mut multiply_acc = BigUint::ZERO;
    multiply_acc += 1u32;
    for () in exponentiate_acc.count_down() {
        let mut add_acc = BigUint::ZERO;
        for () in multiply_acc.count_down() {
            for _ in 0..a {
                add_acc += 1u32;
            }
        }
        multiply_acc = add_acc;
    }
    *exponentiate_acc = multiply_acc;
}

fn tetrate(a: u32, tetrate_acc: &mut BigUint) {
    assert!(a > 0, "we don’t define 0↑↑b");

    let mut exponentiate_acc = BigUint::ZERO;
    exponentiate_acc += 1u32;
    for () in tetrate_acc.count_down() {
        let mut multiply_acc = BigUint::ZERO;
        multiply_acc += 1u32;
        for () in exponentiate_acc.count_down() {
            let mut add_acc = BigUint::ZERO;
            for () in multiply_acc.count_down() {
                for _ in 0..a {
                    add_acc += 1u32;
                }
            }
            multiply_acc = add_acc;
        }
        exponentiate_acc = multiply_acc;
    }
    *tetrate_acc = exponentiate_acc;
}
#[allow(clippy::shadow_unrelated)]
fn main() {
    let mut b = BigUint::from(4u32);
    print!("++{b}\t= ");
    increment(&mut b);
    assert_eq!(b, BigUint::from(5u32));
    println!("{b}");

    let a = 2;
    let mut b = BigUint::from(4u32);
    print!("{a} + {b}\t= ");
    add(a, &mut b);
    assert_eq!(b, BigUint::from(6u32));
    println!("{b}");

    let a = 2;
    let mut b = BigUint::from(4u32);
    print!("{a} * {b}\t= ");
    multiply(a, &mut b);
    assert_eq!(b, BigUint::from(8u32));
    println!("{b}");

    let a = 2;
    let mut b = BigUint::from(4u32);
    print!("{a}^{b}\t= ");
    exponentiate(a, &mut b);
    assert_eq!(b, BigUint::from(16u32));
    println!("{b}");

    let a = 2;
    let mut b = BigUint::from(4u32);
    print!("{a}↑↑{b}\t= ");
    tetrate(a, &mut b);
    assert_eq!(b, BigUint::from(65536u32));
    println!("{b}");

    // let a = 10;
    // let mut b = BigUint::from(15u32);
    // print!("{a}↑↑{b}\t= ");
    // tetrate(a, &mut b);
    // println!("{b}");
}
