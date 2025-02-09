use num_bigint::BigInt;
use num_traits::identities::Zero;

fn two_tetration(x: BigInt) -> BigInt {
    if x.is_zero() {
        return BigInt::from(1);
    }
    two_raised_to_the_power(two_tetration(x - 1))
}

fn two_raised_to_the_power(x: BigInt) -> BigInt {
    if x.is_zero() {
        return BigInt::from(1);
    }

    two_times(two_raised_to_the_power(x - 1))
}

fn two_times(mut x: BigInt) -> BigInt {
    let mut result = BigInt::from(0);
    while !x.is_zero() {
        x -= 1;
        result = two_plus(result);
    }
    result
}

fn two_plus(x: BigInt) -> BigInt {
    let mut result = x;
    for _ in 0..2 {
        result += 1;
    }
    result
}

fn main() -> Result<(), String> {
    // 2 + 5 = 7
    let x = BigInt::from(5);
    let result = two_plus(x.clone());
    println!("2+{x} = {result}");

    // 2  x 2 = 4
    let x = BigInt::from(2);
    let result = two_times(x.clone());
    println!("2×{x} = {result}");

    // 2^3 = 8
    let x = BigInt::from(3);
    let result = two_raised_to_the_power(x.clone());
    println!("2^{x} = {result}");

    // 2^^3 = 16
    let x = BigInt::from(4);
    let result = two_tetration(x.clone());
    println!("2↑↑{x} = {result}");

    Ok(())
}
