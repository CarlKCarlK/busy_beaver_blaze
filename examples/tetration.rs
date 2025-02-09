use num_bigint::BigInt;
use num_traits::identities::Zero;

fn two_tetration(x: BigInt) -> BigInt {
    if x.is_zero() {
        return BigInt::from(1);
    }
    let x_less_one: BigInt = x - 1;
    if x_less_one.is_zero() {
        return BigInt::from(2);
    }

    let less_one = two_tetration(x_less_one);
    let mut result = BigInt::from(0);
    let mut x = less_one.clone();
    while !x.is_zero() {
        x -= 1;
        let mut y = less_one.clone();
        while !y.is_zero() {
            y -= 1;
            result += 1;
        }
    }
    result
}

// fn two_raised_to_the_power(x: BigInt, steps: &mut BigInt) {
//     if x == 0 {
//         *steps += 1;
//         return;
//     }
//     for _ in 0..2 {
//         two_raised_to_the_power(x - 1, steps);
//     }
// }
// fn two_times(x: BigInt, steps: &mut BigInt) {
//     for _ in 0..2 {
//         for _ in 0..x {
//             *steps += 1;
//         }
//     }
//

fn main() -> Result<(), String> {
    // // 2  x 2 = 4
    // let mut steps = 0;
    // let x = 3;
    // two_times(x, &mut steps);
    // println!("2 x {x} = {steps}");

    // // 2^3 = 8
    // let mut steps = 0;
    // let x = 3;
    // two_raised_to_the_power(x, &mut steps);
    // println!("2^{x} = {steps}");

    // 2^^3 = 16
    let x = BigInt::from(6);
    let result = two_tetration(x.clone());
    println!("2^^{x} = {result}");

    Ok(())
}
