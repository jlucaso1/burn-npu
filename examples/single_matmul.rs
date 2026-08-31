//! One 512x768x768 matmul through the backend, to compare against the same
//! operation issued directly to MLTensor from Swift.
use burn::tensor::Tensor;
use burn_npu::{NpuBurnBackend, NpuBurnDevice};
use std::time::Instant;

type B = NpuBurnBackend;

fn main() {
    let device = NpuBurnDevice::Default;
    let lhs = Tensor::<B, 2>::zeros([512, 768], &device) + 0.01;
    let rhs = Tensor::<B, 2>::zeros([768, 768], &device) + 0.01;

    // warm up
    let _ = lhs.clone().matmul(rhs.clone()).into_data();

    let iters = 20;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = lhs.clone().matmul(rhs.clone()).into_data();
    }
    println!(
        "burn-npu matmul+readback: {:?} each",
        start.elapsed() / iters
    );

    // Isolate the readback: a tensor with no pending computation.
    let plain = Tensor::<B, 2>::zeros([512, 768], &device) + 0.01;
    let _ = plain.clone().into_data();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = plain.clone().into_data();
    }
    println!(
        "readback only (512x768): {:?} each",
        start.elapsed() / iters
    );

    // Same, but without the readback in the loop.
    let start = Instant::now();
    let mut acc = lhs.clone();
    for _ in 0..iters {
        acc = acc.matmul(rhs.clone());
    }
    let build = start.elapsed();
    let start2 = Instant::now();
    let _ = acc.into_data();
    println!(
        "graph build: {:?} total, materialize {} chained: {:?}",
        build,
        iters,
        start2.elapsed()
    );
}
