//! Times NPU->CPU readbacks, which is what every comparison op and every
//! `into_data` pays for. Run with: cargo run --release --example readback_bench --features apple
use burn::tensor::Tensor;
use burn_npu::{NpuBurnBackend, NpuBurnDevice};
use std::time::Instant;

type B = NpuBurnBackend;

fn main() {
    let device = NpuBurnDevice::Default;
    let n = 200;

    // Small tensor: dominated by per-readback overhead, not data volume.
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let start = Instant::now();
    for _ in 0..n {
        let _: Vec<f32> = a.clone().into_data().to_vec().unwrap();
    }
    let elapsed = start.elapsed();
    println!(
        "{n} small readbacks: {:?} total, {:?} each",
        elapsed,
        elapsed / n
    );

    // Comparison ops force a readback internally to build the bool tensor.
    let x = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let y = Tensor::<B, 2>::from_floats([[2.0, 2.0], [2.0, 2.0]], &device);
    let start = Instant::now();
    for _ in 0..n {
        let _ = x.clone().greater(y.clone());
    }
    let elapsed = start.elapsed();
    println!(
        "{n} greater() ops:    {:?} total, {:?} each",
        elapsed,
        elapsed / n
    );
}
