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

    // Attention-shaped masking: on the apple backend this should stay on the
    // NPU rather than materialising the value tensor's pending graph.
    let big = Tensor::<B, 3>::zeros([8, 64, 64], &device) + 1.0;
    let mask = big.clone().greater_elem(0.5);
    let start = Instant::now();
    for _ in 0..n {
        let _ = big.clone().mask_fill(mask.clone(), -1.0e9);
    }
    let elapsed = start.elapsed();
    println!(
        "{n} mask_fill 8x64x64: {:?} total, {:?} each",
        elapsed,
        elapsed / n
    );

    // fp16 vs fp32 matmul at GPT-2-ish dimensions. fp16 is the ANE's native
    // format, so this is the precision the hardware actually wants.
    use burn::tensor::DType;
    let iters = 50u32;
    for dtype in [DType::F32, DType::F16] {
        let lhs = Tensor::<B, 2>::zeros([512, 768], &device).cast(dtype) + 0.5;
        let rhs = Tensor::<B, 2>::zeros([768, 768], &device).cast(dtype) + 0.5;
        // Warm up: the first call pays graph setup.
        let _ = lhs.clone().matmul(rhs.clone()).into_data();
        let start = Instant::now();
        for _ in 0..iters {
            let _ = lhs.clone().matmul(rhs.clone()).into_data();
        }
        let elapsed = start.elapsed();
        println!("matmul 512x768x768 {:?}: {:?} each", dtype, elapsed / iters);
    }
}
