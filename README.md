# burn-npu

> **Early development.** Apple backend tested and working. Intel and Qualcomm are
> implemented but have never been run on their target hardware. Contributions welcome.

NPU backend for [Burn](https://burn.dev). Drop-in replacement for `burn-wgpu` or `burn-flex` that runs inference on hardware Neural Processing Units.

```rust
use burn::tensor::Tensor;
use burn_npu::{NpuBurnBackend, NpuBurnDevice};

type B = NpuBurnBackend;  // swap this line — that's it

let device = NpuBurnDevice::Default;
let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
let b = Tensor::<B, 2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
let c = a.matmul(b);
```

Any Burn model works. No code changes needed. Just change the backend type.

## What NPUs can and can't do

NPUs are **inference-only accelerators**. They are not programmable like GPUs — you can't write custom kernels. Each vendor provides their own API (Apple MLTensor, Intel OpenVINO, Qualcomm QNN), and there is no universal standard like Vulkan or WebGPU.

burn-npu works by wrapping each vendor's API behind Burn's `Backend` trait. This means:

- **Works:** matrix multiply, elementwise ops, reductions, attention, feedforward — the ops that make up transformer inference
- **Doesn't work:** training (no autograd), custom kernels, ops the vendor API doesn't support
- **Fragmented:** each platform is a separate implementation, not a single portable backend

For training, use `burn-wgpu` (Metal/Vulkan GPU) or `burn-cuda` (NVIDIA).

## Benchmark

GPT-2 124M forward pass, seq=32, FP32, Apple M2 Pro.

| Backend | Latency | Throughput |
|---|---|---|
| **burn-npu (Apple NPU)** | **29 ms** | **34.8 tok/s** |
| burn-wgpu (Metal GPU) | 37 ms | 27.1 tok/s |
| burn-ndarray (CPU) | 107 ms | 9.4 tok/s |

> These numbers were measured against burn 0.20 and have **not** been re-run
> since the burn 0.21 / burn-flex migration. The CPU baseline is now burn-flex,
> which is SIMD-accelerated where burn-ndarray was not, so expect the CPU row in
> particular to have moved.

```bash
cargo run --release --example bench --features apple
```

## Installation

```toml
[dependencies]
burn-npu = { version = "0.3", features = ["apple"] }
```

| Feature | Hardware | Status | Requires |
|---|---|---|---|
| `apple` | Apple Neural Engine (M1/M2/M3/M4) | **tested, working** | macOS 15+, Xcode |
| `intel` | Intel Core Ultra NPU | implemented, **never run on hardware** | OpenVINO runtime |
| `qualcomm` | Qualcomm Hexagon (Snapdragon) | implemented, **never run on hardware** | QAIRT/QNN SDK at build time (`QNN_SDK_ROOT`) |

Enable one feature at a time. Without any feature, falls back to burn-flex (CPU).

## How It Works

Each platform has a native tensor type that stays on the NPU between operations. No data copies between ops — only `into_data()` reads back to CPU.

| Platform | Tensor type | Dispatch |
|---|---|---|
| Apple | MLTensor handle | ANE / GPU / CPU via Core ML |
| Intel | OpenVINO tensor | NPU / GPU / CPU via OpenVINO |
| Qualcomm | QNN graph on Hexagon (HTP) | matmul on NPU, everything else CPU |

On Apple, 37 float ops run natively on the NPU. On Intel, matmul is NPU-accelerated via OpenVINO. Remaining ops and int/bool tensors delegate to burn-flex.

## Background

This project was motivated by [this discussion](https://github.com/tracel-ai/burn/discussions/4245) in the Burn repo, where NPU support was considered difficult because NPUs "are often not programmable chips" with no common API across vendors. That's true — there is no universal NPU API. burn-npu takes a different approach: per-vendor integration using each vendor's own tensor/inference API (MLTensor, OpenVINO, QNN), wrapped behind Burn's `Backend` trait.

## Contributing

This is an early project. Help is welcome:

- **Intel hardware testing** — run `cargo test --features intel` on a Core Ultra machine and open an issue with results
- **Qualcomm hardware testing** — the QNN integration is written but has never
  been compiled against real SDK headers or run on a device. If you have a
  Snapdragon X Elite or Rubik Pi 3, building with `QNN_SDK_ROOT` set is the
  single most useful thing you could contribute. See the notes in
  `src/backends/qualcomm/qnn.rs` for what is most likely to need fixing first.
- **More NPU-accelerated ops** — move remaining float ops from burn-flex delegation to native NPU execution on Intel/Qualcomm
- **Bug reports** — if a Burn model doesn't work on `NpuBurnBackend`, open an issue

## License

MIT OR Apache-2.0
