# burn-npu

> **Early development.** The Apple backend is tested and working, but does not
> appear to reach the Neural Engine —
> [see the evidence](#does-this-actually-use-the-neural-engine). Intel has been
> tested on an Intel NPU. Qualcomm has not been run on its target hardware.
> Contributions welcome.

NPU backend for [Burn](https://burn.dev). A drop-in replacement for `burn-wgpu` or `burn-flex` that targets hardware Neural Processing Units through each vendor's own API.

```rust
use burn::tensor::Tensor;
use burn_npu::{NpuBurnBackend, NpuBurnDevice};

type B = NpuBurnBackend;  // swap this line — that's it

let device = NpuBurnDevice::Default;
let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
let b = Tensor::<B, 2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
let c = a.matmul(b);
```

Supported inference models keep the standard Burn tensor/module interface when changing the backend type. See the platform notes for accelerated operations and CPU fallback.

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
| **burn-npu (apple)** | **30.5 ms** | **32.6 tok/s** |
| burn-flex (CPU) | 66 ms | 15.1 tok/s |
| burn-wgpu (Metal GPU) | 95 ms | 10.5 tok/s |

> Measured on burn 0.21, Apple M2 Pro, machine under moderate load; each row was
> stable across runs. The burn-wgpu figure is much worse than the 37 ms recorded
> against burn 0.20 and that difference is **not** explained -- it may be a wgpu
> regression or it may be the load. Treat it as unverified.
>
> Note the label: this row is not "Apple NPU". See
> [Does this actually use the Neural Engine?](#does-this-actually-use-the-neural-engine)

```bash
cargo run --release --example bench --features apple
```

## Installation

```toml
[dependencies]
burn-npu = { version = "0.4", features = ["apple"] }
```

| Feature | Hardware | Status | Requires |
|---|---|---|---|
| `apple` | Apple Silicon (M1-M6) via MLTensor | **tested, working** -- but see the [Neural Engine caveat](#does-this-actually-use-the-neural-engine) | macOS 15+, Xcode |
| `intel` | Intel Core Ultra NPU | **tested on Intel NPU** | OpenVINO runtime |
| `qualcomm` | Qualcomm Hexagon (Snapdragon) | implemented, **never run on hardware** | QAIRT/QNN SDK at build time (`QNN_SDK_ROOT`) |

Enable one feature at a time. Without any feature, falls back to burn-flex (CPU).

Intel environment variables: `BURN_NPU_TRACE=1` logs NPU/GPU/CPU dispatch,
`BURN_NPU_DISABLE=1` forces CPU fallback, `BURN_NPU_CONSTANT_WEIGHTS=1` compiles
stable FP32 weights into NPU models for repeated inference.

## How It Works

Storage and dispatch differ by platform. Intel uses shared host storage for Burn tensors, with copies at the OpenVINO boundary. Compiled models and inference requests are reused; the optional constant-weight path embeds reusable weights in compiled models.

| Platform | Tensor type | Dispatch |
|---|---|---|
| Apple | MLTensor handle | GPU / CPU via Core ML (**not** the ANE -- see below) |
| Intel | Shared burn-flex storage and views | FP32 matmul via OpenVINO NPU / GPU / CPU; supported attention graphs on NPU |
| Qualcomm | QNN graph on Hexagon (HTP) | matmul on NPU, everything else CPU |

On Apple, 37 float ops run natively in MLTensor rather than round-tripping to the CPU delegate. On Intel, FP32 matmul and supported attention graphs run on the NPU via OpenVINO; everything else delegates to burn-flex.

## Does this actually use the Neural Engine?

**On Apple, the honest answer is: probably not, today.**

The Apple backend is built on `MLTensor`, whose documentation implies it
dispatches to ANE, GPU or CPU automatically. Measured on an M2 Pro, it does not
appear to reach the ANE. The same 512x768x768 matmul, run under each Core ML
compute policy via `withMLTensorComputePolicy`:

| policy | eager `MLTensor` | compiled Core ML model |
|---|---|---|
| `.cpuOnly` | 0.30 ms | 1.58 ms |
| `.cpuAndNeuralEngine` | 0.32 ms | **0.90 ms** |
| `.cpuAndGPU` | 0.65 ms | 1.71 ms |
| `.all` | 0.59 ms | 0.99 ms |

Allowing the ANE makes no difference to eager `MLTensor` ops. For a *compiled*
Core ML model doing comparable work it is 1.77x faster than CPU-only, and
`MLComputePlan` confirms the assignment directly -- every layer reports
`MLNeuralEngineComputeDevice`. Other developers have reported the same thing for
`MLTensor` using Instruments
([Apple Developer Forums](https://developer.apple.com/forums/thread/775589)).

So what you get on Apple today is a fast **CPU** backend: `MLTensor` routing
through Accelerate/AMX, roughly 2x quicker than burn-flex and 3x quicker than
burn-wgpu on the GPT-2 benchmark above. That is a genuinely useful result. It is
just not the Neural Engine.

The evidence that compiled models reach the ANE is direct. The evidence that
eager `MLTensor` never does is strong but indirect -- no speedup when the ANE is
enabled, plus the forum reports. Confirming it properly needs:

```bash
sudo powermetrics --samplers ane_power -i 1000 -n 5
# ...while `cargo run --release --example bench --features apple` runs
```

Reaching the ANE means giving Core ML a compiled graph instead of one eager op
at a time. That is what burn 0.22's graph-capture backend, or Apple's newer
Core AI framework, would make possible. It is the main open item for this
project -- see [Contributing](#contributing).

The Intel path targets OpenVINO devices explicitly and has been confirmed on NPU hardware.

## Background

This project was motivated by [this discussion](https://github.com/tracel-ai/burn/discussions/4245) in the Burn repo, where NPU support was considered difficult because NPUs "are often not programmable chips" with no common API across vendors. That's true — there is no universal NPU API. burn-npu takes a different approach: per-vendor integration using each vendor's own tensor/inference API (MLTensor, OpenVINO, QNN), wrapped behind Burn's `Backend` trait.

## Contributing

This is an early project. Help is welcome:

- **Reaching the ANE on Apple** — the largest open item. Eager `MLTensor` ops do
  not appear to run on the Neural Engine; compiled Core ML models demonstrably
  do. Closing that gap means batching burn's eager ops into a compiled graph,
  via burn 0.22's graph capture or Apple's Core AI framework. See
  [Does this actually use the Neural Engine?](#does-this-actually-use-the-neural-engine)
- **Confirming the ANE finding** — if you can run `powermetrics` against the
  benchmark and report ANE power draw, that settles the one piece of the above
  that rests on indirect evidence
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
