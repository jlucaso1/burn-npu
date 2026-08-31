# Changelog

All notable changes to this project are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); while below 1.0, a
breaking change bumps the minor version.

## [0.4.0] - 2026-08-31

Catches the project up to burn 0.21, implements the Qualcomm backend that was
previously only described, makes the Apple path substantially faster, and
corrects the README's claim that the Apple backend runs on the Neural Engine.

### Breaking

- **`IntElem` is now `i32`, was `i64`.** This matches burn-flex and burn's
  ecosystem default, so delegated int operations no longer need a dtype
  conversion on every call. Code reading `argmax` output or int tensor data as
  `Vec<i64>` must read `Vec<i32>` instead.
- **Minimum burn version is 0.21.** The `Backend` trait split its associated
  types into `BackendTypes`, tensor-creating and comparison operations take an
  explicit output dtype, scalar operations take `Scalar` rather than raw
  `f32`/`i64`, and `Shape` fields are private. See the burn 0.21 release notes.
- **`burn-ndarray` is replaced by `burn-flex`** as the CPU delegate and as the
  no-feature fallback. Upstream deprecated burn-ndarray in the 0.22 cycle.
- MSRV is now 1.89, inherited from burn 0.21.

### Added

- **Qualcomm Hexagon NPU dispatch via QNN.** `build.rs` generates the QNN ABI
  with bindgen from the SDK's own headers when `QNN_SDK_ROOT` is set; the
  runtime opens `libQnnHtp.so`/`QnnHtp.dll` with `libloading`, selects a
  provider matching the generated bindings, and caches finalized matmul graphs
  by shape. Without the SDK the feature still builds and runs on CPU.
  **Not yet validated on hardware.**
- **Qualcomm hardware detection**, which was missing entirely — `NpuVendor::Qualcomm`
  existed but nothing ever constructed it. Probes the device-tree model and ARM
  implementer ID on Linux, and the HTP backend DLL on Windows.
- **fp16 support on Apple.** f16 tensors stay in f16 through the backend rather
  than being silently widened to f32, `float_cast` actually casts, and
  `dtype_usage` reports what MLTensor can really do (f32 and f16; not bf16 or
  f64) instead of delegating wholesale to the CPU backend.
- **CI**: fmt, clippy `-D warnings`, a feature matrix on Linux and Windows, and
  the test suite on macOS 15. Includes a weekly scheduled run — `Cargo.lock` is
  not committed, so this is what catches upstream breakage early.
- `examples/readback_bench.rs` and `examples/single_matmul.rs`, the harnesses
  behind the performance figures below.

### Fixed

- **The crate did not compile.** The module split in `bd09e3c` dropped an import
  the apple `FloatTensorOps` block depended on.
- Apple chip detection only knew M1–M4 and reported anything newer as an M1 at
  11 TOPS. M5 and M6 are recognised; since Apple has not published a Neural
  Engine figure for either, they report the M4 number as an explicit floor.
- Lint errors on the `intel` and `qualcomm` paths that no job had ever run.

### Performance (Apple, M2 Pro, release build)

- **Readback no longer goes through `MLShapedArray.scalars`**, which was ~420x
  slower than reading the backing buffer for identical data. A 512x768 readback
  went from 24.6 ms to 73 us; matmul plus readback from 24.1 ms to 372 us.
- **Masking stays on the NPU.** `npu_mask_fill`/`npu_mask_where` existed in the
  Swift shim but were never called; masking round-tripped through the CPU,
  forcing MLTensor's lazy graph to materialise. `mask_fill` on an 8x64x64
  tensor went from ~2.0 ms to ~60 us.
- **No more thread per readback.** Each readback spawned an OS thread and spun a
  RunLoop; both now share one semaphore-based bridge. Small readbacks went from
  56-70 us to 14-21 us, `greater()` from 83-127 us to 32-44 us.
- The Qualcomm CPU matmul fallback accumulates along output rows instead of
  recomputing dot products, and skips zero multipliers.

### Documentation

- **The Apple backend does not appear to reach the Neural Engine**, and the
  README no longer says it does. Enabling the ANE via `withMLTensorComputePolicy`
  makes no measurable difference to eager `MLTensor` operations, while a
  compiled Core ML model doing comparable work is 1.77x faster than CPU-only
  with every layer reported on `MLNeuralEngineComputeDevice`. The new
  "Does this actually use the Neural Engine?" section carries the measurements.
  What you get on Apple today is a fast CPU backend via Accelerate/AMX.
- Intel and Qualcomm are now described as "never run on hardware" rather than
  "needs hardware validation".
- The benchmark is re-measured on burn 0.21. The burn-wgpu row is markedly worse
  than the 0.20 figure; that is flagged as unexplained rather than presented as
  a win.

### Known limitations

- Reaching the ANE requires giving Core ML a compiled graph rather than one
  eager operation at a time — the main open item, and a fit for burn 0.22's
  graph capture or Apple's Core AI framework.
- The Intel and Qualcomm backends have never been run on their target hardware.
- The evidence that eager `MLTensor` never uses the ANE is strong but indirect;
  see the README for the `powermetrics` command that would settle it.

## [0.3.0] - 2026-03-18

Initial public release. Apple backend via MLTensor; Intel via OpenVINO;
Qualcomm scaffolding.
