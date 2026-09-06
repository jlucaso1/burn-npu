//! Intel NPU backend via OpenVINO.
//!
//! Burn's Intel primitive shares Flex storage and views; supported FP32 matmul
//! and attention graphs dispatch through OpenVINO. Other operations/dtypes use
//! Flex. `IntelFloatTensor` remains as a compatible Vec-based low-level API.
//! Matmul tries NPU > GPU > CPU, and the compiled attention path targets NPU.
//!
//! The `openvino` crate is used with `runtime-linking`, so the code compiles
//! everywhere even if the OpenVINO runtime is not installed. NPU acceleration
//! is opportunistic: if the runtime or device is unavailable, we fall back to
//! a CPU implementation transparently.

use burn_tensor::{DType, Shape};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

mod attention;
mod cache;
pub(crate) mod diagnostics;
mod range;
pub use diagnostics::ExecutionStats;
use diagnostics::Target;

/// Read process-wide dispatch counters and the last native runtime error.
pub fn execution_stats() -> ExecutionStats {
    diagnostics::snapshot()
}
mod constant_matmul;
use cache::BuildCache;
pub use cache::CacheStats;

/// OpenVINO acceleration was unavailable or conservatively declined.
///
/// This includes unsupported shapes/dtypes, numeric range or cache admission,
/// and native runtime failures. Low-level functions return this signal; the
/// Burn backend then uses Flex. Inspect [`execution_stats`] for native failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenVinoUnavailable;

impl std::fmt::Display for OpenVinoUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OpenVINO acceleration unavailable")
    }
}

impl std::error::Error for OpenVinoUnavailable {}

type MatmulCache = BuildCache<(usize, usize, usize, usize, bool), Mutex<OvCompiledMatmul>>;
static OV_CACHE: LazyLock<MatmulCache> =
    LazyLock::new(|| BuildCache::new(32, 128 * 1024 * 1024, Duration::from_secs(30)));

/// Snapshot of cache accounting. Driver/compiler memory and active evicted
/// requests are additional to these estimates.
#[derive(Clone, Copy, Debug)]
pub struct IntelCacheStats {
    pub matmul: CacheStats,
    pub constant_weights: CacheStats,
    pub attention: CacheStats,
}

pub fn cache_stats() -> IntelCacheStats {
    IntelCacheStats {
        matmul: OV_CACHE.stats(),
        constant_weights: constant_matmul::stats(),
        attention: attention::stats(),
    }
}

/// Release cached models without cancelling active calls. A later call rebuilds
/// its model. Call after workers finish when reclaiming memory between models.
pub fn clear_caches() -> Result<(), OpenVinoUnavailable> {
    // Runtime-linked C function tables are thread-local, including destructors.
    load_openvino()?;
    OV_CACHE.clear();
    constant_matmul::clear();
    attention::clear();
    range::clear();
    Ok(())
}

pub(super) fn elements(shape: &[usize]) -> Result<usize, OpenVinoUnavailable> {
    shape.iter().try_fold(1usize, |count, &dim| {
        if dim > i64::MAX as usize {
            return Err(OpenVinoUnavailable);
        }
        count.checked_mul(dim).ok_or(OpenVinoUnavailable)
    })
}

struct OvCompiledMatmul {
    // Inference and buffer writes must hold this entry's mutex together. A request
    // cannot execute concurrently with another caller updating its inputs.
    request: openvino::InferRequest,
    lhs: openvino::Tensor,
    rhs: openvino::Tensor,
    _compiled: openvino::CompiledModel,
    target: Target,
    failed: bool,
}

/// Load the OpenVINO shared library (required for runtime-linking feature).
fn ensure_openvino_loaded() -> Result<(), OpenVinoUnavailable> {
    if std::env::var("BURN_NPU_DISABLE").as_deref() == Ok("1") {
        return Err(OpenVinoUnavailable);
    }
    load_openvino()
}

pub(super) fn load_openvino() -> Result<(), OpenVinoUnavailable> {
    thread_local! { static READY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
    static FAILED: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    if READY.get() {
        return Ok(());
    }
    let mut failed = FAILED.lock().map_err(|_| OpenVinoUnavailable)?;
    if failed.is_some_and(|at| at.elapsed() < Duration::from_secs(1)) {
        return Err(OpenVinoUnavailable);
    }
    match openvino_sys::library::load() {
        Ok(()) => {
            READY.set(true);
            *failed = None;
            Ok(())
        }
        Err(error) => {
            *failed = Some(std::time::Instant::now());
            drop(failed);
            Err(diagnostics::failure("OpenVINO library loading", error))
        }
    }
}

// ---------------------------------------------------------------------------
// IntelFloatTensor
// ---------------------------------------------------------------------------

/// A float tensor backed by a `Vec<f32>` with shape metadata.
///
/// Compatibility type for the low-level matmul API. Burn's float primitive
/// uses shared `FlexTensor` storage instead.
#[derive(Debug, Clone)]
pub struct IntelFloatTensor {
    /// Flat f32 data in row-major order.
    pub data: Vec<f32>,
    /// Shape dimensions.
    pub shape: Vec<usize>,
}

impl IntelFloatTensor {
    /// Create a tensor from flat data and shape. Panics if the shape is invalid.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            elements(&shape).expect("tensor shape overflows"),
            "IntelFloatTensor::new: data.len() != product(shape)"
        );
        Self { data, shape }
    }
}

impl burn_tensor::TensorMetadata for IntelFloatTensor {
    fn dtype(&self) -> DType {
        DType::F32
    }

    fn shape(&self) -> Shape {
        Shape::from(self.shape.clone())
    }
}

// ---------------------------------------------------------------------------
// OpenVINO matmul (best-effort NPU acceleration)
// ---------------------------------------------------------------------------

/// Attempt to perform matmul via OpenVINO, targeting NPU > GPU > CPU.
///
/// Dispatch all matching batch dimensions in one OpenVINO MatMul. Reuse the
/// compiled model, synchronous request and input buffers between calls.
/// Returns [`OpenVinoUnavailable`] if the OpenVINO runtime is unavailable.
pub fn openvino_matmul(
    lhs: &IntelFloatTensor,
    rhs: &IntelFloatTensor,
) -> Result<IntelFloatTensor, OpenVinoUnavailable> {
    openvino_matmul_slices(&lhs.data, &lhs.shape, &rhs.data, &rhs.shape, false, None)
}

/// Strict hardware probe: compile on NPU only, without OpenVINO GPU/CPU fallback.
pub fn openvino_matmul_npu(
    lhs: &IntelFloatTensor,
    rhs: &IntelFloatTensor,
) -> Result<IntelFloatTensor, OpenVinoUnavailable> {
    openvino_matmul_slices(&lhs.data, &lhs.shape, &rhs.data, &rhs.shape, true, None)
}

/// Burn's Intel primitive shares Flex storage and keeps views lazy until the
/// OpenVINO boundary. Non-FP32 dtypes remain in Flex rather than being mislabeled.
pub fn openvino_matmul_flex(
    lhs: &burn_flex::FlexTensor,
    rhs: &burn_flex::FlexTensor,
) -> Result<burn_flex::FlexTensor, OpenVinoUnavailable> {
    use burn_tensor::TensorMetadata;
    if std::env::var("BURN_NPU_DISABLE").as_deref() == Ok("1") {
        return Err(OpenVinoUnavailable);
    }
    if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 {
        return Err(OpenVinoUnavailable);
    }
    let shape = lhs.shape();
    let k = *shape.last().ok_or(OpenVinoUnavailable)?;
    let bounds = (range::max_abs(lhs.storage::<f32>())?, range::rhs_max(rhs)?);
    range::safe_matmul(k, bounds.0, bounds.1)?;
    if std::env::var("BURN_NPU_CONSTANT_WEIGHTS").as_deref() == Ok("1") {
        ensure_openvino_loaded()?;
        if let Ok(output) = constant_matmul::matmul(lhs, rhs) {
            return Ok(output);
        }
    }
    let lhs = lhs.to_contiguous();
    let rhs = rhs.to_contiguous();
    let output = openvino_matmul_slices(
        lhs.storage::<f32>(),
        &lhs.shape(),
        rhs.storage::<f32>(),
        &rhs.shape(),
        false,
        Some(bounds),
    )?;
    Ok(burn_flex::FlexTensor::from_data(
        burn_tensor::TensorData::new(output.data, output.shape),
    ))
}

fn openvino_matmul_slices(
    lhs_data: &[f32],
    lhs_shape: &[usize],
    rhs_data: &[f32],
    rhs_shape: &[usize],
    npu_only: bool,
    bounds: Option<(f64, f64)>,
) -> Result<IntelFloatTensor, OpenVinoUnavailable> {
    let lhs_ndim = lhs_shape.len();
    let rhs_ndim = rhs_shape.len();
    if lhs_ndim < 2 || rhs_ndim < 2 {
        return Err(OpenVinoUnavailable);
    }

    let m = lhs_shape[lhs_ndim - 2];
    let k = lhs_shape[lhs_ndim - 1];
    let n = rhs_shape[rhs_ndim - 1];

    if rhs_shape[rhs_ndim - 2] != k || m == 0 || k == 0 || n == 0 {
        return Err(OpenVinoUnavailable);
    }

    if elements(lhs_shape)? != lhs_data.len() || elements(rhs_shape)? != rhs_data.len() {
        return Err(OpenVinoUnavailable);
    }
    let (a, b) = match bounds {
        Some(bounds) => bounds,
        None => (range::max_abs(lhs_data)?, range::max_abs(rhs_data)?),
    };
    range::safe_matmul(k, a, b)?;

    // Compute batch dimensions: everything before the last 2 dims
    let lhs_batch: Vec<usize> = lhs_shape[..lhs_ndim - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs_shape[..rhs_ndim - 2].to_vec();
    let mut out_shape = vec![1; rhs_ndim.saturating_sub(lhs_ndim)];
    out_shape.extend(lhs_batch.iter().copied());
    out_shape.extend([m, n]);
    // A shared RHS can multiply all contiguous LHS rows in one 2D graph.
    let (batch_size, m) = if rhs_batch.iter().all(|&d| d == 1) {
        (1, elements(&lhs_shape[..lhs_ndim - 1])?)
    } else if lhs_batch == rhs_batch {
        (elements(&lhs_batch)?, m)
    } else {
        return Err(OpenVinoUnavailable);
    };
    if batch_size == 0 {
        return Err(OpenVinoUnavailable);
    }
    // Skip OpenVINO overhead for small matmuls.
    if m.checked_mul(k)
        .and_then(|v| v.checked_mul(n))
        .ok_or(OpenVinoUnavailable)?
        < 4096
    {
        return Err(OpenVinoUnavailable);
    }

    let lhs_stride = m.checked_mul(k).ok_or(OpenVinoUnavailable)?;
    let rhs_stride = k.checked_mul(n).ok_or(OpenVinoUnavailable)?;
    let out_stride = m.checked_mul(n).ok_or(OpenVinoUnavailable)?;
    let batch_lhs = batch_size
        .checked_mul(lhs_stride)
        .ok_or(OpenVinoUnavailable)?;
    let batch_rhs = batch_size
        .checked_mul(rhs_stride)
        .ok_or(OpenVinoUnavailable)?;
    let batch_out = batch_size
        .checked_mul(out_stride)
        .ok_or(OpenVinoUnavailable)?;

    // Cache hits also execute native functions on this calling thread.
    ensure_openvino_loaded()?;
    let cache_key = (batch_size, m, k, n, npu_only);
    let charge = lhs_data
        .len()
        .checked_add(rhs_data.len())
        .and_then(|v| v.checked_add(batch_out))
        .and_then(|v| v.checked_mul(4))
        .ok_or(OpenVinoUnavailable)?;
    let cached = OV_CACHE.get_or_try_init(cache_key, charge, || {
        compile_matmul(batch_size, m, k, n, npu_only).map(Mutex::new)
    })?;
    let mut entry = cached.lock().map_err(|_| OpenVinoUnavailable)?;
    if entry.failed {
        return Err(OpenVinoUnavailable);
    }
    let outcome = (|| {
        let lt_buf = entry
            .lhs
            .get_data_mut::<f32>()
            .map_err(|e| diagnostics::failure("OpenVINO matmul I/O", e))?;
        if lt_buf.len() != batch_lhs || lt_buf.len() != lhs_data.len() {
            return Err(OpenVinoUnavailable);
        }
        lt_buf.copy_from_slice(lhs_data);
        let rt_buf = entry
            .rhs
            .get_data_mut::<f32>()
            .map_err(|e| diagnostics::failure("OpenVINO matmul I/O", e))?;
        if rt_buf.len() != batch_rhs || rt_buf.len() != rhs_data.len() {
            return Err(OpenVinoUnavailable);
        }
        rt_buf.copy_from_slice(rhs_data);
        entry
            .request
            .infer()
            .map_err(|e| diagnostics::failure("OpenVINO matmul inference", e))?;
        diagnostics::executed(entry.target);
        let output = entry
            .request
            .get_output_tensor_by_index(0)
            .map_err(|e| diagnostics::failure("OpenVINO matmul I/O", e))?;
        let out_buf = output
            .get_data::<f32>()
            .map_err(|e| diagnostics::failure("OpenVINO matmul I/O", e))?;
        if out_buf.len() != batch_out || range::max_abs(out_buf).is_err() {
            return Err(OpenVinoUnavailable);
        }
        // Copy while holding the lock: the request owns/reuses the output storage.
        Ok(out_buf.to_vec())
    })();
    if outcome.is_err() {
        entry.failed = true;
        drop(entry);
        OV_CACHE.mark_failed(&cache_key, &cached);
    }
    let result_data = outcome?;

    Ok(IntelFloatTensor::new(result_data, out_shape))
}

fn compile_matmul(
    batch_size: usize,
    m: usize,
    k: usize,
    n: usize,
    npu_only: bool,
) -> Result<OvCompiledMatmul, OpenVinoUnavailable> {
    use openvino::{Core, DeviceType};
    // Flatten matching batch dimensions without changing their element order.
    // Keep single matrices 2D for compatibility with existing compiled kernels.
    let prefix = if batch_size > 1 {
        format!("{batch_size},")
    } else {
        String::new()
    };
    let batch_dims = if batch_size > 1 {
        format!("<dim>{batch_size}</dim>")
    } else {
        String::new()
    };

    let ir_xml = format!(
        r#"<?xml version="1.0"?>
<net name="matmul" version="11">
  <layers>
    <layer id="0" name="lhs" type="Parameter" version="opset1">
      <data shape="{prefix}{m},{k}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="lhs">{batch_dims}<dim>{m}</dim><dim>{k}</dim></port></output>
    </layer>
    <layer id="1" name="rhs" type="Parameter" version="opset1">
      <data shape="{prefix}{k},{n}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="rhs">{batch_dims}<dim>{k}</dim><dim>{n}</dim></port></output>
    </layer>
    <layer id="2" name="mm" type="MatMul" version="opset1">
      <data transpose_a="false" transpose_b="false"/>
      <input>
<port id="0">{batch_dims}<dim>{m}</dim><dim>{k}</dim></port>
<port id="1">{batch_dims}<dim>{k}</dim><dim>{n}</dim></port>
      </input>
      <output><port id="2" precision="FP32">{batch_dims}<dim>{m}</dim><dim>{n}</dim></port></output>
    </layer>
    <layer id="3" name="result" type="Result" version="opset1">
      <input><port id="0">{batch_dims}<dim>{m}</dim><dim>{n}</dim></port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
  </edges>
</net>"#
    );

    let mut core = Core::new().map_err(|_| OpenVinoUnavailable)?;
    let model = core
        .read_model_from_buffer(ir_xml.as_bytes(), None)
        .map_err(|_| OpenVinoUnavailable)?;

    let devices = [
        (DeviceType::NPU, Target::Npu),
        (DeviceType::GPU, Target::Gpu),
        (DeviceType::CPU, Target::Cpu),
    ];
    let mut compiled = None;
    for (dev, target) in &devices {
        if npu_only && *target != Target::Npu {
            continue;
        }
        match core.compile_model(&model, dev.to_owned()) {
            Ok(c) => {
                if std::env::var_os("BURN_NPU_TRACE").is_some() {
                    eprintln!(
                        "OpenVINO matmul batch={batch_size} {m}x{k}x{n}: compiled on {dev:?}"
                    );
                }
                compiled = Some((c, *target));
                break;
            }
            Err(err) => {
                diagnostics::failure(&format!("OpenVINO {dev:?} matmul compilation"), err);
            }
        }
    }
    let (mut compiled, target) = compiled.ok_or(OpenVinoUnavailable)?;
    let request = compiled
        .create_infer_request()
        .map_err(|_| OpenVinoUnavailable)?;
    // Write into the request's own host tensors. Binding separate user
    // tensors can force another copy inside the NPU plugin.
    let lt = request.get_tensor("lhs").map_err(|_| OpenVinoUnavailable)?;
    let rt = request.get_tensor("rhs").map_err(|_| OpenVinoUnavailable)?;
    Ok(OvCompiledMatmul {
        request,
        lhs: lt,
        rhs: rt,
        _compiled: compiled,
        target,
        failed: false,
    })
}

/// Compiled attention for supported FP32 configurations. The caller must use
/// its semantic fallback for unsupported cases and softcapping.
pub fn openvino_attention(
    query: &burn_flex::FlexTensor,
    key: &burn_flex::FlexTensor,
    value: &burn_flex::FlexTensor,
    bias: Option<&burn_flex::FlexTensor>,
    options: &burn_tensor::ops::AttentionModuleOptions,
) -> Result<burn_flex::FlexTensor, OpenVinoUnavailable> {
    ensure_openvino_loaded()?;
    attention::execute(query, key, value, bias, None, options)
}

pub(crate) fn openvino_attention_masked(
    query: &burn_flex::FlexTensor,
    key: &burn_flex::FlexTensor,
    value: &burn_flex::FlexTensor,
    bias: Option<&burn_flex::FlexTensor>,
    mask: Option<&burn_flex::FlexTensor>,
    options: &burn_tensor::ops::AttentionModuleOptions,
) -> Result<burn_flex::FlexTensor, OpenVinoUnavailable> {
    ensure_openvino_loaded()?;
    attention::execute(query, key, value, bias, mask, options)
}

/// CPU fallback using Flex's SIMD kernels and NumPy-style batch broadcasting.
/// Invalid input shapes panic, matching Burn's tensor operation contract.
pub fn cpu_matmul(lhs: &IntelFloatTensor, rhs: &IntelFloatTensor) -> IntelFloatTensor {
    use burn_tensor::ops::FloatTensorOps;
    let output = <burn_flex::Flex as FloatTensorOps<burn_flex::Flex>>::float_matmul(
        intel_to_ndarray(lhs),
        intel_to_ndarray(rhs),
    );
    ndarray_to_intel(&output)
}

// ---------------------------------------------------------------------------
// Conversion helpers for Flex interop
// ---------------------------------------------------------------------------

/// Convert IntelFloatTensor -> FlexTensor (for delegating ops to burn-flex).
pub fn intel_to_ndarray(tensor: &IntelFloatTensor) -> burn_flex::FlexTensor {
    burn_flex::FlexTensor::from_data(burn_tensor::TensorData::new(
        tensor.data.clone(),
        tensor.shape.clone(),
    ))
}

/// Convert FlexTensor (f32) -> IntelFloatTensor.
pub fn ndarray_to_intel(tensor: &burn_flex::FlexTensor) -> IntelFloatTensor {
    assert_eq!(
        burn_tensor::TensorMetadata::dtype(tensor),
        DType::F32,
        "ndarray_to_intel: expected an f32 tensor"
    );
    let contig = tensor.to_contiguous();
    let shape = burn_tensor::TensorMetadata::shape(tensor).to_vec();
    IntelFloatTensor::new(contig.storage::<f32>().to_vec(), shape)
}
