//! Opt-in specialization for large, contiguous FP32 RHS tensors. The cache keeps
//! a strong reference to each RHS, so Flex copy-on-write makes pointer identity
//! safe against mutation and allocator address reuse. Entry count and retained
//! source bytes are bounded; driver/compiler memory is additional.
use super::cache::{BuildCache, CacheStats};
use super::{
    diagnostics::{self, Target},
    OpenVinoUnavailable,
};
use burn_flex::FlexTensor;
use burn_tensor::{TensorData, TensorMetadata};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

#[derive(Clone)]
struct Key {
    weight: FlexTensor,
    m: usize,
    k: usize,
    n: usize,
}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        (self.m, self.k, self.n) == (other.m, other.k, other.n)
            && Arc::ptr_eq(&self.weight.data_arc(), &other.weight.data_arc())
    }
}
impl Eq for Key {}
impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (
            Arc::as_ptr(&self.weight.data_arc()) as usize,
            self.m,
            self.k,
            self.n,
        )
            .hash(state);
    }
}
const MAX_WEIGHT_BYTES: usize = 512 * 1024 * 1024;
static CACHE: LazyLock<BuildCache<Key, Mutex<Entry>>> =
    LazyLock::new(|| BuildCache::new(64, MAX_WEIGHT_BYTES, Duration::from_secs(30)));
pub(super) fn stats() -> CacheStats {
    CACHE.stats()
}
pub(super) fn clear() {
    CACHE.clear();
}
struct Entry {
    request: openvino::InferRequest,
    input: openvino::Tensor,
    _compiled: openvino::CompiledModel,
    failed: bool,
    // Holds the exact allocation used by the key, not just a copied value.
    _weight: FlexTensor,
}

pub(super) fn matmul(
    lhs: &FlexTensor,
    rhs: &FlexTensor,
) -> Result<FlexTensor, OpenVinoUnavailable> {
    let ls = lhs.shape();
    let rs = rhs.shape();
    if ls.len() < 2
        || rs.len() < 2
        || !rhs.is_contiguous()
        || rhs.layout().start_offset() != 0
        || rs[..rs.len() - 2].iter().any(|&d| d != 1)
    {
        return Err(OpenVinoUnavailable);
    }
    let (m, k, n) = (
        super::elements(&ls[..ls.len() - 1])?,
        ls[ls.len() - 1],
        rs[rs.len() - 1],
    );
    if m == 0 || k < 256 || n < 256 || rs[rs.len() - 2] != k {
        return Err(OpenVinoUnavailable);
    }
    let weight_bytes = k
        .checked_mul(n)
        .and_then(|v| v.checked_mul(4))
        .ok_or(OpenVinoUnavailable)?;
    if rhs.bytes().len() != weight_bytes || weight_bytes > MAX_WEIGHT_BYTES {
        return Err(OpenVinoUnavailable);
    }
    let key = Key {
        weight: rhs.clone(),
        m,
        k,
        n,
    };
    let charge = m
        .checked_mul(k)
        .and_then(|v| v.checked_add(m.checked_mul(n)?))
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_add(weight_bytes))
        .ok_or(OpenVinoUnavailable)?;
    let cached = CACHE.get_or_try_init(key.clone(), charge, || {
        compile(rhs, m, k, n).map(Mutex::new)
    })?;
    let lhs = lhs.to_contiguous();
    let mut entry = cached.lock().map_err(|_| OpenVinoUnavailable)?;
    if entry.failed {
        return Err(OpenVinoUnavailable);
    }
    let outcome = (|| {
        let input = entry
            .input
            .get_data_mut::<f32>()
            .map_err(|e| diagnostics::failure("OpenVINO constant matmul I/O", e))?;
        if input.len() != lhs.storage::<f32>().len() {
            return Err(OpenVinoUnavailable);
        }
        input.copy_from_slice(lhs.storage::<f32>());
        entry
            .request
            .infer()
            .map_err(|e| diagnostics::failure("OpenVINO constant matmul inference", e))?;
        diagnostics::executed(Target::Npu);
        let output = entry
            .request
            .get_output_tensor_by_index(0)
            .map_err(|e| diagnostics::failure("OpenVINO constant matmul I/O", e))?;
        let data = output
            .get_data::<f32>()
            .map_err(|e| diagnostics::failure("OpenVINO constant matmul I/O", e))?;
        if data.len() != m * n || super::range::max_abs(data).is_err() {
            return Err(OpenVinoUnavailable);
        }
        Ok(data.to_vec())
    })();
    if outcome.is_err() {
        entry.failed = true;
        drop(entry);
        CACHE.mark_failed(&key, &cached);
    }
    let data = outcome?;
    let mut shape = vec![1; rs.len().saturating_sub(ls.len())];
    shape.extend(ls.iter().copied());
    *shape.last_mut().unwrap() = n;
    Ok(FlexTensor::from_data(TensorData::new(data, shape)))
}

fn compile(rhs: &FlexTensor, m: usize, k: usize, n: usize) -> Result<Entry, OpenVinoUnavailable> {
    use openvino::{Core, DeviceType, ElementType, Shape, Tensor};
    let size = rhs.bytes().len();
    let xml = format!(
        r#"<?xml version="1.0"?>
<net name="constant_matmul" version="11"><layers>
<layer id="0" name="lhs" type="Parameter" version="opset1">
<data shape="{m},{k}" element_type="f32"/>
<output><port id="0" precision="FP32" names="lhs"><dim>{m}</dim><dim>{k}</dim></port></output></layer>
<layer id="1" name="weight" type="Const" version="opset1">
<data shape="{k},{n}" element_type="f32" offset="0" size="{size}"/>
<output><port id="0" precision="FP32"><dim>{k}</dim><dim>{n}</dim></port></output></layer>
<layer id="2" name="mm" type="MatMul" version="opset1"><data transpose_a="false" transpose_b="false"/>
<input><port id="0"><dim>{m}</dim><dim>{k}</dim></port><port id="1"><dim>{k}</dim><dim>{n}</dim></port></input>
<output><port id="2" precision="FP32"><dim>{m}</dim><dim>{n}</dim></port></output></layer>
<layer id="3" name="result" type="Result" version="opset1"><input><port id="0"><dim>{m}</dim><dim>{n}</dim></port></input></layer>
</layers><edges>
<edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
<edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
</edges></net>"#
    );
    let mut weights = Tensor::new(
        ElementType::U8,
        &Shape::new(&[size as i64]).map_err(|_| OpenVinoUnavailable)?,
    )
    .map_err(|_| OpenVinoUnavailable)?;
    weights
        .get_data_mut::<u8>()
        .map_err(|_| OpenVinoUnavailable)?
        .copy_from_slice(rhs.bytes());
    let mut core = Core::new().map_err(|_| OpenVinoUnavailable)?;
    let model = core
        .read_model_from_buffer(xml.as_bytes(), Some(&weights))
        .map_err(|_| OpenVinoUnavailable)?;
    let mut compiled = core.compile_model(&model, DeviceType::NPU).map_err(|err| {
        if std::env::var_os("BURN_NPU_TRACE").is_some() {
            eprintln!("OpenVINO constant matmul {m}x{k}x{n} unavailable: {err}");
        }
        diagnostics::failure("OpenVINO constant_matmul compilation", err)
    })?;
    let request = compiled
        .create_infer_request()
        .map_err(|_| OpenVinoUnavailable)?;
    let input = request.get_tensor("lhs").map_err(|_| OpenVinoUnavailable)?;
    if std::env::var_os("BURN_NPU_TRACE").is_some() {
        eprintln!("OpenVINO constant matmul {m}x{k}x{n}: compiled on NPU");
    }
    Ok(Entry {
        request,
        input,
        _compiled: compiled,
        failed: false,
        _weight: rhs.clone(),
    })
}
