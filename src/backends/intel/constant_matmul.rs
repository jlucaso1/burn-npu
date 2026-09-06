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
    // Owns the blob handed to `read_model_from_buffer`, so the compiled
    // model can never outlive it.
    _weight_blob: openvino::Tensor,
}

impl super::cache::CachedEntry for Entry {
    fn has_failed(&self) -> bool {
        self.failed
    }

    fn set_failed(&mut self) {
        self.failed = true;
    }
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
    let lhs = lhs.to_contiguous();
    let data = super::cache::run_cached(
        &CACHE,
        key.clone(),
        charge,
        || compile(rhs, m, k, n),
        |entry| {
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
            if data.len() != m.checked_mul(n).ok_or(OpenVinoUnavailable)?
                || super::range::max_abs(data).is_err()
            {
                return Err(OpenVinoUnavailable);
            }
            Ok(data.to_vec())
        },
    )?;
    let mut shape = vec![1; rs.len().saturating_sub(ls.len())];
    shape.extend(ls.iter().copied());
    *shape.last_mut().unwrap() = n;
    Ok(FlexTensor::from_data(TensorData::new(data, shape)))
}

fn compile(rhs: &FlexTensor, m: usize, k: usize, n: usize) -> Result<Entry, OpenVinoUnavailable> {
    use super::ir::{dims, edge, layer, net};
    use openvino::{Core, DeviceType, ElementType, Shape, Tensor};
    let size = rhs.bytes().len();
    let mk = [m, k];
    let kn = [k, n];
    let mn = [m, n];
    let mut layers = layer(
        0,
        "lhs",
        "Parameter",
        "opset1",
        &format!(r#"shape="{m},{k}" element_type="f32""#),
        &[],
        Some(&mk),
    );
    layers += &format!(
        r#"<layer id="1" name="weight" type="Const" version="opset1"><data shape="{k},{n}" element_type="f32" offset="0" size="{size}"/><output><port id="0" precision="FP32">{}</port></output></layer>"#,
        dims(&kn)
    );
    layers += &layer(
        2,
        "mm",
        "MatMul",
        "opset1",
        r#"transpose_a="false" transpose_b="false""#,
        &[&mk, &kn],
        Some(&mn),
    );
    // Result layers omit the precision attribute the other ports carry.
    layers += &format!(
        r#"<layer id="3" name="result" type="Result" version="opset1"><input><port id="0">{}</port></input></layer>"#,
        dims(&mn)
    );
    let edges = [edge(0, 0, 2, 0), edge(1, 0, 2, 1), edge(2, 2, 3, 0)].concat();
    let xml = net("constant_matmul", &layers, &edges);
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
    let mut compiled = super::ir::compile_on_device(
        &mut core,
        &xml,
        Some(&weights),
        DeviceType::NPU,
        "OpenVINO constant_matmul compilation",
    )?;
    let request = compiled
        .create_infer_request()
        .map_err(|_| OpenVinoUnavailable)?;
    let input = request.get_tensor("lhs").map_err(|_| OpenVinoUnavailable)?;
    if super::trace() {
        eprintln!("OpenVINO constant matmul {m}x{k}x{n}: compiled on NPU");
    }
    Ok(Entry {
        request,
        input,
        _compiled: compiled,
        failed: false,
        _weight: rhs.clone(),
        _weight_blob: weights,
    })
}
