//! One compiled graph for QK^T, scaling, additive bias, softmax and PV.
//! Unsupported options/layouts fall back through the caller to Burn Flex.
use super::cache::{BuildCache, CacheStats};
use super::{
    diagnostics::{self, Target},
    OpenVinoUnavailable,
};
use burn_flex::FlexTensor;
use burn_tensor::{ops::AttentionModuleOptions, DType, TensorData, TensorMetadata};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;
static CACHE: LazyLock<BuildCache<Vec<usize>, Mutex<Entry>>> =
    LazyLock::new(|| BuildCache::new(16, 64 * 1024 * 1024, Duration::from_secs(30)));
pub(super) fn stats() -> CacheStats {
    CACHE.stats()
}
pub(super) fn clear() {
    CACHE.clear();
}
struct Entry {
    request: openvino::InferRequest,
    inputs: Vec<openvino::Tensor>,
    _compiled: openvino::CompiledModel,
    failed: bool,
}

pub(super) fn execute(
    q: &FlexTensor,
    k: &FlexTensor,
    v: &FlexTensor,
    bias: Option<&FlexTensor>,
    mask: Option<&FlexTensor>,
    options: &AttentionModuleOptions,
) -> Result<FlexTensor, OpenVinoUnavailable> {
    if options.softcap.is_some() {
        return Err(OpenVinoUnavailable);
    }
    if [q, k, v].iter().any(|t| t.dtype() != DType::F32)
        || bias.is_some_and(|t| t.dtype() != DType::F32)
    {
        return Err(OpenVinoUnavailable);
    }
    let (qs, ks, vs) = (q.shape(), k.shape(), v.shape());
    if qs.len() != 4
        || ks.len() != 4
        || vs.len() != 4
        || qs.iter().chain(ks.iter()).chain(vs.iter()).any(|&d| d == 0)
        || qs[..2] != ks[..2]
        || qs[..2] != vs[..2]
        || qs[3] != ks[3]
        || ks[2] != vs[2]
    {
        return Err(OpenVinoUnavailable);
    }
    let zero;
    let bias = match bias {
        Some(t) => t,
        None => {
            zero = FlexTensor::zeros([1, 1, 1, 1].into(), DType::F32);
            &zero
        }
    };
    let score = vec![qs[0], qs[1], qs[2], ks[2]];
    let original_shape = bias.shape();
    if original_shape.len() != 4
        || original_shape
            .iter()
            .zip(&score)
            .any(|(&b, &d)| b != 1 && b != d)
    {
        return Err(OpenVinoUnavailable);
    }
    let merged;
    let bias = if mask.is_some() || options.is_causal {
        merged = masked_bias(bias, mask, &score, options.is_causal)?;
        &merged
    } else {
        bias
    };
    let bs = bias.shape();
    let bias_contiguous = bias.to_contiguous();
    // Fully masked rows and nonfinite user biases use Burn's NaN-safe fallback.
    let row_len = bs[3];
    if row_len == 0
        || super::range::max_abs(bias_contiguous.storage::<f32>()).is_err()
        || bias_contiguous
            .storage::<f32>()
            .chunks(row_len)
            .any(|row| row.iter().all(|&v| v <= -super::range::LIMIT as f32))
    {
        return Err(OpenVinoUnavailable);
    }
    let scale = options.scale.unwrap_or_else(|| 1.0 / (qs[3] as f64).sqrt()) as f32;
    if !scale.is_finite() {
        return Err(OpenVinoUnavailable);
    }
    super::range::safe_attention(q, k, v, &bias_contiguous, scale)?;
    let mut key = Vec::new();
    for s in [&qs, &ks, &vs, &bs] {
        key.extend(s.iter().copied());
    }
    key.push(scale.to_bits() as usize);
    let shapes = vec![qs.to_vec(), ks.to_vec(), vs.to_vec(), bs.to_vec(), vec![1]];
    let out = vec![qs[0], qs[1], qs[2], vs[3]];
    let count = shapes
        .iter()
        .try_fold(super::elements(&out)?, |count, shape| {
            count
                .checked_add(super::elements(shape)?)
                .ok_or(OpenVinoUnavailable)
        })?;
    let charge = count.checked_mul(4).ok_or(OpenVinoUnavailable)?;
    let cached = CACHE.get_or_try_init(key.clone(), charge, || {
        compile(&shapes, &score, &out, scale).map(Mutex::new)
    })?;
    let mut entry = cached.lock().map_err(|_| OpenVinoUnavailable)?;
    if entry.failed {
        return Err(OpenVinoUnavailable);
    }
    let outcome = (|| {
        for (i, tensor) in [q, k, v, bias].iter().enumerate() {
            let contiguous = tensor.to_contiguous();
            let input = entry.inputs[i]
                .get_data_mut::<f32>()
                .map_err(|e| diagnostics::failure("OpenVINO attention I/O", e))?;
            if input.len() != contiguous.storage::<f32>().len() {
                return Err(OpenVinoUnavailable);
            }
            input.copy_from_slice(contiguous.storage::<f32>());
        }
        entry
            .request
            .infer()
            .map_err(|e| diagnostics::failure("OpenVINO attention inference", e))?;
        diagnostics::executed(Target::Npu);
        let output = entry
            .request
            .get_output_tensor_by_index(0)
            .map_err(|e| diagnostics::failure("OpenVINO attention I/O", e))?;
        let values = output
            .get_data::<f32>()
            .map_err(|e| diagnostics::failure("OpenVINO attention I/O", e))?;
        // Burn's fallback has a NaN-safe softmax for fully masked rows. If the NPU
        // softmax cannot represent them, use that implementation instead.
        if values.len() != super::elements(&out)? || super::range::max_abs(values).is_err() {
            return Err(OpenVinoUnavailable);
        }
        Ok(values.to_vec())
    })();
    if outcome.is_err() {
        entry.failed = true;
        drop(entry);
        CACHE.mark_failed(&key, &cached);
    }
    Ok(FlexTensor::from_data(TensorData::new(outcome?, out)))
}

/// Materialize only the combined bias. A bounded score range makes -1e9 an
/// exact zero-probability mask even if a driver clips FP16 infinities.
fn masked_bias(
    bias: &FlexTensor,
    mask: Option<&FlexTensor>,
    score: &[usize],
    causal: bool,
) -> Result<FlexTensor, OpenVinoUnavailable> {
    if causal && score[2] > score[3] {
        return Err(OpenVinoUnavailable);
    } // leading fully masked rows
    let bias = bias.to_contiguous();
    let bs = bias.shape();
    if super::range::max_abs(bias.storage::<f32>()).is_err() {
        return Err(OpenVinoUnavailable);
    }
    let mask_data = mask
        .map(|m| {
            let ms = m.shape();
            if ms.len() != 4 || ms.iter().zip(score).any(|(&m, &d)| m != 1 && m != d) {
                return Err(OpenVinoUnavailable);
            }
            let values = m
                .clone()
                .into_data()
                .to_vec::<bool>()
                .map_err(|_| OpenVinoUnavailable)?;
            Ok((ms, values))
        })
        .transpose()?;
    let shape = vec![
        bs[0].max(mask_data.as_ref().map_or(1, |m| m.0[0])),
        bs[1].max(mask_data.as_ref().map_or(1, |m| m.0[1])),
        score[2],
        score[3],
    ];
    let len = super::elements(&shape)?;
    if len > 16 * 1024 * 1024 {
        return Err(OpenVinoUnavailable);
    }
    let index = |shape: &[usize], b: usize, h: usize, i: usize, j: usize| {
        (((b % shape[0]) * shape[1] + h % shape[1]) * shape[2] + i % shape[2]) * shape[3]
            + j % shape[3]
    };
    let mut values = Vec::with_capacity(len);
    for b in 0..shape[0] {
        for h in 0..shape[1] {
            for i in 0..shape[2] {
                for j in 0..shape[3] {
                    let hidden = (causal && j > i + score[3] - score[2])
                        || mask_data
                            .as_ref()
                            .is_some_and(|(ms, data)| data[index(ms, b, h, i, j)]);
                    values.push(if hidden {
                        -1e9
                    } else {
                        bias.storage::<f32>()[index(&bs, b, h, i, j)]
                    });
                }
            }
        }
    }
    Ok(FlexTensor::from_data(TensorData::new(values, shape)))
}

fn dims(shape: &[usize]) -> String {
    shape.iter().map(|d| format!("<dim>{d}</dim>")).collect()
}
fn layer(
    id: usize,
    name: &str,
    kind: &str,
    version: &str,
    data: &str,
    inputs: &[&[usize]],
    out: Option<&[usize]>,
) -> String {
    let mut xml = format!(
        r#"<layer id="{id}" name="{name}" type="{kind}" version="{version}"><data {data}/><input>"#
    );
    for (port, shape) in inputs.iter().enumerate() {
        xml += &format!(
            r#"<port id="{port}" precision="FP32">{}</port>"#,
            dims(shape)
        );
    }
    xml += "</input>";
    if let Some(shape) = out {
        xml += &format!(
            r#"<output><port id="{}" precision="FP32" names="{name}">{}</port></output>"#,
            inputs.len(),
            dims(shape)
        );
    }
    xml += "</layer>";
    xml
}
fn compile(
    shapes: &[Vec<usize>],
    score: &[usize],
    out: &[usize],
    scale: f32,
) -> Result<Entry, OpenVinoUnavailable> {
    use openvino::{Core, DeviceType};
    let mut layers = String::new();
    for (i, shape) in shapes.iter().enumerate() {
        let shape_text = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        layers += &layer(
            i,
            &format!("input{i}"),
            "Parameter",
            "opset1",
            &format!(r#"shape="{shape_text}" element_type="f32""#),
            &[],
            Some(shape),
        );
    }
    layers += &layer(
        5,
        "qk",
        "MatMul",
        "opset1",
        r#"transpose_a="false" transpose_b="true""#,
        &[&shapes[0], &shapes[1]],
        Some(score),
    );
    layers += &layer(
        6,
        "scaled",
        "Multiply",
        "opset1",
        r#"auto_broadcast="numpy""#,
        &[score, &shapes[4]],
        Some(score),
    );
    layers += &layer(
        7,
        "biased",
        "Add",
        "opset1",
        r#"auto_broadcast="numpy""#,
        &[score, &shapes[3]],
        Some(score),
    );
    layers += &layer(
        8,
        "probs",
        "Softmax",
        "opset8",
        r#"axis="-1""#,
        &[score],
        Some(score),
    );
    layers += &layer(
        9,
        "context",
        "MatMul",
        "opset1",
        r#"transpose_a="false" transpose_b="false""#,
        &[score, &shapes[2]],
        Some(out),
    );
    layers += &layer(10, "result", "Result", "opset1", "", &[out], None);
    let mut edges = String::new();
    for (from, port, to, input) in [
        (0, 0, 5, 0),
        (1, 0, 5, 1),
        (5, 2, 6, 0),
        (4, 0, 6, 1),
        (6, 2, 7, 0),
        (3, 0, 7, 1),
        (7, 2, 8, 0),
        (8, 1, 9, 0),
        (2, 0, 9, 1),
        (9, 2, 10, 0),
    ] {
        edges += &format!(
            r#"<edge from-layer="{from}" from-port="{port}" to-layer="{to}" to-port="{input}"/>"#
        );
    }
    let xml = format!(
        r#"<?xml version="1.0"?><net name="attention" version="11"><layers>{layers}</layers><edges>{edges}</edges></net>"#
    );
    let mut core = Core::new().map_err(|_| OpenVinoUnavailable)?;
    let model = core
        .read_model_from_buffer(xml.as_bytes(), None)
        .map_err(|_| OpenVinoUnavailable)?;
    let mut compiled = core.compile_model(&model, DeviceType::NPU).map_err(|err| {
        if std::env::var_os("BURN_NPU_TRACE").is_some() {
            eprintln!("OpenVINO attention compilation unavailable: {err}");
        }
        diagnostics::failure("OpenVINO attention compilation", err)
    })?;
    let request = compiled
        .create_infer_request()
        .map_err(|_| OpenVinoUnavailable)?;
    let mut inputs = (0..5)
        .map(|i| {
            request
                .get_tensor(&format!("input{i}"))
                .map_err(|_| OpenVinoUnavailable)
        })
        .collect::<Result<Vec<_>, _>>()?;
    inputs[4]
        .get_data_mut::<f32>()
        .map_err(|_| OpenVinoUnavailable)?[0] = scale;
    if std::env::var_os("BURN_NPU_TRACE").is_some() {
        eprintln!("OpenVINO fused attention {:?}: compiled on NPU", shapes[0]);
    }
    Ok(Entry {
        request,
        inputs,
        _compiled: compiled,
        failed: false,
    })
}
