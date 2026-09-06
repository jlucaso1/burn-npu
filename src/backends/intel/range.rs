//! Conservative FP16 range checks. OpenVINO NPU may saturate to a finite value,
//! so checking only for NaN/Inf in the output is insufficient. Weak identities
//! cache large RHS bounds without retaining model weights. Flex mutations use
//! Arc::make_mut/get_mut, which detach or reject storage with weak references.
use super::OpenVinoUnavailable;
use burn_flex::FlexTensor;
use burn_tensor::Bytes;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock, Mutex, Weak};
const LIMIT: f64 = 65504.0 * 0.99;
#[derive(Default)]
struct Bounds {
    entries: HashMap<usize, (Weak<Bytes>, f64)>,
    order: VecDeque<usize>,
}
static BOUNDS: LazyLock<Mutex<Bounds>> = LazyLock::new(|| Mutex::new(Bounds::default()));
pub(super) fn max_abs(data: &[f32]) -> Result<f64, OpenVinoUnavailable> {
    // After clearing the sign bit, IEEE-754 bit patterns have unsigned order.
    // Integer max is associative and vectorizes without unsafe SIMD or fast-math.
    // Inf and every NaN payload sort above all finite magnitudes.
    let bits = data
        .iter()
        .map(|v| v.to_bits() & 0x7fff_ffff)
        .max()
        .unwrap_or(0);
    if bits >= f32::INFINITY.to_bits() {
        return Err(OpenVinoUnavailable);
    }
    Ok(f32::from_bits(bits) as f64)
}
pub(super) fn rhs_max(tensor: &FlexTensor) -> Result<f64, OpenVinoUnavailable> {
    if tensor.storage::<f32>().len() < 65536 {
        return max_abs(tensor.storage::<f32>());
    }
    let data = tensor.data_arc();
    let id = Arc::as_ptr(&data) as usize;
    let mut bounds = BOUNDS.lock().map_err(|_| OpenVinoUnavailable)?;
    if let Some((_, max)) = bounds.entries.get(&id) {
        let max = *max;
        bounds.order.retain(|&key| key != id);
        bounds.order.push_back(id);
        return Ok(max);
    }
    drop(bounds);
    let max = max_abs(tensor.storage::<f32>())?;
    let mut bounds = BOUNDS.lock().map_err(|_| OpenVinoUnavailable)?;
    // Another caller may have populated this identity while we scanned it.
    bounds.order.retain(|&key| key != id);
    while bounds.entries.len() >= 256 && !bounds.entries.contains_key(&id) {
        if let Some(old) = bounds.order.pop_front() {
            bounds.entries.remove(&old);
        }
    }
    bounds.entries.insert(id, (Arc::downgrade(&data), max));
    bounds.order.push_back(id);
    Ok(max)
}
pub(super) fn safe_matmul(k: usize, a: f64, b: f64) -> Result<(), OpenVinoUnavailable> {
    if a > LIMIT || b > LIMIT || k as f64 * a * b > LIMIT {
        Err(OpenVinoUnavailable)
    } else {
        Ok(())
    }
}
pub(super) fn safe_attention(
    q: &FlexTensor,
    k: &FlexTensor,
    v: &FlexTensor,
    bias: &FlexTensor,
    scale: f32,
) -> Result<(), OpenVinoUnavailable> {
    use burn_tensor::TensorMetadata;
    let q_max = max_abs(q.storage::<f32>())?;
    let k_max = max_abs(k.storage::<f32>())?;
    // Small products do not make individually unrepresentable operands safe.
    // The scalar scale is also uploaded as an NPU input.
    if q_max > LIMIT || k_max > LIMIT || (scale as f64).abs() > LIMIT {
        return Err(OpenVinoUnavailable);
    }
    let bound = q.shape()[3] as f64 * q_max * k_max;
    let non_mask_bias = bias
        .storage::<f32>()
        .iter()
        .filter(|&&x| x > -65504.0)
        .fold(0_f32, |m, x| m.max(x.abs())) as f64;
    if bound > LIMIT
        || bound * (scale as f64).abs() + non_mask_bias > 16000.0
        || max_abs(v.storage::<f32>())? > LIMIT
    {
        Err(OpenVinoUnavailable)
    } else {
        Ok(())
    }
}
pub(super) fn clear() {
    let mut bounds = BOUNDS.lock().unwrap_or_else(|e| e.into_inner());
    bounds.entries.clear();
    bounds.order.clear();
}
#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;
    #[test]
    fn weak_bounds_follow_mutation_and_do_not_retain_weights() {
        let mut t = FlexTensor::from_data(TensorData::new(vec![1_f32; 65536], [256, 256]));
        assert_eq!(rhs_max(&t).unwrap(), 1.);
        assert_eq!(Arc::strong_count(&t.data_arc()), 2); // t plus temporary returned Arc
        t.storage_mut::<f32>()[0] = 512.;
        assert_eq!(rhs_max(&t).unwrap(), 512.);
        assert!(safe_matmul(64, 256., 256.).is_err());
    }
    #[test]
    fn bound_rejects_all_nonfinite_values() {
        assert_eq!(max_abs(&[-0., -512., 1.]).unwrap(), 512.);
        for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert!(max_abs(&[1., value, 2.]).is_err());
        }
    }
}
