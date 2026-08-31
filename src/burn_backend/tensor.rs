//! `NpuFloatTensor` type definitions and conversion helpers for all platforms.

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
use burn_flex::FlexTensor;
#[cfg(feature = "apple")]
use burn_tensor::{DType, Shape, TensorData, TensorMetadata};

// ---------------------------------------------------------------------------
// NpuFloatTensor — MLTensor handle (apple only)
// ---------------------------------------------------------------------------

#[cfg(feature = "apple")]
use super::ffi::*;

#[cfg(feature = "apple")]
#[derive(Debug)]
pub struct NpuFloatTensor {
    pub(crate) handle: i32,
}

// SAFETY: MLTensor handles are thread-safe integers into a Swift-side table.
#[cfg(feature = "apple")]
unsafe impl Send for NpuFloatTensor {}
#[cfg(feature = "apple")]
unsafe impl Sync for NpuFloatTensor {}

#[cfg(feature = "apple")]
impl Clone for NpuFloatTensor {
    fn clone(&self) -> Self {
        Self {
            handle: unsafe { npu_clone(self.handle) },
        }
    }
}

#[cfg(feature = "apple")]
impl Drop for NpuFloatTensor {
    fn drop(&mut self) {
        unsafe { npu_free_tensor(self.handle) };
    }
}

#[cfg(feature = "apple")]
impl burn_tensor::TensorMetadata for NpuFloatTensor {
    fn dtype(&self) -> DType {
        DType::F32
    }

    fn shape(&self) -> Shape {
        let mut buf = [0i32; 8];
        let ndim = unsafe { npu_get_shape(self.handle, buf.as_mut_ptr(), 8) } as usize;
        Shape::from(buf[..ndim].iter().map(|&d| d as usize).collect::<Vec<_>>())
    }
}

// ---------------------------------------------------------------------------
// NpuFloatTensor — IntelFloatTensor wrapper (intel feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "intel")]
pub type NpuFloatTensor = crate::backends::intel::IntelFloatTensor;

// ---------------------------------------------------------------------------
// NpuFloatTensor — QnnFloatTensor wrapper (qualcomm feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "qualcomm")]
pub type NpuFloatTensor = crate::backends::qualcomm::QnnFloatTensor;

// ===========================================================================
// Conversion helpers (intel)
// ===========================================================================

#[cfg(feature = "intel")]
pub(super) fn npu_to_ndarray(tensor: &NpuFloatTensor) -> FlexTensor {
    crate::backends::intel::intel_to_ndarray(tensor)
}

#[cfg(feature = "intel")]
pub(super) fn ndarray_to_npu(tensor: &FlexTensor) -> NpuFloatTensor {
    crate::backends::intel::ndarray_to_intel(tensor)
}

// ===========================================================================
// Conversion helpers (qualcomm)
// ===========================================================================

#[cfg(feature = "qualcomm")]
pub(super) fn npu_to_ndarray(tensor: &NpuFloatTensor) -> FlexTensor {
    crate::backends::qualcomm::qnn_to_ndarray(tensor)
}

#[cfg(feature = "qualcomm")]
pub(super) fn ndarray_to_npu(tensor: &FlexTensor) -> NpuFloatTensor {
    crate::backends::qualcomm::ndarray_to_qnn(tensor)
}

// ===========================================================================
// Conversion helpers (apple)
// ===========================================================================

#[cfg(feature = "apple")]
pub(super) fn read_f32(handle: i32) -> (Vec<f32>, Vec<usize>) {
    let mut shape_buf = [0i32; 8];
    let ndim = unsafe { npu_get_shape(handle, shape_buf.as_mut_ptr(), 8) } as usize;
    let shape: Vec<usize> = shape_buf[..ndim].iter().map(|&d| d as usize).collect();
    let total: usize = shape.iter().product();
    let mut data = vec![0.0f32; total];
    unsafe { npu_get_data(handle, data.as_mut_ptr(), total as i32) };
    (data, shape)
}

#[cfg(feature = "apple")]
pub(super) fn read_int(handle: i32) -> (Vec<i32>, Vec<usize>) {
    let mut shape_buf = [0i32; 8];
    let ndim = unsafe { npu_get_shape(handle, shape_buf.as_mut_ptr(), 8) } as usize;
    let shape: Vec<usize> = shape_buf[..ndim].iter().map(|&d| d as usize).collect();
    let total: usize = shape.iter().product();
    let mut data = vec![0i32; total];
    unsafe { npu_get_int_data(handle, data.as_mut_ptr(), total as i32) };
    (data, shape)
}

/// Read an integer FlexTensor as `i64`, whatever width it is stored at.
///
/// `IntElem` is `i32` to match burn-flex, but index buffers reaching the FFI
/// are widened so a single code path covers both.
#[cfg(feature = "apple")]
pub(super) fn extract_i64(tensor: &FlexTensor) -> Vec<i64> {
    let contig = tensor.to_contiguous();
    match contig.dtype() {
        DType::I64 => contig.storage::<i64>().to_vec(),
        DType::I32 => contig.storage::<i32>().iter().map(|&v| v as i64).collect(),
        other => panic!("extract_i64: expected an integer tensor, got {other:?}"),
    }
}

#[cfg(feature = "apple")]
pub(super) fn npu_to_ndarray(tensor: &NpuFloatTensor) -> FlexTensor {
    let (data, shape) = read_f32(tensor.handle);
    FlexTensor::from_data(TensorData::new(data, shape))
}

#[cfg(feature = "apple")]
pub(super) fn ndarray_to_npu(tensor: &FlexTensor) -> NpuFloatTensor {
    assert_eq!(
        tensor.dtype(),
        DType::F32,
        "ndarray_to_npu: expected an f32 tensor"
    );
    let contig = tensor.to_contiguous();
    let data = contig.storage::<f32>();
    let shape: Vec<i32> = TensorMetadata::shape(tensor)
        .iter()
        .map(|&d| d as i32)
        .collect();
    NpuFloatTensor {
        handle: unsafe {
            npu_create_tensor(
                shape.as_ptr(),
                shape.len() as i32,
                data.as_ptr(),
                data.len() as i32,
            )
        },
    }
}

#[cfg(feature = "apple")]
#[inline]
pub(super) fn shape_i32(shape: &Shape) -> Vec<i32> {
    shape.iter().map(|&d| d as i32).collect()
}

#[cfg(feature = "apple")]
pub(super) fn int_handle_to_ndarray(handle: i32) -> FlexTensor {
    let (int_data, shape) = read_int(handle);
    unsafe { npu_free_tensor(handle) };
    // IntElem is i32, matching both the MLTensor int payload and burn-flex.
    FlexTensor::from_data(TensorData::new(int_data, shape))
}

/// Read a comparison result handle as a bool tensor, inverting it.
///
/// `greater_equal` is `NOT less` and `lower_equal` is `NOT greater`; MLTensor
/// has no primitive for either, so both pairs go through here.
#[cfg(feature = "apple")]
pub(super) fn float_handle_to_inverted_bool(handle: i32) -> FlexTensor {
    let (data, shape) = read_f32(handle);
    unsafe { npu_free_tensor(handle) };
    let bool_data: Vec<bool> = data.iter().map(|&v| v == 0.0).collect();
    FlexTensor::from_data(TensorData::new(bool_data, shape))
}

#[cfg(feature = "apple")]
pub(super) fn float_handle_to_bool_ndarray(handle: i32) -> FlexTensor {
    let (data, shape) = read_f32(handle);
    unsafe { npu_free_tensor(handle) };
    let bool_data: Vec<bool> = data.iter().map(|&v| v != 0.0).collect();
    FlexTensor::from_data(TensorData::new(bool_data, shape))
}
