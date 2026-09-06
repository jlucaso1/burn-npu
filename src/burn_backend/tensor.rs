//! `NpuFloatTensor` type definitions and conversion helpers for all platforms.

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
use burn_flex::FlexTensor;
#[cfg(feature = "apple")]
use burn_tensor::{f16, DType, Shape, TensorData, TensorMetadata};

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

/// Scalar-type codes shared with the Swift shim's `npu_get_dtype`.
#[cfg(feature = "apple")]
pub(super) const DTYPE_F32: i32 = 0;
#[cfg(feature = "apple")]
pub(super) const DTYPE_F16: i32 = 1;

#[cfg(feature = "apple")]
impl burn_tensor::TensorMetadata for NpuFloatTensor {
    /// Asks the live MLTensor for its scalar type rather than assuming f32.
    ///
    /// MLTensor tracks this itself, so querying keeps every op's result dtype
    /// correct without threading a dtype through all the construction sites.
    fn dtype(&self) -> DType {
        match unsafe { npu_get_dtype(self.handle) } {
            DTYPE_F16 => DType::F16,
            _ => DType::F32,
        }
    }

    fn shape(&self) -> Shape {
        let mut buf = [0i32; 8];
        let ndim = unsafe { npu_get_shape(self.handle, buf.as_mut_ptr(), 8) } as usize;
        Shape::from(buf[..ndim].iter().map(|&d| d as usize).collect::<Vec<_>>())
    }
}

// ---------------------------------------------------------------------------
// NpuFloatTensor — shared Flex storage with OpenVINO matmul (intel feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "intel")]
pub type NpuFloatTensor = FlexTensor;

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
    tensor.clone()
}

#[cfg(feature = "intel")]
pub(super) fn ndarray_to_npu(tensor: &FlexTensor) -> NpuFloatTensor {
    tensor.clone()
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

/// Read an f16 tensor back as raw bit patterns.
#[cfg(feature = "apple")]
pub(super) fn read_f16(handle: i32) -> (Vec<f16>, Vec<usize>) {
    let mut shape_buf = [0i32; 8];
    let ndim = unsafe { npu_get_shape(handle, shape_buf.as_mut_ptr(), 8) } as usize;
    let shape: Vec<usize> = shape_buf[..ndim].iter().map(|&d| d as usize).collect();
    let total: usize = shape.iter().product();
    let mut bits = vec![0u16; total];
    unsafe { npu_get_data_f16(handle, bits.as_mut_ptr(), total as i32) };
    (bits.into_iter().map(f16::from_bits).collect(), shape)
}

/// Upload f16 data to the NPU, keeping it in the ANE's native format.
#[cfg(feature = "apple")]
pub(super) fn f16_to_npu(values: &[f16], shape: &[usize]) -> NpuFloatTensor {
    let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    let bits: Vec<u16> = values.iter().map(|v| v.to_bits()).collect();
    NpuFloatTensor {
        handle: unsafe {
            npu_create_tensor_f16(
                dims.as_ptr(),
                dims.len() as i32,
                bits.as_ptr(),
                bits.len() as i32,
            )
        },
    }
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

/// Upload a CPU bool mask to the NPU as a 0.0/1.0 float tensor.
///
/// Uploading the mask is cheaper than materialising the value tensor: it moves
/// less data and does not force the lazy MLTensor graph to be evaluated.
#[cfg(feature = "apple")]
pub(super) fn bool_mask_to_npu(mask: FlexTensor) -> NpuFloatTensor {
    let shape: Vec<i32> = TensorMetadata::shape(&mask)
        .iter()
        .map(|&d| d as i32)
        .collect();
    let bits: Vec<bool> = mask.into_data().to_vec().expect("bool mask");
    let floats: Vec<f32> = bits.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    NpuFloatTensor {
        handle: unsafe {
            npu_create_tensor(
                shape.as_ptr(),
                shape.len() as i32,
                floats.as_ptr(),
                floats.len() as i32,
            )
        },
    }
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
