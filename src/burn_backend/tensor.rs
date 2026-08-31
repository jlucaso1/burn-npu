//! `NpuFloatTensor` type definitions and conversion helpers for all platforms.

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
use burn_ndarray::NdArrayTensor;
#[cfg(feature = "apple")]
use burn_tensor::{DType, Shape};

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
pub(super) fn npu_to_ndarray(tensor: &NpuFloatTensor) -> NdArrayTensor {
    crate::backends::intel::intel_to_ndarray(tensor)
}

#[cfg(feature = "intel")]
pub(super) fn ndarray_to_npu(tensor: &NdArrayTensor) -> NpuFloatTensor {
    crate::backends::intel::ndarray_to_intel(tensor)
}

// ===========================================================================
// Conversion helpers (qualcomm)
// ===========================================================================

#[cfg(feature = "qualcomm")]
pub(super) fn npu_to_ndarray(tensor: &NpuFloatTensor) -> NdArrayTensor {
    crate::backends::qualcomm::qnn_to_ndarray(tensor)
}

#[cfg(feature = "qualcomm")]
pub(super) fn ndarray_to_npu(tensor: &NdArrayTensor) -> NpuFloatTensor {
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

#[cfg(feature = "apple")]
pub(super) fn extract_i64(tensor: &NdArrayTensor) -> Vec<i64> {
    if let NdArrayTensor::I64(ref storage) = tensor {
        let view = storage.view();
        let contig = view.as_standard_layout();
        return contig.as_slice().unwrap().to_vec();
    }
    if let NdArrayTensor::I32(ref storage) = tensor {
        let view = storage.view();
        let contig = view.as_standard_layout();
        return contig
            .as_slice()
            .unwrap()
            .iter()
            .map(|&v| v as i64)
            .collect();
    }
    panic!("extract_i64: expected I64 or I32 NdArrayTensor");
}

#[cfg(feature = "apple")]
pub(super) fn npu_to_ndarray(tensor: &NpuFloatTensor) -> NdArrayTensor {
    let (data, shape) = read_f32(tensor.handle);
    let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), data)
        .unwrap()
        .into_shared();
    NdArrayTensor::from(array)
}

#[cfg(feature = "apple")]
pub(super) fn ndarray_to_npu(tensor: &NdArrayTensor) -> NpuFloatTensor {
    if let NdArrayTensor::F32(ref storage) = tensor {
        let view = storage.view();
        let contig = view.as_standard_layout();
        let data = contig.as_slice().unwrap();
        let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
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
    } else {
        panic!("ndarray_to_npu: expected F32 NdArrayTensor");
    }
}

#[cfg(feature = "apple")]
#[inline]
pub(super) fn shape_i32(shape: &Shape) -> Vec<i32> {
    shape.dims.iter().map(|&d| d as i32).collect()
}

#[cfg(feature = "apple")]
pub(super) fn int_handle_to_ndarray(handle: i32) -> NdArrayTensor {
    let (int_data, shape) = read_int(handle);
    unsafe { npu_free_tensor(handle) };
    let i64_data: Vec<i64> = int_data.iter().map(|&v| v as i64).collect();
    let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), i64_data)
        .unwrap()
        .into_shared();
    NdArrayTensor::from(array)
}

#[cfg(feature = "apple")]
pub(super) fn float_handle_to_bool_ndarray(handle: i32) -> NdArrayTensor {
    let (data, shape) = read_f32(handle);
    unsafe { npu_free_tensor(handle) };
    let bool_data: Vec<bool> = data.iter().map(|&v| v != 0.0).collect();
    let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), bool_data)
        .unwrap()
        .into_shared();
    NdArrayTensor::from(array)
}
