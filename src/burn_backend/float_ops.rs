//! `FloatTensorOps` implementations for all platform variants.

use burn_tensor::backend::ExecutionError;
use burn_tensor::ops::*;
use burn_tensor::ops::{BoolTensor, FloatTensor, IntTensor};
use burn_tensor::{Distribution, FloatDType, Shape, Slice, TensorData};

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
use super::tensor::*;
use super::{nd_dev, Nd, NpuBurnBackend, NpuBurnDevice};

// ===========================================================================
// FloatTensorOps — apple: all ops go through MLTensor handles
// ===========================================================================
#[cfg(feature = "apple")]
use super::ffi::*;
#[cfg(feature = "apple")]
use burn_ndarray::NdArrayTensor;

#[cfg(feature = "apple")]
impl FloatTensorOps<Self> for NpuBurnBackend {
    fn float_from_data(data: TensorData, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        let floats: Vec<f32> = data.to_vec().unwrap();
        let shape: Vec<i32> = data.shape.iter().map(|&d| d as i32).collect();
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

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NpuBurnDevice,
    ) -> FloatTensor<Self> {
        // Generate random data on CPU, then send to NPU
        let nd_tensor = <Nd as FloatTensorOps<Nd>>::float_random(shape, distribution, &nd_dev());
        ndarray_to_npu(&nd_tensor)
    }

    fn float_zeros(shape: Shape, _device: &NpuBurnDevice, _dtype: FloatDType) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_zeros(s.as_ptr(), s.len() as i32) },
        }
    }

    fn float_ones(shape: Shape, _device: &NpuBurnDevice, _dtype: FloatDType) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_ones(s.as_ptr(), s.len() as i32) },
        }
    }

    fn float_full(
        shape: Shape,
        fill_value: f32,
        _device: &NpuBurnDevice,
        _dtype: FloatDType,
    ) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_full(s.as_ptr(), s.len() as i32, fill_value) },
        }
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }

    fn float_to_device(tensor: FloatTensor<Self>, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        tensor
    }

    fn float_empty(shape: Shape, device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        Self::float_zeros(shape, device, dtype)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        let shape = burn_tensor::TensorMetadata::shape(&tensor);
        let total: usize = shape.dims.iter().product();
        let mut data = vec![0.0f32; total];
        unsafe { npu_get_data(tensor.handle, data.as_mut_ptr(), total as i32) };
        // tensor drops here, freeing the MLTensor handle
        Ok(TensorData::new(data, shape))
    }

    // ── Matmul ──────────────────────────────────────────────────────────

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_matmul(lhs.handle, rhs.handle) },
        }
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        // No FFI for cross product — round-trip through NdArray
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_cross(nd_lhs, nd_rhs, dim);
        ndarray_to_npu(&result)
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        let (data, shape) = read_f32(tensor.handle);
        let int_data: Vec<i64> = data.iter().map(|&v| v as i64).collect();
        let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), int_data)
            .unwrap()
            .into_shared();
        NdArrayTensor::from(array)
    }

    // ── Arithmetic ──────────────────────────────────────────────────────

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_add(lhs.handle, rhs.handle) },
        }
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_add_scalar(lhs.handle, rhs) },
        }
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sub(lhs.handle, rhs.handle) },
        }
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sub_scalar(lhs.handle, rhs) },
        }
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mul(lhs.handle, rhs.handle) },
        }
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mul_scalar(lhs.handle, rhs) },
        }
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_div(lhs.handle, rhs.handle) },
        }
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_div_scalar(lhs.handle, rhs) },
        }
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        // remainder = lhs - (lhs / rhs).floor() * rhs
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_remainder(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_remainder_scalar(nd_lhs, rhs);
        ndarray_to_npu(&result)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // recip = 1.0 / tensor
        NpuFloatTensor {
            handle: unsafe {
                let one = npu_scalar_tensor(1.0);
                let r = npu_div(one, tensor.handle);
                npu_free_tensor(one);
                r
            },
        }
    }

    // ── Shape / layout ──────────────────────────────────────────────────

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_transpose(tensor.handle, dim1 as i32, dim2 as i32) },
        }
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let perm: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
        NpuFloatTensor {
            handle: unsafe { npu_permute(tensor.handle, perm.as_ptr(), perm.len() as i32) },
        }
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        // No direct FFI — round-trip
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_flip(nd, axes);
        ndarray_to_npu(&result)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_reshape(tensor.handle, s.as_ptr(), s.len() as i32) },
        }
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let s = shape_i32(&shape);
        NpuFloatTensor {
            handle: unsafe { npu_expand(tensor.handle, s.as_ptr(), s.len() as i32) },
        }
    }

    // ── Gather / scatter / select ───────────────────────────────────────

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let idx_data = extract_i64(&indices);
        let idx_i32: Vec<i32> = idx_data.iter().map(|&v| v as i32).collect();
        let idx_len = idx_data.len();
        let idx_shape: Vec<i32> = vec![idx_len as i32];
        unsafe {
            let idx_handle = npu_create_int_tensor(
                idx_shape.as_ptr(),
                idx_shape.len() as i32,
                idx_i32.as_ptr(),
                idx_i32.len() as i32,
            );
            let result = npu_gather(tensor.handle, dim as i32, idx_handle);
            npu_free_tensor(idx_handle);
            NpuFloatTensor { handle: result }
        }
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result =
            <Nd as FloatTensorOps<Nd>>::float_scatter_add(dim, nd_tensor, indices, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        // Native NPU gather — no readback of weight tensor
        let idx_data = extract_i64(&indices);
        let idx_i32: Vec<i32> = idx_data.iter().map(|&v| v as i32).collect();
        let idx_len = idx_data.len();
        let idx_shape: Vec<i32> = vec![idx_len as i32];
        unsafe {
            let idx_handle = npu_create_int_tensor(
                idx_shape.as_ptr(),
                idx_shape.len() as i32,
                idx_i32.as_ptr(),
                idx_i32.len() as i32,
            );
            let result = npu_gather(tensor.handle, dim as i32, idx_handle);
            npu_free_tensor(idx_handle);
            NpuFloatTensor { handle: result }
        }
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result =
            <Nd as FloatTensorOps<Nd>>::float_select_add(nd_tensor, dim, indices, nd_value);
        ndarray_to_npu(&result)
    }

    // ── Slice ───────────────────────────────────────────────────────────

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        // Fast path: simple contiguous ranges (step=1, no negative indices)
        // This covers narrow() which is the common case
        let ndim = slices.len();
        if ndim <= 3 {
            let all_simple = slices.iter().all(|s| s.step == 1 && s.start >= 0);
            if all_simple {
                let mut ranges = Vec::with_capacity(ndim * 2);
                let shape = {
                    let mut buf = [0i32; 8];
                    let n = unsafe { npu_get_shape(tensor.handle, buf.as_mut_ptr(), 8) } as usize;
                    buf[..n].iter().map(|&d| d as usize).collect::<Vec<_>>()
                };
                for (i, s) in slices.iter().enumerate() {
                    let start = s.start as i32;
                    let end = s.end.map(|e| e as i32).unwrap_or(shape[i] as i32);
                    ranges.push(start);
                    ranges.push(end);
                }
                let result = unsafe { npu_slice(tensor.handle, ranges.as_ptr(), ndim as i32) };
                if result >= 0 {
                    return NpuFloatTensor { handle: result };
                }
            }
        }
        // Fallback for complex slices
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_slice(nd, slices);
        ndarray_to_npu(&result)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // No direct FFI for slice_assign — round-trip
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Nd as FloatTensorOps<Nd>>::float_slice_assign(nd_tensor, slices, nd_value);
        ndarray_to_npu(&result)
    }

    // ── Mask ────────────────────────────────────────────────────────────

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // Round-trip: mask is NdArrayTensor<bool>, needs conversion
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Nd as FloatTensorOps<Nd>>::float_mask_where(nd_tensor, mask, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: f32,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_mask_fill(nd_tensor, mask, value);
        ndarray_to_npu(&result)
    }

    // ── Comparison (return BoolTensor = NdArrayTensor<bool>) ────────────

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let h = unsafe { npu_equal(lhs.handle, rhs.handle) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs) };
        let h = unsafe { npu_equal(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let h = unsafe { npu_greater(lhs.handle, rhs.handle) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs) };
        let h = unsafe { npu_greater(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        // greater_equal = NOT less
        let h = unsafe { npu_less(lhs.handle, rhs.handle) };
        let (data, shape) = read_f32(h);
        unsafe { npu_free_tensor(h) };
        let bool_data: Vec<bool> = data.iter().map(|&v| v == 0.0).collect(); // invert
        let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), bool_data)
            .unwrap()
            .into_shared();
        NdArrayTensor::from(array)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs) };
        let h = unsafe { npu_less(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        let (data, shape) = read_f32(h);
        unsafe { npu_free_tensor(h) };
        let bool_data: Vec<bool> = data.iter().map(|&v| v == 0.0).collect();
        let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), bool_data)
            .unwrap()
            .into_shared();
        NdArrayTensor::from(array)
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let h = unsafe { npu_less(lhs.handle, rhs.handle) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs) };
        let h = unsafe { npu_less(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        float_handle_to_bool_ndarray(h)
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        // lower_equal = NOT greater
        let h = unsafe { npu_greater(lhs.handle, rhs.handle) };
        let (data, shape) = read_f32(h);
        unsafe { npu_free_tensor(h) };
        let bool_data: Vec<bool> = data.iter().map(|&v| v == 0.0).collect();
        let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), bool_data)
            .unwrap()
            .into_shared();
        NdArrayTensor::from(array)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let rhs_h = unsafe { npu_scalar_tensor(rhs) };
        let h = unsafe { npu_greater(lhs.handle, rhs_h) };
        unsafe { npu_free_tensor(rhs_h) };
        let (data, shape) = read_f32(h);
        unsafe { npu_free_tensor(h) };
        let bool_data: Vec<bool> = data.iter().map(|&v| v == 0.0).collect();
        let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), bool_data)
            .unwrap()
            .into_shared();
        NdArrayTensor::from(array)
    }

    // ── Reductions ──────────────────────────────────────────────────────

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sum(tensor.handle) },
        }
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sum_dim(tensor.handle, dim as i32) },
        }
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mean_all(tensor.handle) },
        }
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_mean(tensor.handle, dim as i32) },
        }
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_prod(nd);
        ndarray_to_npu(&result)
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_prod_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cumsum(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cumprod(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cummin(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cummax(nd, dim);
        ndarray_to_npu(&result)
    }

    // ── Argmax / Argmin (return IntTensor = NdArrayTensor) ──────────────

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let h = unsafe { npu_argmax(tensor.handle, dim as i32) };
        int_handle_to_ndarray(h)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let h = unsafe { npu_argmin(tensor.handle, dim as i32) };
        int_handle_to_ndarray(h)
    }

    // ── Max / Min ───────────────────────────────────────────────────────

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_max(tensor.handle) },
        }
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_max_dim(tensor.handle, dim as i32) },
        }
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_min(tensor.handle) },
        }
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_min_dim(tensor.handle, dim as i32) },
        }
    }

    // ── Unary math ──────────────────────────────────────────────────────

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_exp(tensor.handle) },
        }
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_log(tensor.handle) },
        }
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // log1p(x) = log(1 + x)
        let one = unsafe { npu_scalar_tensor(1.0) };
        let sum = unsafe { npu_add(tensor.handle, one) };
        let result = unsafe { npu_log(sum) };
        unsafe {
            npu_free_tensor(one);
            npu_free_tensor(sum);
        }
        NpuFloatTensor { handle: result }
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_pow(lhs.handle, rhs.handle) },
        }
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_pow_scalar(tensor.handle, value) },
        }
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sqrt(tensor.handle) },
        }
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_abs(tensor.handle) },
        }
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_cos(tensor.handle) },
        }
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_sin(tensor.handle) },
        }
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_tanh(tensor.handle) },
        }
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_erf(tensor.handle) },
        }
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_floor(tensor.handle) },
        }
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_ceil(tensor.handle) },
        }
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_neg(tensor.handle) },
        }
    }

    // Trig ops without direct FFI — round-trip through NdArray
    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_tan(nd);
        ndarray_to_npu(&result)
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_acos(nd);
        ndarray_to_npu(&result)
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_acosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_asin(nd);
        ndarray_to_npu(&result)
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_asinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_atan(nd);
        ndarray_to_npu(&result)
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_atanh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_atan2(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_round(nd);
        ndarray_to_npu(&result)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_trunc(nd);
        ndarray_to_npu(&result)
    }

    // ── Clamp ───────────────────────────────────────────────────────────

    fn float_clamp_min(tensor: FloatTensor<Self>, min: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_clamp_min(tensor.handle, min) },
        }
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_clamp_max(tensor.handle, max) },
        }
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: f32, max: f32) -> FloatTensor<Self> {
        NpuFloatTensor {
            handle: unsafe { npu_clamp(tensor.handle, min, max) },
        }
    }

    // ── Cat ─────────────────────────────────────────────────────────────

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        let handles: Vec<i32> = tensors.iter().map(|t| t.handle).collect();
        NpuFloatTensor {
            handle: unsafe { npu_cat(handles.as_ptr(), handles.len() as i32, dim as i32) },
        }
    }

    // ── Sign ────────────────────────────────────────────────────────────

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sign(nd);
        ndarray_to_npu(&result)
    }

    // ── Cast ────────────────────────────────────────────────────────────

    fn float_cast(tensor: FloatTensor<Self>, _dtype: FloatDType) -> FloatTensor<Self> {
        // MLTensor only supports f32; casting is a no-op
        tensor
    }

    // ── Grid sample ─────────────────────────────────────────────────────

    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        options: GridSampleOptions,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_grid = npu_to_ndarray(&grid);
        let result = <Nd as FloatTensorOps<Nd>>::float_grid_sample_2d(nd_tensor, nd_grid, options);
        ndarray_to_npu(&result)
    }

    // ── Unfold ──────────────────────────────────────────────────────────

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_unfold(nd, dim, size, step);
        ndarray_to_npu(&result)
    }
}

// ===========================================================================
// FloatTensorOps — no feature: full NdArray delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl FloatTensorOps<Self> for NpuBurnBackend {
    fn float_from_data(data: TensorData, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_from_data(data, &nd_dev())
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NpuBurnDevice,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_random(shape, distribution, &nd_dev())
    }

    fn float_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_zeros(shape, &nd_dev(), dtype)
    }

    fn float_ones(shape: Shape, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_ones(shape, &nd_dev(), dtype)
    }

    fn float_full(
        shape: Shape,
        fill_value: f32,
        _device: &NpuBurnDevice,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_full(shape, fill_value, &nd_dev(), dtype)
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }
    fn float_to_device(tensor: FloatTensor<Self>, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        tensor
    }

    fn float_empty(shape: Shape, _device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_empty(shape, &nd_dev(), dtype)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as FloatTensorOps<Nd>>::float_into_data(tensor).await
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_matmul(lhs, rhs)
    }
    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cross(lhs, rhs, dim)
    }
    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_into_int(tensor)
    }
    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_add(lhs, rhs)
    }
    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_add_scalar(lhs, rhs)
    }
    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sub(lhs, rhs)
    }
    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sub_scalar(lhs, rhs)
    }
    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mul(lhs, rhs)
    }
    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mul_scalar(lhs, rhs)
    }
    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_div(lhs, rhs)
    }
    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_div_scalar(lhs, rhs)
    }
    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_remainder(lhs, rhs)
    }
    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_remainder_scalar(lhs, rhs)
    }
    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_recip(tensor)
    }
    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_swap_dims(tensor, dim1, dim2)
    }
    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_permute(tensor, axes)
    }
    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_flip(tensor, axes)
    }
    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_reshape(tensor, shape)
    }
    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_expand(tensor, shape)
    }
    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_gather(dim, tensor, indices)
    }
    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_scatter_add(dim, tensor, indices, value)
    }
    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_select(tensor, dim, indices)
    }
    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_select_add(tensor, dim, indices, value)
    }
    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_slice(tensor, slices)
    }
    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_slice_assign(tensor, slices, value)
    }
    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mask_where(tensor, mask, value)
    }
    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: f32,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mask_fill(tensor, mask, value)
    }
    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_equal(lhs, rhs)
    }
    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_equal_elem(lhs, rhs)
    }
    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_greater(lhs, rhs)
    }
    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_greater_elem(lhs, rhs)
    }
    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_greater_equal(lhs, rhs)
    }
    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_greater_equal_elem(lhs, rhs)
    }
    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_lower(lhs, rhs)
    }
    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_lower_elem(lhs, rhs)
    }
    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_lower_equal(lhs, rhs)
    }
    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_lower_equal_elem(lhs, rhs)
    }
    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sum(tensor)
    }
    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sum_dim(tensor, dim)
    }
    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mean(tensor)
    }
    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_mean_dim(tensor, dim)
    }
    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_prod(tensor)
    }
    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_prod_dim(tensor, dim)
    }
    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cumsum(tensor, dim)
    }
    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cumprod(tensor, dim)
    }
    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cummin(tensor, dim)
    }
    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cummax(tensor, dim)
    }
    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_argmax(tensor, dim)
    }
    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_argmin(tensor, dim)
    }
    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_exp(tensor)
    }
    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_log(tensor)
    }
    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_log1p(tensor)
    }
    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_powf(lhs, rhs)
    }
    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_powf_scalar_impl(tensor, value)
    }
    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sqrt(tensor)
    }
    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_abs(tensor)
    }
    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cos(tensor)
    }
    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sin(tensor)
    }
    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_tan(tensor)
    }
    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cosh(tensor)
    }
    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sinh(tensor)
    }
    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_tanh(tensor)
    }
    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_acos(tensor)
    }
    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_acosh(tensor)
    }
    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_asin(tensor)
    }
    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_asinh(tensor)
    }
    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_atan(tensor)
    }
    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_atanh(tensor)
    }
    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_atan2(lhs, rhs)
    }
    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_round(tensor)
    }
    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_floor(tensor)
    }
    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_ceil(tensor)
    }
    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_trunc(tensor)
    }
    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_erf(tensor)
    }
    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cat(tensors, dim)
    }
    fn float_clamp_min(tensor: FloatTensor<Self>, min: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_clamp_min(tensor, min)
    }
    fn float_clamp_max(tensor: FloatTensor<Self>, max: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_clamp_max(tensor, max)
    }
    fn float_clamp(tensor: FloatTensor<Self>, min: f32, max: f32) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_clamp(tensor, min, max)
    }
    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_neg(tensor)
    }
    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_sign(tensor)
    }
    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_cast(tensor, dtype)
    }
    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        options: GridSampleOptions,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_grid_sample_2d(tensor, grid, options)
    }
    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_unfold(tensor, dim, size, step)
    }
    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_max(tensor)
    }
    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_max_dim(tensor, dim)
    }
    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_min(tensor)
    }
    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        <Nd as FloatTensorOps<Nd>>::float_min_dim(tensor, dim)
    }
}

// ===========================================================================
// FloatTensorOps — intel/qualcomm: Vec<f32> tensor, NdArray delegation
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl FloatTensorOps<Self> for NpuBurnBackend {
    fn float_from_data(data: TensorData, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        let floats: Vec<f32> = data.to_vec().unwrap();
        let shape: Vec<usize> = data.shape.to_vec();
        NpuFloatTensor::new(floats, shape)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NpuBurnDevice,
    ) -> FloatTensor<Self> {
        let nd_tensor = <Nd as FloatTensorOps<Nd>>::float_random(shape, distribution, &nd_dev());
        ndarray_to_npu(&nd_tensor)
    }

    fn float_zeros(shape: Shape, _device: &NpuBurnDevice, _dtype: FloatDType) -> FloatTensor<Self> {
        NpuFloatTensor::zeros(shape.dims.to_vec())
    }

    fn float_ones(shape: Shape, _device: &NpuBurnDevice, _dtype: FloatDType) -> FloatTensor<Self> {
        NpuFloatTensor::ones(shape.dims.to_vec())
    }

    fn float_full(
        shape: Shape,
        fill_value: f32,
        _device: &NpuBurnDevice,
        _dtype: FloatDType,
    ) -> FloatTensor<Self> {
        NpuFloatTensor::full(shape.dims.to_vec(), fill_value)
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }

    fn float_to_device(tensor: FloatTensor<Self>, _device: &NpuBurnDevice) -> FloatTensor<Self> {
        tensor
    }

    fn float_empty(shape: Shape, device: &NpuBurnDevice, dtype: FloatDType) -> FloatTensor<Self> {
        Self::float_zeros(shape, device, dtype)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        let shape = burn_tensor::TensorMetadata::shape(&tensor);
        Ok(TensorData::new(tensor.data, shape))
    }

    // ── Matmul ──────────────────────────────────────────────────────────

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        // Intel: try OpenVINO NPU for large matmuls, else CPU.
        // Takes precedence if both features are somehow enabled at once.
        #[cfg(feature = "intel")]
        {
            crate::backends::intel::openvino_matmul(&lhs, &rhs)
                .unwrap_or_else(|_| crate::backends::intel::cpu_matmul(&lhs, &rhs))
        }

        // Qualcomm: CPU matmul (TODO: QNN HTP dispatch)
        #[cfg(all(feature = "qualcomm", not(feature = "intel")))]
        {
            crate::backends::qualcomm::cpu_matmul(&lhs, &rhs)
        }
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_cross(nd_lhs, nd_rhs, dim);
        ndarray_to_npu(&result)
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        <Nd as FloatTensorOps<Nd>>::float_into_int(nd)
    }

    // ── Arithmetic ──────────────────────────────────────────────────────

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_add(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_add_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_sub(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_sub_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_mul(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_mul_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_div(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_div_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_remainder(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_remainder_scalar(nd, rhs);
        ndarray_to_npu(&result)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_recip(nd);
        ndarray_to_npu(&result)
    }

    // ── Shape / layout ──────────────────────────────────────────────────

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_swap_dims(nd, dim1, dim2);
        ndarray_to_npu(&result)
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_permute(nd, axes);
        ndarray_to_npu(&result)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_flip(nd, axes);
        ndarray_to_npu(&result)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_reshape(nd, shape);
        ndarray_to_npu(&result)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_expand(nd, shape);
        ndarray_to_npu(&result)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_gather(dim, nd, indices);
        ndarray_to_npu(&result)
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result =
            <Nd as FloatTensorOps<Nd>>::float_scatter_add(dim, nd_tensor, indices, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_select(nd, dim, indices);
        ndarray_to_npu(&result)
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result =
            <Nd as FloatTensorOps<Nd>>::float_select_add(nd_tensor, dim, indices, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_slice(nd, slices);
        ndarray_to_npu(&result)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Nd as FloatTensorOps<Nd>>::float_slice_assign(nd_tensor, slices, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_value = npu_to_ndarray(&value);
        let result = <Nd as FloatTensorOps<Nd>>::float_mask_where(nd_tensor, mask, nd_value);
        ndarray_to_npu(&result)
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: f32,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_mask_fill(nd_tensor, mask, value);
        ndarray_to_npu(&result)
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as FloatTensorOps<Nd>>::float_equal(nd_lhs, nd_rhs)
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Nd as FloatTensorOps<Nd>>::float_equal_elem(nd, rhs)
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as FloatTensorOps<Nd>>::float_greater(nd_lhs, nd_rhs)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Nd as FloatTensorOps<Nd>>::float_greater_elem(nd, rhs)
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as FloatTensorOps<Nd>>::float_greater_equal(nd_lhs, nd_rhs)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Nd as FloatTensorOps<Nd>>::float_greater_equal_elem(nd, rhs)
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as FloatTensorOps<Nd>>::float_lower(nd_lhs, nd_rhs)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Nd as FloatTensorOps<Nd>>::float_lower_elem(nd, rhs)
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as FloatTensorOps<Nd>>::float_lower_equal(nd_lhs, nd_rhs)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: f32) -> BoolTensor<Self> {
        let nd = npu_to_ndarray(&lhs);
        <Nd as FloatTensorOps<Nd>>::float_lower_equal_elem(nd, rhs)
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sum(nd);
        ndarray_to_npu(&result)
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sum_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_mean(nd);
        ndarray_to_npu(&result)
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_mean_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_prod(nd);
        ndarray_to_npu(&result)
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_prod_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cumsum(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cumprod(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cummin(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cummax(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        <Nd as FloatTensorOps<Nd>>::float_argmax(nd, dim)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        <Nd as FloatTensorOps<Nd>>::float_argmin(nd, dim)
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_max(nd);
        ndarray_to_npu(&result)
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_max_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_min(nd);
        ndarray_to_npu(&result)
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_min_dim(nd, dim);
        ndarray_to_npu(&result)
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_exp(nd);
        ndarray_to_npu(&result)
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_log(nd);
        ndarray_to_npu(&result)
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_log1p(nd);
        ndarray_to_npu(&result)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_powf(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_powf_scalar_impl(nd, value);
        ndarray_to_npu(&result)
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sqrt(nd);
        ndarray_to_npu(&result)
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_abs(nd);
        ndarray_to_npu(&result)
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cos(nd);
        ndarray_to_npu(&result)
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sin(nd);
        ndarray_to_npu(&result)
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_tanh(nd);
        ndarray_to_npu(&result)
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_erf(nd);
        ndarray_to_npu(&result)
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_floor(nd);
        ndarray_to_npu(&result)
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_ceil(nd);
        ndarray_to_npu(&result)
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_neg(nd);
        ndarray_to_npu(&result)
    }

    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_tan(nd);
        ndarray_to_npu(&result)
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_cosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_acos(nd);
        ndarray_to_npu(&result)
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_acosh(nd);
        ndarray_to_npu(&result)
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_asin(nd);
        ndarray_to_npu(&result)
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_asinh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_atan(nd);
        ndarray_to_npu(&result)
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_atanh(nd);
        ndarray_to_npu(&result)
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd_lhs = npu_to_ndarray(&lhs);
        let nd_rhs = npu_to_ndarray(&rhs);
        let result = <Nd as FloatTensorOps<Nd>>::float_atan2(nd_lhs, nd_rhs);
        ndarray_to_npu(&result)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_round(nd);
        ndarray_to_npu(&result)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_trunc(nd);
        ndarray_to_npu(&result)
    }

    fn float_clamp_min(tensor: FloatTensor<Self>, min: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_clamp_min(nd, min);
        ndarray_to_npu(&result)
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_clamp_max(nd, max);
        ndarray_to_npu(&result)
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: f32, max: f32) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_clamp(nd, min, max);
        ndarray_to_npu(&result)
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        let nd_tensors: Vec<_> = tensors.iter().map(npu_to_ndarray).collect();
        let result = <Nd as FloatTensorOps<Nd>>::float_cat(nd_tensors, dim);
        ndarray_to_npu(&result)
    }

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_sign(nd);
        ndarray_to_npu(&result)
    }

    fn float_cast(tensor: FloatTensor<Self>, _dtype: FloatDType) -> FloatTensor<Self> {
        // Only f32 supported; casting is a no-op.
        tensor
    }

    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        options: GridSampleOptions,
    ) -> FloatTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_grid = npu_to_ndarray(&grid);
        let result = <Nd as FloatTensorOps<Nd>>::float_grid_sample_2d(nd_tensor, nd_grid, options);
        ndarray_to_npu(&result)
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        let nd = npu_to_ndarray(&tensor);
        let result = <Nd as FloatTensorOps<Nd>>::float_unfold(nd, dim, size, step);
        ndarray_to_npu(&result)
    }
}
