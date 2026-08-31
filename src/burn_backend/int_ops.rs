//! `IntTensorOps` implementations for all platform variants.

use burn_tensor::backend::ExecutionError;
use burn_tensor::ops::*;
use burn_tensor::ops::{BoolTensor, FloatTensor, IntTensor};
use burn_tensor::{Distribution, IntDType, Shape, Slice, TensorData};

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
use super::tensor::*;
use super::{nd_dev, Nd, NpuBurnBackend, NpuBurnDevice};

// ===========================================================================
// IntTensorOps — apple: Int is NdArrayTensor, Float is NpuFloatTensor
// ===========================================================================
#[cfg(feature = "apple")]
impl IntTensorOps<Self> for NpuBurnBackend {
    fn int_from_data(data: TensorData, _device: &NpuBurnDevice) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_from_data(data, &nd_dev())
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as IntTensorOps<Nd>>::int_into_data(tensor).await
    }

    fn int_device(_tensor: &IntTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }
    fn int_to_device(tensor: IntTensor<Self>, _device: &NpuBurnDevice) -> IntTensor<Self> {
        tensor
    }

    fn int_empty(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_empty(shape, &nd_dev(), dtype)
    }
    fn int_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_zeros(shape, &nd_dev(), dtype)
    }
    fn int_ones(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_ones(shape, &nd_dev(), dtype)
    }
    fn int_full(
        shape: Shape,
        fill_value: i64,
        _device: &NpuBurnDevice,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_full(shape, fill_value, &nd_dev(), dtype)
    }
    fn int_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NpuBurnDevice,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_random(shape, distribution, &nd_dev())
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_reshape(tensor, shape)
    }
    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice(tensor, slices)
    }
    fn int_slice_assign(
        tensor: IntTensor<Self>,
        slices: &[Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice_assign(tensor, slices, value)
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        let nd_float = <Nd as IntTensorOps<Nd>>::int_into_float(tensor);
        ndarray_to_npu(&nd_float)
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_where(tensor, mask, value)
    }
    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: i64,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_fill(tensor, mask, value)
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_gather(dim, tensor, indices)
    }
    fn int_scatter_add(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_scatter_add(dim, tensor, indices, value)
    }
    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select(tensor, dim, indices)
    }
    fn int_select_add(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select_add(tensor, dim, indices, value)
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cat(tensors, dim)
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal(lhs, rhs)
    }
    fn int_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal_elem(lhs, rhs)
    }
    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater(lhs, rhs)
    }
    fn int_greater_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_elem(lhs, rhs)
    }
    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal(lhs, rhs)
    }
    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal_elem(lhs, rhs)
    }
    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower(lhs, rhs)
    }
    fn int_lower_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_elem(lhs, rhs)
    }
    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal(lhs, rhs)
    }
    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal_elem(lhs, rhs)
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add(lhs, rhs)
    }
    fn int_add_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add_scalar(lhs, rhs)
    }
    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub(lhs, rhs)
    }
    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub_scalar(lhs, rhs)
    }
    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul(lhs, rhs)
    }
    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul_scalar(lhs, rhs)
    }
    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div(lhs, rhs)
    }
    fn int_div_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div_scalar(lhs, rhs)
    }
    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder(lhs, rhs)
    }
    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder_scalar(lhs, rhs)
    }

    fn int_neg(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_neg(tensor)
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum(tensor)
    }
    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum_dim(tensor, dim)
    }
    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod(tensor)
    }
    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod_dim(tensor, dim)
    }
    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean(tensor)
    }
    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean_dim(tensor, dim)
    }
    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_max(tensor)
    }
    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_min(tensor)
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumsum(tensor, dim)
    }
    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumprod(tensor, dim)
    }
    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummin(tensor, dim)
    }
    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummax(tensor, dim)
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_matmul(lhs, rhs)
    }
    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmax(tensor, dim)
    }
    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmin(tensor, dim)
    }
    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_abs(tensor)
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_swap_dims(tensor, dim1, dim2)
    }
    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_permute(tensor, axes)
    }
    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_flip(tensor, axes)
    }
    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_expand(tensor, shape)
    }

    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sign(tensor)
    }
    fn int_powi(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powi(lhs, rhs)
    }

    fn int_powf(lhs: IntTensor<Self>, rhs: FloatTensor<Self>) -> IntTensor<Self> {
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as IntTensorOps<Nd>>::int_powf(lhs, nd_rhs)
    }

    fn int_powf_scalar_impl(lhs: IntTensor<Self>, rhs: f32) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powf_scalar_impl(lhs, rhs)
    }

    fn int_clamp_min(tensor: IntTensor<Self>, min: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_min(tensor, min)
    }
    fn int_clamp_max(tensor: IntTensor<Self>, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_max(tensor, max)
    }
    fn int_clamp(tensor: IntTensor<Self>, min: i64, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp(tensor, min, max)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and(lhs, rhs)
    }
    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and_scalar(lhs, rhs)
    }
    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or(lhs, rhs)
    }
    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or_scalar(lhs, rhs)
    }
    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor(lhs, rhs)
    }
    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor_scalar(lhs, rhs)
    }
    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_not(tensor)
    }
    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift(lhs, rhs)
    }
    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift_scalar(lhs, rhs)
    }
    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift(lhs, rhs)
    }
    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift_scalar(lhs, rhs)
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cast(tensor, dtype)
    }
    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_unfold(tensor, dim, size, step)
    }
}

// ===========================================================================
// IntTensorOps — no feature: full NdArray delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl IntTensorOps<Self> for NpuBurnBackend {
    fn int_from_data(data: TensorData, _device: &NpuBurnDevice) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_from_data(data, &nd_dev())
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as IntTensorOps<Nd>>::int_into_data(tensor).await
    }

    fn int_device(_tensor: &IntTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }
    fn int_to_device(tensor: IntTensor<Self>, _device: &NpuBurnDevice) -> IntTensor<Self> {
        tensor
    }

    fn int_empty(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_empty(shape, &nd_dev(), dtype)
    }
    fn int_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_zeros(shape, &nd_dev(), dtype)
    }
    fn int_ones(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_ones(shape, &nd_dev(), dtype)
    }
    fn int_full(
        shape: Shape,
        fill_value: i64,
        _device: &NpuBurnDevice,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_full(shape, fill_value, &nd_dev(), dtype)
    }
    fn int_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NpuBurnDevice,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_random(shape, distribution, &nd_dev())
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_reshape(tensor, shape)
    }
    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice(tensor, slices)
    }
    fn int_slice_assign(
        tensor: IntTensor<Self>,
        slices: &[Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice_assign(tensor, slices, value)
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_into_float(tensor)
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_where(tensor, mask, value)
    }
    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: i64,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_fill(tensor, mask, value)
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_gather(dim, tensor, indices)
    }
    fn int_scatter_add(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_scatter_add(dim, tensor, indices, value)
    }
    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select(tensor, dim, indices)
    }
    fn int_select_add(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select_add(tensor, dim, indices, value)
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cat(tensors, dim)
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal(lhs, rhs)
    }
    fn int_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal_elem(lhs, rhs)
    }
    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater(lhs, rhs)
    }
    fn int_greater_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_elem(lhs, rhs)
    }
    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal(lhs, rhs)
    }
    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal_elem(lhs, rhs)
    }
    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower(lhs, rhs)
    }
    fn int_lower_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_elem(lhs, rhs)
    }
    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal(lhs, rhs)
    }
    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal_elem(lhs, rhs)
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add(lhs, rhs)
    }
    fn int_add_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add_scalar(lhs, rhs)
    }
    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub(lhs, rhs)
    }
    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub_scalar(lhs, rhs)
    }
    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul(lhs, rhs)
    }
    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul_scalar(lhs, rhs)
    }
    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div(lhs, rhs)
    }
    fn int_div_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div_scalar(lhs, rhs)
    }
    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder(lhs, rhs)
    }
    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder_scalar(lhs, rhs)
    }

    fn int_neg(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_neg(tensor)
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum(tensor)
    }
    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum_dim(tensor, dim)
    }
    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod(tensor)
    }
    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod_dim(tensor, dim)
    }
    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean(tensor)
    }
    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean_dim(tensor, dim)
    }
    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_max(tensor)
    }
    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_min(tensor)
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumsum(tensor, dim)
    }
    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumprod(tensor, dim)
    }
    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummin(tensor, dim)
    }
    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummax(tensor, dim)
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_matmul(lhs, rhs)
    }
    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmax(tensor, dim)
    }
    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmin(tensor, dim)
    }
    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_abs(tensor)
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_swap_dims(tensor, dim1, dim2)
    }
    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_permute(tensor, axes)
    }
    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_flip(tensor, axes)
    }
    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_expand(tensor, shape)
    }

    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sign(tensor)
    }
    fn int_powi(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powi(lhs, rhs)
    }

    fn int_powf(lhs: IntTensor<Self>, rhs: FloatTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powf(lhs, rhs)
    }

    fn int_powf_scalar_impl(lhs: IntTensor<Self>, rhs: f32) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powf_scalar_impl(lhs, rhs)
    }

    fn int_clamp_min(tensor: IntTensor<Self>, min: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_min(tensor, min)
    }
    fn int_clamp_max(tensor: IntTensor<Self>, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_max(tensor, max)
    }
    fn int_clamp(tensor: IntTensor<Self>, min: i64, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp(tensor, min, max)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and(lhs, rhs)
    }
    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and_scalar(lhs, rhs)
    }
    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or(lhs, rhs)
    }
    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or_scalar(lhs, rhs)
    }
    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor(lhs, rhs)
    }
    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor_scalar(lhs, rhs)
    }
    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_not(tensor)
    }
    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift(lhs, rhs)
    }
    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift_scalar(lhs, rhs)
    }
    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift(lhs, rhs)
    }
    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift_scalar(lhs, rhs)
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cast(tensor, dtype)
    }
    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_unfold(tensor, dim, size, step)
    }
}

// ===========================================================================
// IntTensorOps — intel/qualcomm: Int is NdArrayTensor, Float is NpuFloatTensor
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl IntTensorOps<Self> for NpuBurnBackend {
    fn int_from_data(data: TensorData, _device: &NpuBurnDevice) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_from_data(data, &nd_dev())
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as IntTensorOps<Nd>>::int_into_data(tensor).await
    }

    fn int_device(_tensor: &IntTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }
    fn int_to_device(tensor: IntTensor<Self>, _device: &NpuBurnDevice) -> IntTensor<Self> {
        tensor
    }

    fn int_empty(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_empty(shape, &nd_dev(), dtype)
    }
    fn int_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_zeros(shape, &nd_dev(), dtype)
    }
    fn int_ones(shape: Shape, _device: &NpuBurnDevice, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_ones(shape, &nd_dev(), dtype)
    }
    fn int_full(
        shape: Shape,
        fill_value: i64,
        _device: &NpuBurnDevice,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_full(shape, fill_value, &nd_dev(), dtype)
    }
    fn int_random(
        shape: Shape,
        distribution: Distribution,
        _device: &NpuBurnDevice,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_random(shape, distribution, &nd_dev())
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_reshape(tensor, shape)
    }
    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice(tensor, slices)
    }
    fn int_slice_assign(
        tensor: IntTensor<Self>,
        slices: &[Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_slice_assign(tensor, slices, value)
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        let nd_float = <Nd as IntTensorOps<Nd>>::int_into_float(tensor);
        ndarray_to_npu(&nd_float)
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_where(tensor, mask, value)
    }
    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: i64,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mask_fill(tensor, mask, value)
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_gather(dim, tensor, indices)
    }
    fn int_scatter_add(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_scatter_add(dim, tensor, indices, value)
    }
    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select(tensor, dim, indices)
    }
    fn int_select_add(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_select_add(tensor, dim, indices, value)
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cat(tensors, dim)
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal(lhs, rhs)
    }
    fn int_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_equal_elem(lhs, rhs)
    }
    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater(lhs, rhs)
    }
    fn int_greater_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_elem(lhs, rhs)
    }
    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal(lhs, rhs)
    }
    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_greater_equal_elem(lhs, rhs)
    }
    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower(lhs, rhs)
    }
    fn int_lower_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_elem(lhs, rhs)
    }
    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal(lhs, rhs)
    }
    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: i64) -> BoolTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_lower_equal_elem(lhs, rhs)
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add(lhs, rhs)
    }
    fn int_add_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_add_scalar(lhs, rhs)
    }
    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub(lhs, rhs)
    }
    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sub_scalar(lhs, rhs)
    }
    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul(lhs, rhs)
    }
    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mul_scalar(lhs, rhs)
    }
    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div(lhs, rhs)
    }
    fn int_div_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_div_scalar(lhs, rhs)
    }
    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder(lhs, rhs)
    }
    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_remainder_scalar(lhs, rhs)
    }

    fn int_neg(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_neg(tensor)
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum(tensor)
    }
    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sum_dim(tensor, dim)
    }
    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod(tensor)
    }
    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_prod_dim(tensor, dim)
    }
    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean(tensor)
    }
    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_mean_dim(tensor, dim)
    }
    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_max(tensor)
    }
    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_min(tensor)
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumsum(tensor, dim)
    }
    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cumprod(tensor, dim)
    }
    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummin(tensor, dim)
    }
    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cummax(tensor, dim)
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_matmul(lhs, rhs)
    }
    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmax(tensor, dim)
    }
    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_argmin(tensor, dim)
    }
    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_abs(tensor)
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_swap_dims(tensor, dim1, dim2)
    }
    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_permute(tensor, axes)
    }
    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_flip(tensor, axes)
    }
    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_expand(tensor, shape)
    }

    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_sign(tensor)
    }
    fn int_powi(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powi(lhs, rhs)
    }

    fn int_powf(lhs: IntTensor<Self>, rhs: FloatTensor<Self>) -> IntTensor<Self> {
        let nd_rhs = npu_to_ndarray(&rhs);
        <Nd as IntTensorOps<Nd>>::int_powf(lhs, nd_rhs)
    }

    fn int_powf_scalar_impl(lhs: IntTensor<Self>, rhs: f32) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_powf_scalar_impl(lhs, rhs)
    }

    fn int_clamp_min(tensor: IntTensor<Self>, min: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_min(tensor, min)
    }
    fn int_clamp_max(tensor: IntTensor<Self>, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp_max(tensor, max)
    }
    fn int_clamp(tensor: IntTensor<Self>, min: i64, max: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_clamp(tensor, min, max)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and(lhs, rhs)
    }
    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_and_scalar(lhs, rhs)
    }
    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or(lhs, rhs)
    }
    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_or_scalar(lhs, rhs)
    }
    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor(lhs, rhs)
    }
    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_xor_scalar(lhs, rhs)
    }
    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_not(tensor)
    }
    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift(lhs, rhs)
    }
    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_left_shift_scalar(lhs, rhs)
    }
    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift(lhs, rhs)
    }
    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: i64) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::bitwise_right_shift_scalar(lhs, rhs)
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_cast(tensor, dtype)
    }
    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        <Nd as IntTensorOps<Nd>>::int_unfold(tensor, dim, size, step)
    }
}
