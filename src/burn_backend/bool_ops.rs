//! `BoolTensorOps` implementations for all platform variants.

use burn_tensor::backend::ExecutionError;
use burn_tensor::ops::*;
use burn_tensor::ops::{BoolTensor, FloatTensor, IntTensor};
use burn_tensor::{BoolDType, FloatDType, IntDType, Scalar, Shape, Slice, TensorData};

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
use super::tensor::*;
use super::{flex_dev, Fx, NpuBurnBackend, NpuBurnDevice};

// ===========================================================================
// BoolTensorOps — apple: bool_into_float returns NpuFloatTensor
// ===========================================================================
#[cfg(feature = "apple")]
impl BoolTensorOps<Self> for NpuBurnBackend {
    fn bool_from_data(data: TensorData, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_from_data(data, &flex_dev())
    }
    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Fx as BoolTensorOps<Fx>>::bool_into_data(tensor).await
    }
    fn bool_device(_tensor: &BoolTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }
    fn bool_to_device(tensor: BoolTensor<Self>, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        tensor
    }
    fn bool_empty(shape: Shape, _device: &NpuBurnDevice, dtype: BoolDType) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_empty(shape, &flex_dev(), dtype)
    }
    fn bool_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: BoolDType) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_zeros(shape, &flex_dev(), dtype)
    }
    fn bool_ones(shape: Shape, _device: &NpuBurnDevice, dtype: BoolDType) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_ones(shape, &flex_dev(), dtype)
    }
    fn bool_into_int(tensor: BoolTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_into_int(tensor, out_dtype)
    }

    fn bool_into_float(tensor: BoolTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        // Convert FlexTensor<bool> -> f32 -> NpuFloatTensor
        let nd_float = <Fx as BoolTensorOps<Fx>>::bool_into_float(tensor, out_dtype);
        ndarray_to_npu(&nd_float)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_reshape(tensor, shape)
    }
    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_slice(tensor, slices)
    }
    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_slice_assign(tensor, slices, value)
    }
    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_equal(lhs, rhs)
    }
    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_not(tensor)
    }
    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_and(lhs, rhs)
    }
    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_or(lhs, rhs)
    }
    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_swap_dims(tensor, dim1, dim2)
    }
    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_permute(tensor, axes)
    }
    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_flip(tensor, axes)
    }
    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_expand(tensor, shape)
    }
    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_cat(tensors, dim)
    }
    fn bool_select(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_select(tensor, dim, indices)
    }
    fn bool_select_or(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_select_or(tensor, dim, indices, value)
    }
    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_unfold(tensor, dim, size, step)
    }
    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_mask_where(tensor, mask, value)
    }
    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_mask_fill(tensor, mask, value)
    }
    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_gather(dim, tensor, indices)
    }
    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_scatter_or(dim, tensor, indices, value)
    }
    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_equal_elem(lhs, rhs)
    }
    fn bool_any(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_any(tensor)
    }
    fn bool_all(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_all(tensor)
    }
}

// ===========================================================================
// BoolTensorOps — no feature: full Flex delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl BoolTensorOps<Self> for NpuBurnBackend {
    fn bool_from_data(data: TensorData, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_from_data(data, &flex_dev())
    }
    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Fx as BoolTensorOps<Fx>>::bool_into_data(tensor).await
    }
    fn bool_device(_tensor: &BoolTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }
    fn bool_to_device(tensor: BoolTensor<Self>, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        tensor
    }
    fn bool_empty(shape: Shape, _device: &NpuBurnDevice, dtype: BoolDType) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_empty(shape, &flex_dev(), dtype)
    }
    fn bool_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: BoolDType) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_zeros(shape, &flex_dev(), dtype)
    }
    fn bool_ones(shape: Shape, _device: &NpuBurnDevice, dtype: BoolDType) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_ones(shape, &flex_dev(), dtype)
    }
    fn bool_into_int(tensor: BoolTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_into_int(tensor, out_dtype)
    }
    fn bool_into_float(tensor: BoolTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_into_float(tensor, out_dtype)
    }
    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_reshape(tensor, shape)
    }
    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_slice(tensor, slices)
    }
    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_slice_assign(tensor, slices, value)
    }
    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_equal(lhs, rhs)
    }
    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_not(tensor)
    }
    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_and(lhs, rhs)
    }
    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_or(lhs, rhs)
    }
    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_swap_dims(tensor, dim1, dim2)
    }
    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_permute(tensor, axes)
    }
    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_flip(tensor, axes)
    }
    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_expand(tensor, shape)
    }
    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_cat(tensors, dim)
    }
    fn bool_select(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_select(tensor, dim, indices)
    }
    fn bool_select_or(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_select_or(tensor, dim, indices, value)
    }
    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_unfold(tensor, dim, size, step)
    }
    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_mask_where(tensor, mask, value)
    }
    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_mask_fill(tensor, mask, value)
    }
    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_gather(dim, tensor, indices)
    }
    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_scatter_or(dim, tensor, indices, value)
    }
    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_equal_elem(lhs, rhs)
    }
    fn bool_any(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_any(tensor)
    }
    fn bool_all(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_all(tensor)
    }
}

// ===========================================================================
// BoolTensorOps — intel/qualcomm: bool_into_float bridges to NpuFloatTensor
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl BoolTensorOps<Self> for NpuBurnBackend {
    fn bool_from_data(data: TensorData, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_from_data(data, &flex_dev())
    }
    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Fx as BoolTensorOps<Fx>>::bool_into_data(tensor).await
    }
    fn bool_device(_tensor: &BoolTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }
    fn bool_to_device(tensor: BoolTensor<Self>, _device: &NpuBurnDevice) -> BoolTensor<Self> {
        tensor
    }
    fn bool_empty(shape: Shape, _device: &NpuBurnDevice, dtype: BoolDType) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_empty(shape, &flex_dev(), dtype)
    }
    fn bool_zeros(shape: Shape, _device: &NpuBurnDevice, dtype: BoolDType) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_zeros(shape, &flex_dev(), dtype)
    }
    fn bool_ones(shape: Shape, _device: &NpuBurnDevice, dtype: BoolDType) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_ones(shape, &flex_dev(), dtype)
    }
    fn bool_into_int(tensor: BoolTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_into_int(tensor, out_dtype)
    }

    fn bool_into_float(tensor: BoolTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        // Convert FlexTensor<bool> -> FlexTensor<f32> -> NpuFloatTensor
        let nd_float = <Fx as BoolTensorOps<Fx>>::bool_into_float(tensor, out_dtype);
        ndarray_to_npu(&nd_float)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_reshape(tensor, shape)
    }
    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_slice(tensor, slices)
    }
    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_slice_assign(tensor, slices, value)
    }
    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_equal(lhs, rhs)
    }
    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_not(tensor)
    }
    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_and(lhs, rhs)
    }
    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_or(lhs, rhs)
    }
    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_swap_dims(tensor, dim1, dim2)
    }
    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_permute(tensor, axes)
    }
    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_flip(tensor, axes)
    }
    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_expand(tensor, shape)
    }
    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_cat(tensors, dim)
    }
    fn bool_select(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_select(tensor, dim, indices)
    }
    fn bool_select_or(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_select_or(tensor, dim, indices, value)
    }
    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_unfold(tensor, dim, size, step)
    }
    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_mask_where(tensor, mask, value)
    }
    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_mask_fill(tensor, mask, value)
    }
    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_gather(dim, tensor, indices)
    }
    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_scatter_or(dim, tensor, indices, value)
    }
    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_equal_elem(lhs, rhs)
    }
    fn bool_any(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_any(tensor)
    }
    fn bool_all(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        <Fx as BoolTensorOps<Fx>>::bool_all(tensor)
    }
}
