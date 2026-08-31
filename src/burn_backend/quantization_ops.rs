//! `QTensorOps` implementations for all platform variants.

use burn_tensor::backend::ExecutionError;
use burn_tensor::ops::*;
use burn_tensor::ops::{FloatTensor, IntTensor, QuantizedTensor};
use burn_tensor::quantization::{QuantScheme, QuantizationParametersPrimitive};
use burn_tensor::{Shape, Slice, TensorData};

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
use super::tensor::*;
use super::{nd_dev, Nd, NpuBurnBackend, NpuBurnDevice};

// ===========================================================================
// QTensorOps — apple: quantize/dequantize bridge NpuFloatTensor <-> NdArray
// ===========================================================================
#[cfg(feature = "apple")]
impl QTensorOps<Self> for NpuBurnBackend {
    fn q_from_data(data: TensorData, _device: &NpuBurnDevice) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_from_data(data, &nd_dev())
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        // Convert NpuFloatTensor -> NdArrayTensor for NdArray's quantize
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_scales = npu_to_ndarray(&qparams.scales);
        let nd_qparams = QuantizationParametersPrimitive::<Nd> { scales: nd_scales };
        <Nd as QTensorOps<Nd>>::quantize(nd_tensor, scheme, nd_qparams)
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        let nd_result = <Nd as QTensorOps<Nd>>::dequantize(tensor);
        ndarray_to_npu(&nd_result)
    }

    fn q_device(_tensor: &QuantizedTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }

    fn q_to_device(
        tensor: QuantizedTensor<Self>,
        _device: &NpuBurnDevice,
    ) -> QuantizedTensor<Self> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as QTensorOps<Nd>>::q_into_data(tensor).await
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_flip(tensor, axes)
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_gather(dim, tensor, indices)
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_select(tensor, dim, indices)
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_slice(tensor, slices)
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmax(tensor, dim)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmin(tensor, dim)
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_expand(tensor, shape)
    }
}

// ===========================================================================
// QTensorOps — no feature: full delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl QTensorOps<Self> for NpuBurnBackend {
    fn q_from_data(data: TensorData, _device: &NpuBurnDevice) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_from_data(data, &nd_dev())
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        let nd_qparams = QuantizationParametersPrimitive::<Nd> {
            scales: qparams.scales,
        };
        <Nd as QTensorOps<Nd>>::quantize(tensor, scheme, nd_qparams)
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        <Nd as QTensorOps<Nd>>::dequantize(tensor)
    }

    fn q_device(_tensor: &QuantizedTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }

    fn q_to_device(
        tensor: QuantizedTensor<Self>,
        _device: &NpuBurnDevice,
    ) -> QuantizedTensor<Self> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as QTensorOps<Nd>>::q_into_data(tensor).await
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_flip(tensor, axes)
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_gather(dim, tensor, indices)
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_select(tensor, dim, indices)
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_slice(tensor, slices)
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmax(tensor, dim)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmin(tensor, dim)
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_expand(tensor, shape)
    }
}

// ===========================================================================
// QTensorOps — intel/qualcomm: quantize/dequantize bridge NpuFloatTensor
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl QTensorOps<Self> for NpuBurnBackend {
    fn q_from_data(data: TensorData, _device: &NpuBurnDevice) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_from_data(data, &nd_dev())
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        let nd_tensor = npu_to_ndarray(&tensor);
        let nd_scales = npu_to_ndarray(&qparams.scales);
        let nd_qparams = QuantizationParametersPrimitive::<Nd> { scales: nd_scales };
        <Nd as QTensorOps<Nd>>::quantize(nd_tensor, scheme, nd_qparams)
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        let nd_result = <Nd as QTensorOps<Nd>>::dequantize(tensor);
        ndarray_to_npu(&nd_result)
    }

    fn q_device(_tensor: &QuantizedTensor<Self>) -> NpuBurnDevice {
        NpuBurnDevice::Default
    }

    fn q_to_device(
        tensor: QuantizedTensor<Self>,
        _device: &NpuBurnDevice,
    ) -> QuantizedTensor<Self> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        <Nd as QTensorOps<Nd>>::q_into_data(tensor).await
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_flip(tensor, axes)
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_gather(dim, tensor, indices)
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_select(tensor, dim, indices)
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_slice(tensor, slices)
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmax(tensor, dim)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_argmin(tensor, dim)
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        <Nd as QTensorOps<Nd>>::q_expand(tensor, shape)
    }
}
