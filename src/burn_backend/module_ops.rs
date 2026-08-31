//! `ModuleOps` implementations for all platform variants.

use burn_tensor::ops::*;
use burn_tensor::ops::{FloatTensor, IntTensor};

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
use super::tensor::*;
use super::{Nd, NpuBurnBackend};

// ===========================================================================
// ModuleOps — apple: round-trip through NdArray for conv/pool/interpolate
// ===========================================================================
#[cfg(feature = "apple")]
impl ModuleOps<Self> for NpuBurnBackend {
    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv2d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_off = npu_to_ndarray(&offset);
        let nd_w = npu_to_ndarray(&weight);
        let nd_m = mask.as_ref().map(npu_to_ndarray);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::deform_conv2d(nd_x, nd_off, nd_w, nd_m, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_off = npu_to_ndarray(&offset);
        let nd_w = npu_to_ndarray(&weight);
        let nd_m = mask.as_ref().map(npu_to_ndarray);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let nd_g = npu_to_ndarray(&output_grad);
        let r = <Nd as ModuleOps<Nd>>::deform_conv2d_backward(
            nd_x, nd_off, nd_w, nd_m, nd_b, nd_g, options,
        );
        DeformConv2dBackward::new(
            ndarray_to_npu(&r.x_grad),
            ndarray_to_npu(&r.offset_grad),
            ndarray_to_npu(&r.weight_grad),
            r.mask_grad.map(|g| ndarray_to_npu(&g)),
            r.bias_grad.map(|g| ndarray_to_npu(&g)),
        )
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv3d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv_transpose2d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv_transpose3d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::avg_pool2d(
            nd_x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        );
        ndarray_to_npu(&result)
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::avg_pool2d_backward(
            nd_x,
            nd_g,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        );
        ndarray_to_npu(&result)
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d(nd_x, output_size);
        ndarray_to_npu(&result)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d_backward(nd_x, nd_g);
        ndarray_to_npu(&result)
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d(
            nd_x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        );
        ndarray_to_npu(&result)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices(
            nd_x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        );
        MaxPool2dWithIndices::new(ndarray_to_npu(&result.output), result.indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&output_grad);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices_backward(
            nd_x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            nd_g,
            indices,
        );
        MaxPool2dBackward::new(ndarray_to_npu(&result.x_grad))
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::interpolate(nd_x, output_size, options);
        ndarray_to_npu(&result)
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::interpolate_backward(nd_x, nd_g, output_size, options);
        ndarray_to_npu(&result)
    }
}

// ===========================================================================
// ModuleOps — no feature: full NdArray delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl ModuleOps<Self> for NpuBurnBackend {
    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::conv2d(x, weight, bias, options)
    }
    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::deform_conv2d(x, offset, weight, mask, bias, options)
    }
    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let r = <Nd as ModuleOps<Nd>>::deform_conv2d_backward(
            x,
            offset,
            weight,
            mask,
            bias,
            output_grad,
            options,
        );
        DeformConv2dBackward::new(
            r.x_grad,
            r.offset_grad,
            r.weight_grad,
            r.mask_grad,
            r.bias_grad,
        )
    }
    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::conv3d(x, weight, bias, options)
    }
    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::conv_transpose2d(x, weight, bias, options)
    }
    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::conv_transpose3d(x, weight, bias, options)
    }
    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::avg_pool2d(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }
    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::avg_pool2d_backward(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }
    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d(x, output_size)
    }
    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d_backward(x, grad)
    }
    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }
    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let r = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        );
        MaxPool2dWithIndices::new(r.output, r.indices)
    }
    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        let r = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices_backward(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad,
            indices,
        );
        MaxPool2dBackward::new(r.x_grad)
    }
    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::interpolate(x, output_size, options)
    }
    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        <Nd as ModuleOps<Nd>>::interpolate_backward(x, grad, output_size, options)
    }
}

// ===========================================================================
// ModuleOps — intel/qualcomm: round-trip through NdArray
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl ModuleOps<Self> for NpuBurnBackend {
    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv2d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_off = npu_to_ndarray(&offset);
        let nd_w = npu_to_ndarray(&weight);
        let nd_m = mask.as_ref().map(npu_to_ndarray);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::deform_conv2d(nd_x, nd_off, nd_w, nd_m, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_off = npu_to_ndarray(&offset);
        let nd_w = npu_to_ndarray(&weight);
        let nd_m = mask.as_ref().map(npu_to_ndarray);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let nd_g = npu_to_ndarray(&output_grad);
        let r = <Nd as ModuleOps<Nd>>::deform_conv2d_backward(
            nd_x, nd_off, nd_w, nd_m, nd_b, nd_g, options,
        );
        DeformConv2dBackward::new(
            ndarray_to_npu(&r.x_grad),
            ndarray_to_npu(&r.offset_grad),
            ndarray_to_npu(&r.weight_grad),
            r.mask_grad.map(|g| ndarray_to_npu(&g)),
            r.bias_grad.map(|g| ndarray_to_npu(&g)),
        )
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv3d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv_transpose2d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_w = npu_to_ndarray(&weight);
        let nd_b = bias.as_ref().map(npu_to_ndarray);
        let result = <Nd as ModuleOps<Nd>>::conv_transpose3d(nd_x, nd_w, nd_b, options);
        ndarray_to_npu(&result)
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::avg_pool2d(
            nd_x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        );
        ndarray_to_npu(&result)
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::avg_pool2d_backward(
            nd_x,
            nd_g,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        );
        ndarray_to_npu(&result)
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d(nd_x, output_size);
        ndarray_to_npu(&result)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::adaptive_avg_pool2d_backward(nd_x, nd_g);
        ndarray_to_npu(&result)
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d(
            nd_x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        );
        ndarray_to_npu(&result)
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices(
            nd_x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        );
        MaxPool2dWithIndices::new(ndarray_to_npu(&result.output), result.indices)
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&output_grad);
        let result = <Nd as ModuleOps<Nd>>::max_pool2d_with_indices_backward(
            nd_x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            nd_g,
            indices,
        );
        MaxPool2dBackward::new(ndarray_to_npu(&result.x_grad))
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let result = <Nd as ModuleOps<Nd>>::interpolate(nd_x, output_size, options);
        ndarray_to_npu(&result)
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Nd as ModuleOps<Nd>>::interpolate_backward(nd_x, nd_g, output_size, options);
        ndarray_to_npu(&result)
    }
}
