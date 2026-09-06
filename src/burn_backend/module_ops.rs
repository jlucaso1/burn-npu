//! `ModuleOps` implementations for all platform variants.

use burn_tensor::ops::*;
use burn_tensor::ops::{BoolTensor, FloatTensor, IntTensor};

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
use super::tensor::*;
use super::{Fx, NpuBurnBackend};

// ===========================================================================
// ModuleOps — apple: round-trip through Flex for conv/pool/interpolate
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
        let result = <Fx as ModuleOps<Fx>>::conv2d(nd_x, nd_w, nd_b, options);
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
        let result = <Fx as ModuleOps<Fx>>::deform_conv2d(nd_x, nd_off, nd_w, nd_m, nd_b, options);
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
        let r = <Fx as ModuleOps<Fx>>::deform_conv2d_backward(
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
        let result = <Fx as ModuleOps<Fx>>::conv3d(nd_x, nd_w, nd_b, options);
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
        let result = <Fx as ModuleOps<Fx>>::conv_transpose2d(nd_x, nd_w, nd_b, options);
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
        let result = <Fx as ModuleOps<Fx>>::conv_transpose3d(nd_x, nd_w, nd_b, options);
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
        let result = <Fx as ModuleOps<Fx>>::avg_pool2d(
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
        let result = <Fx as ModuleOps<Fx>>::avg_pool2d_backward(
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
        let result = <Fx as ModuleOps<Fx>>::adaptive_avg_pool2d(nd_x, output_size);
        ndarray_to_npu(&result)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Fx as ModuleOps<Fx>>::adaptive_avg_pool2d_backward(nd_x, nd_g);
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
        let result = <Fx as ModuleOps<Fx>>::max_pool2d(
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
        let result = <Fx as ModuleOps<Fx>>::max_pool2d_with_indices(
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
        let result = <Fx as ModuleOps<Fx>>::max_pool2d_with_indices_backward(
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
        let result = <Fx as ModuleOps<Fx>>::interpolate(nd_x, output_size, options);
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
        let result = <Fx as ModuleOps<Fx>>::interpolate_backward(nd_x, nd_g, output_size, options);
        ndarray_to_npu(&result)
    }

    fn attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        attn_bias: Option<FloatTensor<Self>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Self> {
        // No fused-attention primitive on the vendor APIs yet; round-trip
        // through the CPU delegate. This is a prime candidate for native
        // dispatch once the backend builds graphs instead of single ops.
        let q = npu_to_ndarray(&query);
        let k = npu_to_ndarray(&key);
        let v = npu_to_ndarray(&value);
        let b = attn_bias.as_ref().map(npu_to_ndarray);
        let result = <Fx as ModuleOps<Fx>>::attention(q, k, v, mask, b, options);
        ndarray_to_npu(&result)
    }

    fn rfft(
        signal: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let nd = npu_to_ndarray(&signal);
        let (re, im) = <Fx as ModuleOps<Fx>>::rfft(nd, dim, n);
        (ndarray_to_npu(&re), ndarray_to_npu(&im))
    }

    fn irfft(
        spectrum_re: FloatTensor<Self>,
        spectrum_im: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Self> {
        let re = npu_to_ndarray(&spectrum_re);
        let im = npu_to_ndarray(&spectrum_im);
        let result = <Fx as ModuleOps<Fx>>::irfft(re, im, dim, n);
        ndarray_to_npu(&result)
    }
}

// ===========================================================================
// ModuleOps — no feature: full Flex delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl ModuleOps<Self> for NpuBurnBackend {
    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        <Fx as ModuleOps<Fx>>::conv2d(x, weight, bias, options)
    }
    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        <Fx as ModuleOps<Fx>>::deform_conv2d(x, offset, weight, mask, bias, options)
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
        let r = <Fx as ModuleOps<Fx>>::deform_conv2d_backward(
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
        <Fx as ModuleOps<Fx>>::conv3d(x, weight, bias, options)
    }
    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        <Fx as ModuleOps<Fx>>::conv_transpose2d(x, weight, bias, options)
    }
    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        <Fx as ModuleOps<Fx>>::conv_transpose3d(x, weight, bias, options)
    }
    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        <Fx as ModuleOps<Fx>>::avg_pool2d(
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
        <Fx as ModuleOps<Fx>>::avg_pool2d_backward(
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
        <Fx as ModuleOps<Fx>>::adaptive_avg_pool2d(x, output_size)
    }
    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        <Fx as ModuleOps<Fx>>::adaptive_avg_pool2d_backward(x, grad)
    }
    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        <Fx as ModuleOps<Fx>>::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }
    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let r = <Fx as ModuleOps<Fx>>::max_pool2d_with_indices(
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
        let r = <Fx as ModuleOps<Fx>>::max_pool2d_with_indices_backward(
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
        <Fx as ModuleOps<Fx>>::interpolate(x, output_size, options)
    }
    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        <Fx as ModuleOps<Fx>>::interpolate_backward(x, grad, output_size, options)
    }

    fn attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        attn_bias: Option<FloatTensor<Self>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Self> {
        <Fx as ModuleOps<Fx>>::attention(query, key, value, mask, attn_bias, options)
    }

    fn rfft(
        signal: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        <Fx as ModuleOps<Fx>>::rfft(signal, dim, n)
    }

    fn irfft(
        spectrum_re: FloatTensor<Self>,
        spectrum_im: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Self> {
        <Fx as ModuleOps<Fx>>::irfft(spectrum_re, spectrum_im, dim, n)
    }
}

// ===========================================================================
// ModuleOps — intel/qualcomm: round-trip through Flex
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
        let result = <Fx as ModuleOps<Fx>>::conv2d(nd_x, nd_w, nd_b, options);
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
        let result = <Fx as ModuleOps<Fx>>::deform_conv2d(nd_x, nd_off, nd_w, nd_m, nd_b, options);
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
        let r = <Fx as ModuleOps<Fx>>::deform_conv2d_backward(
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
        let result = <Fx as ModuleOps<Fx>>::conv3d(nd_x, nd_w, nd_b, options);
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
        let result = <Fx as ModuleOps<Fx>>::conv_transpose2d(nd_x, nd_w, nd_b, options);
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
        let result = <Fx as ModuleOps<Fx>>::conv_transpose3d(nd_x, nd_w, nd_b, options);
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
        let result = <Fx as ModuleOps<Fx>>::avg_pool2d(
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
        let result = <Fx as ModuleOps<Fx>>::avg_pool2d_backward(
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
        let result = <Fx as ModuleOps<Fx>>::adaptive_avg_pool2d(nd_x, output_size);
        ndarray_to_npu(&result)
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let nd_x = npu_to_ndarray(&x);
        let nd_g = npu_to_ndarray(&grad);
        let result = <Fx as ModuleOps<Fx>>::adaptive_avg_pool2d_backward(nd_x, nd_g);
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
        let result = <Fx as ModuleOps<Fx>>::max_pool2d(
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
        let result = <Fx as ModuleOps<Fx>>::max_pool2d_with_indices(
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
        let result = <Fx as ModuleOps<Fx>>::max_pool2d_with_indices_backward(
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
        let result = <Fx as ModuleOps<Fx>>::interpolate(nd_x, output_size, options);
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
        let result = <Fx as ModuleOps<Fx>>::interpolate_backward(nd_x, nd_g, output_size, options);
        ndarray_to_npu(&result)
    }

    fn attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        attn_bias: Option<FloatTensor<Self>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Self> {
        #[cfg(feature = "intel")]
        {
            if let Ok(output) = crate::backends::intel::openvino_attention_masked(
                &query,
                &key,
                &value,
                attn_bias.as_ref(),
                mask.as_ref(),
                &options,
            ) {
                return output;
            }
            if std::env::var_os("BURN_NPU_TRACE").is_some() {
                eprintln!("OpenVINO attention unavailable; Flex CPU fallback");
            }
        }
        #[cfg(feature = "intel")]
        crate::backends::intel::diagnostics::fallback();
        // Preserve the CPU delegate's semantics for unsupported options and
        // dtypes. Intel shares this storage; Qualcomm converts its tensors.
        let q = npu_to_ndarray(&query);
        let k = npu_to_ndarray(&key);
        let v = npu_to_ndarray(&value);
        let b = attn_bias.as_ref().map(npu_to_ndarray);
        let result = <Fx as ModuleOps<Fx>>::attention(q, k, v, mask, b, options);
        ndarray_to_npu(&result)
    }

    fn rfft(
        signal: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let nd = npu_to_ndarray(&signal);
        let (re, im) = <Fx as ModuleOps<Fx>>::rfft(nd, dim, n);
        (ndarray_to_npu(&re), ndarray_to_npu(&im))
    }

    fn irfft(
        spectrum_re: FloatTensor<Self>,
        spectrum_im: FloatTensor<Self>,
        dim: usize,
        n: Option<usize>,
    ) -> FloatTensor<Self> {
        let re = npu_to_ndarray(&spectrum_re);
        let im = npu_to_ndarray(&spectrum_im);
        let result = <Fx as ModuleOps<Fx>>::irfft(re, im, dim, n);
        ndarray_to_npu(&result)
    }
}
