//! Apple Neural Engine FFI declarations.
//!
//! These functions are implemented in Swift and linked via `build.rs`.
//! Each `i32` return value is a handle into a Swift-side tensor table.

#[cfg(feature = "apple")]
#[allow(dead_code)]
extern "C" {
    pub(super) fn npu_create_tensor(
        shape: *const i32,
        dims: i32,
        data: *const f32,
        len: i32,
    ) -> i32;
    pub(super) fn npu_create_int_tensor(
        shape: *const i32,
        dims: i32,
        data: *const i32,
        len: i32,
    ) -> i32;
    pub(super) fn npu_free_tensor(id: i32);
    pub(super) fn npu_get_shape(id: i32, out: *mut i32, max: i32) -> i32;
    pub(super) fn npu_get_data(id: i32, out: *mut f32, max: i32) -> i32;
    pub(super) fn npu_get_int_data(id: i32, out: *mut i32, max: i32) -> i32;
    pub(super) fn npu_clone(id: i32) -> i32;

    // Matmul
    pub(super) fn npu_matmul(a: i32, b: i32) -> i32;

    // Binary arithmetic
    pub(super) fn npu_add(a: i32, b: i32) -> i32;
    pub(super) fn npu_sub(a: i32, b: i32) -> i32;
    pub(super) fn npu_mul(a: i32, b: i32) -> i32;
    pub(super) fn npu_div(a: i32, b: i32) -> i32;

    // Scalar arithmetic
    pub(super) fn npu_add_scalar(a: i32, s: f32) -> i32;
    pub(super) fn npu_sub_scalar(a: i32, s: f32) -> i32;
    pub(super) fn npu_mul_scalar(a: i32, s: f32) -> i32;
    pub(super) fn npu_div_scalar(a: i32, s: f32) -> i32;

    // Unary math
    pub(super) fn npu_neg(a: i32) -> i32;
    pub(super) fn npu_exp(a: i32) -> i32;
    pub(super) fn npu_log(a: i32) -> i32;
    pub(super) fn npu_sqrt(a: i32) -> i32;
    pub(super) fn npu_abs(a: i32) -> i32;
    pub(super) fn npu_tanh(a: i32) -> i32;
    pub(super) fn npu_sin(a: i32) -> i32;
    pub(super) fn npu_cos(a: i32) -> i32;
    pub(super) fn npu_floor(a: i32) -> i32;
    pub(super) fn npu_ceil(a: i32) -> i32;
    pub(super) fn npu_erf(a: i32) -> i32;

    // Power
    pub(super) fn npu_pow(a: i32, b: i32) -> i32;
    pub(super) fn npu_pow_scalar(a: i32, p: f32) -> i32;

    // Clamp
    pub(super) fn npu_clamp_min(a: i32, min: f32) -> i32;
    pub(super) fn npu_clamp_max(a: i32, max: f32) -> i32;
    pub(super) fn npu_clamp(a: i32, min: f32, max: f32) -> i32;

    // Softmax
    pub(super) fn npu_softmax(a: i32, dim: i32) -> i32;

    // Reductions
    pub(super) fn npu_sum(a: i32) -> i32;
    pub(super) fn npu_sum_dim(a: i32, dim: i32) -> i32;
    pub(super) fn npu_mean_all(a: i32) -> i32;
    pub(super) fn npu_mean(a: i32, dim: i32) -> i32;
    pub(super) fn npu_max(a: i32) -> i32;
    pub(super) fn npu_max_dim(a: i32, dim: i32) -> i32;
    pub(super) fn npu_min(a: i32) -> i32;
    pub(super) fn npu_min_dim(a: i32, dim: i32) -> i32;

    // Argmax / argmin (return int tensor handles)
    pub(super) fn npu_argmax(a: i32, dim: i32) -> i32;
    pub(super) fn npu_argmin(a: i32, dim: i32) -> i32;

    // Shape ops
    pub(super) fn npu_reshape(a: i32, shape: *const i32, dims: i32) -> i32;
    pub(super) fn npu_transpose(a: i32, dim0: i32, dim1: i32) -> i32;
    pub(super) fn npu_permute(a: i32, perm: *const i32, len: i32) -> i32;
    pub(super) fn npu_narrow(a: i32, dim: i32, start: i32, length: i32) -> i32;
    pub(super) fn npu_expand(a: i32, shape: *const i32, dims: i32) -> i32;

    // Concat
    pub(super) fn npu_cat(ids: *const i32, count: i32, dim: i32) -> i32;

    // Indexing
    pub(super) fn npu_index_select(a: i32, indices: *const i32, len: i32) -> i32;
    pub(super) fn npu_gather(a: i32, dim: i32, indices: i32) -> i32;
    pub(super) fn npu_slice(a: i32, ranges: *const i32, num_ranges: i32) -> i32;

    // Comparison
    pub(super) fn npu_equal(a: i32, b: i32) -> i32;
    pub(super) fn npu_greater(a: i32, b: i32) -> i32;
    pub(super) fn npu_less(a: i32, b: i32) -> i32;

    // Masking
    pub(super) fn npu_mask_fill(a: i32, mask: i32, value: f32) -> i32;
    /// f16 payloads cross as raw bit patterns, matching `half::f16`'s layout.
    pub(super) fn npu_create_tensor_f16(
        shape: *const i32,
        dims: i32,
        data: *const u16,
        len: i32,
    ) -> i32;
    pub(super) fn npu_get_data_f16(id: i32, out: *mut u16, max_len: i32) -> i32;
    /// 0 = f32, 1 = f16, 2 = i32, -1 = unknown.
    pub(super) fn npu_get_dtype(id: i32) -> i32;
    pub(super) fn npu_cast_float(id: i32, code: i32) -> i32;
    pub(super) fn npu_mask_where(a: i32, mask: i32, source: i32) -> i32;

    // Creation
    pub(super) fn npu_zeros(shape: *const i32, dims: i32) -> i32;
    pub(super) fn npu_ones(shape: *const i32, dims: i32) -> i32;
    pub(super) fn npu_full(shape: *const i32, dims: i32, value: f32) -> i32;
    pub(super) fn npu_scalar_tensor(value: f32) -> i32;

    // Casting
    pub(super) fn npu_cast_to_int(a: i32) -> i32;
    pub(super) fn npu_cast_to_float(a: i32) -> i32;
}
