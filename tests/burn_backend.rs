use burn::tensor::{Shape, Tensor};
use burn_npu::{NpuBurnBackend, NpuBurnDevice};

type B = NpuBurnBackend;

fn dev() -> NpuBurnDevice {
    NpuBurnDevice::Default
}

// ── Matmul ──

#[test]
fn matmul_2x2() {
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev());
    let b = Tensor::<B, 2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &dev());
    let c = a.matmul(b);
    let d: Vec<f32> = c.into_data().to_vec().unwrap();
    assert_eq!(d, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn matmul_non_square() {
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &dev());
    let b = Tensor::<B, 2>::from_floats([[4.0], [5.0], [6.0]], &dev());
    let c = a.matmul(b);
    let d: Vec<f32> = c.into_data().to_vec().unwrap();
    assert_eq!(d, vec![32.0]);
}

// ── Arithmetic ──

#[test]
fn add_tensors() {
    let a = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &dev());
    let b = Tensor::<B, 1>::from_floats([10.0, 20.0, 30.0], &dev());
    let c: Vec<f32> = (a + b).into_data().to_vec().unwrap();
    assert_eq!(c, vec![11.0, 22.0, 33.0]);
}

#[test]
fn sub_tensors() {
    let a = Tensor::<B, 1>::from_floats([10.0, 20.0, 30.0], &dev());
    let b = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &dev());
    let c: Vec<f32> = (a - b).into_data().to_vec().unwrap();
    assert_eq!(c, vec![9.0, 18.0, 27.0]);
}

#[test]
fn mul_tensors() {
    let a = Tensor::<B, 1>::from_floats([2.0, 3.0, 4.0], &dev());
    let b = Tensor::<B, 1>::from_floats([5.0, 6.0, 7.0], &dev());
    let c: Vec<f32> = (a * b).into_data().to_vec().unwrap();
    assert_eq!(c, vec![10.0, 18.0, 28.0]);
}

#[test]
fn div_tensors() {
    let a = Tensor::<B, 1>::from_floats([10.0, 20.0, 30.0], &dev());
    let b = Tensor::<B, 1>::from_floats([2.0, 5.0, 10.0], &dev());
    let c: Vec<f32> = (a / b).into_data().to_vec().unwrap();
    assert_eq!(c, vec![5.0, 4.0, 3.0]);
}

// ── Scalar ops ──

#[test]
fn add_scalar() {
    let a = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &dev());
    let c: Vec<f32> = (a + 10.0).into_data().to_vec().unwrap();
    assert_eq!(c, vec![11.0, 12.0, 13.0]);
}

#[test]
fn mul_scalar() {
    let a = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &dev());
    let c: Vec<f32> = (a * 10.0).into_data().to_vec().unwrap();
    assert_eq!(c, vec![10.0, 20.0, 30.0]);
}

// ── Unary math ──

#[test]
fn exp_log_roundtrip() {
    let a = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &dev());
    let b: Vec<f32> = a.clone().exp().log().into_data().to_vec().unwrap();
    for (got, expected) in b.iter().zip([1.0, 2.0, 3.0]) {
        assert!((got - expected).abs() < 1e-5, "exp(log(x)) != x: got {got}");
    }
}

#[test]
fn sqrt_values() {
    let a = Tensor::<B, 1>::from_floats([1.0, 4.0, 9.0, 16.0], &dev());
    let b: Vec<f32> = a.sqrt().into_data().to_vec().unwrap();
    assert_eq!(b, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn neg_values() {
    let a = Tensor::<B, 1>::from_floats([1.0, -2.0, 3.0], &dev());
    let b: Vec<f32> = a.neg().into_data().to_vec().unwrap();
    assert_eq!(b, vec![-1.0, 2.0, -3.0]);
}

#[test]
fn abs_values() {
    let a = Tensor::<B, 1>::from_floats([-3.0, -1.0, 0.0, 2.0, 5.0], &dev());
    let b: Vec<f32> = a.abs().into_data().to_vec().unwrap();
    assert_eq!(b, vec![3.0, 1.0, 0.0, 2.0, 5.0]);
}

#[test]
fn tanh_range() {
    let a = Tensor::<B, 1>::from_floats([-100.0, 0.0, 100.0], &dev());
    let b: Vec<f32> = a.tanh().into_data().to_vec().unwrap();
    assert!((b[0] - (-1.0)).abs() < 1e-5);
    assert!((b[1] - 0.0).abs() < 1e-5);
    assert!((b[2] - 1.0).abs() < 1e-5);
}

#[test]
fn floor_ceil() {
    let a = Tensor::<B, 1>::from_floats([1.3, 2.7, -0.5], &dev());
    let f: Vec<f32> = a.clone().floor().into_data().to_vec().unwrap();
    let c: Vec<f32> = a.ceil().into_data().to_vec().unwrap();
    assert_eq!(f, vec![1.0, 2.0, -1.0]);
    assert_eq!(c, vec![2.0, 3.0, 0.0]);
}

#[test]
fn sin_cos_identity() {
    // sin^2 + cos^2 = 1
    let a = Tensor::<B, 1>::from_floats([0.0, 1.0, 2.0, 3.0], &dev());
    let s = a.clone().sin();
    let c = a.cos();
    let sum: Vec<f32> = (s.clone() * s + c.clone() * c)
        .into_data()
        .to_vec()
        .unwrap();
    for v in &sum {
        assert!((v - 1.0).abs() < 1e-4, "sin^2+cos^2 = {v}, expected 1.0");
    }
}

// ── Reductions ──

#[test]
fn sum_all() {
    let a = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &dev());
    let s: Vec<f32> = a.sum().into_data().to_vec().unwrap();
    assert_eq!(s, vec![10.0]);
}

#[test]
fn sum_dim() {
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev());
    let s: Vec<f32> = a.sum_dim(1).into_data().to_vec().unwrap();
    assert_eq!(s, vec![3.0, 7.0]);
}

#[test]
fn mean_dim() {
    let a = Tensor::<B, 2>::from_floats([[2.0, 4.0], [6.0, 8.0]], &dev());
    let m: Vec<f32> = a.mean_dim(1).into_data().to_vec().unwrap();
    assert_eq!(m, vec![3.0, 7.0]);
}

#[test]
fn max_min() {
    let a = Tensor::<B, 1>::from_floats([3.0, 1.0, 4.0, 1.0, 5.0], &dev());
    let mx: Vec<f32> = a.clone().max().into_data().to_vec().unwrap();
    let mn: Vec<f32> = a.min().into_data().to_vec().unwrap();
    assert_eq!(mx, vec![5.0]);
    assert_eq!(mn, vec![1.0]);
}

#[test]
fn argmax_dim() {
    let a = Tensor::<B, 2>::from_floats([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]], &dev());
    let idx: Vec<i32> = a.argmax(1).into_data().to_vec().unwrap();
    assert_eq!(idx, vec![1, 2]);
}

// ── Shape ops ──

#[test]
fn reshape() {
    let a = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &dev());
    let b = a.reshape([2, 3]);
    assert_eq!(b.shape(), Shape::new([2, 3]));
    let d: Vec<f32> = b.into_data().to_vec().unwrap();
    assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn transpose_2d() {
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &dev());
    let t = a.transpose();
    assert_eq!(t.shape(), Shape::new([3, 2]));
    let d: Vec<f32> = t.into_data().to_vec().unwrap();
    assert_eq!(d, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn narrow_dim() {
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &dev());
    let b = a.narrow(1, 1, 2); // cols 1..3
    assert_eq!(b.shape(), Shape::new([2, 2]));
    let d: Vec<f32> = b.into_data().to_vec().unwrap();
    assert_eq!(d, vec![2.0, 3.0, 5.0, 6.0]);
}

#[test]
fn cat_tensors() {
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &dev());
    let b = Tensor::<B, 2>::from_floats([[3.0, 4.0]], &dev());
    let c = Tensor::cat(vec![a, b], 0);
    assert_eq!(c.shape(), Shape::new([2, 2]));
    let d: Vec<f32> = c.into_data().to_vec().unwrap();
    assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0]);
}

// ── Clamp ──

#[test]
fn clamp_min() {
    let a = Tensor::<B, 1>::from_floats([-2.0, -1.0, 0.0, 1.0, 2.0], &dev());
    let b: Vec<f32> = a.clamp_min(0.0).into_data().to_vec().unwrap();
    assert_eq!(b, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn clamp_max() {
    let a = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &dev());
    let b: Vec<f32> = a.clamp_max(3.0).into_data().to_vec().unwrap();
    assert_eq!(b, vec![1.0, 2.0, 3.0, 3.0, 3.0]);
}

// ── Erf ──

#[test]
fn erf_known_values() {
    let a = Tensor::<B, 1>::from_floats([0.0, 1.0, -1.0], &dev());
    let b: Vec<f32> = a.erf().into_data().to_vec().unwrap();
    assert!((b[0] - 0.0).abs() < 1e-3, "erf(0) = {}", b[0]);
    assert!((b[1] - 0.8427).abs() < 1e-3, "erf(1) = {}", b[1]);
    assert!((b[2] - (-0.8427)).abs() < 1e-3, "erf(-1) = {}", b[2]);
}

// ── Powf ──

#[test]
fn powf_scalar() {
    let a = Tensor::<B, 1>::from_floats([2.0, 3.0, 4.0], &dev());
    let b: Vec<f32> = a.powf_scalar(2.0).into_data().to_vec().unwrap();
    assert_eq!(b, vec![4.0, 9.0, 16.0]);
}

// ── Zeros / Ones ──

#[test]
fn zeros_and_ones() {
    let z = Tensor::<B, 2>::zeros([2, 3], &dev());
    let zd: Vec<f32> = z.into_data().to_vec().unwrap();
    assert!(zd.iter().all(|&v| v == 0.0));

    let o = Tensor::<B, 2>::ones([2, 3], &dev());
    let od: Vec<f32> = o.into_data().to_vec().unwrap();
    assert!(od.iter().all(|&v| v == 1.0));
}

// ── Int tensors ──

#[test]
fn int_from_data() {
    let a = Tensor::<B, 1, burn::tensor::Int>::from_ints([1, 2, 3], &dev());
    let d: Vec<i32> = a.into_data().to_vec().unwrap();
    assert_eq!(d, vec![1, 2, 3]);
}

// ── Bool tensors ──

#[test]
fn bool_equal() {
    let a = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &dev());
    let b = Tensor::<B, 1>::from_floats([1.0, 0.0, 3.0], &dev());
    let eq = a.equal(b);
    let d: Vec<bool> = eq.into_data().to_vec().unwrap();
    assert_eq!(d, vec![true, false, true]);
}

// ── Masking ──
//
// On the apple backend these run on the NPU via npu_mask_fill/npu_mask_where
// rather than round-tripping to the CPU, so they need direct coverage.

#[test]
fn mask_fill_replaces_selected_elements() {
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev());
    let mask = a.clone().greater_elem(2.0);
    let out: Vec<f32> = a.mask_fill(mask, 0.0).into_data().to_vec().unwrap();
    assert_eq!(out, vec![1.0, 2.0, 0.0, 0.0]);
}

#[test]
fn mask_fill_with_all_false_mask_is_identity() {
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev());
    let mask = a.clone().greater_elem(100.0);
    let out: Vec<f32> = a.mask_fill(mask, -1.0).into_data().to_vec().unwrap();
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn mask_fill_with_all_true_mask_replaces_everything() {
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev());
    let mask = a.clone().greater_elem(-1.0);
    let out: Vec<f32> = a.mask_fill(mask, 7.0).into_data().to_vec().unwrap();
    assert_eq!(out, vec![7.0, 7.0, 7.0, 7.0]);
}

#[test]
fn mask_where_selects_between_tensors() {
    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev());
    let b = Tensor::<B, 2>::from_floats([[10.0, 20.0], [30.0, 40.0]], &dev());
    let mask = a.clone().greater_elem(2.0);
    let out: Vec<f32> = a.mask_where(mask, b).into_data().to_vec().unwrap();
    assert_eq!(out, vec![1.0, 2.0, 30.0, 40.0]);
}

/// The attention-style shape: mask a 3D tensor with a large negative value
/// before softmax.
#[test]
fn mask_fill_on_batched_tensor() {
    let a =
        Tensor::<B, 3>::from_floats([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], &dev());
    let mask = a.clone().greater_elem(4.0);
    let out: Vec<f32> = a.mask_fill(mask, -1.0).into_data().to_vec().unwrap();
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, -1.0, -1.0, -1.0, -1.0]);
}

// ── Precision ──
//
// fp16 is the ANE's native format. These pin that an f16 tensor stays f16
// through the backend rather than being silently widened to f32.

#[test]
fn f16_tensor_reports_f16_dtype() {
    use burn::tensor::TensorData;
    use burn_tensor::f16;

    let data = TensorData::new(
        vec![f16::from_f32(1.0), f16::from_f32(2.0)],
        vec![1usize, 2],
    );
    let t = Tensor::<B, 2>::from_data(data, (&dev(), burn::tensor::DType::F16));
    assert_eq!(t.dtype(), burn::tensor::DType::F16);
}

#[test]
fn f16_roundtrips_through_the_backend() {
    use burn::tensor::TensorData;
    use burn_tensor::f16;

    let values: Vec<f16> = [1.0f32, -2.5, 0.25, 100.0]
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let t = Tensor::<B, 2>::from_data(
        TensorData::new(values, vec![2usize, 2]),
        (&dev(), burn::tensor::DType::F16),
    );
    assert_eq!(t.dtype(), burn::tensor::DType::F16);
    let out: Vec<f16> = t.into_data().to_vec().unwrap();
    let out: Vec<f32> = out.iter().map(|v| v.to_f32()).collect();
    assert_eq!(out, vec![1.0, -2.5, 0.25, 100.0]);
}

#[test]
fn f16_matmul_is_correct() {
    use burn::tensor::{DType, TensorData};
    use burn_tensor::f16;

    let a = TensorData::new(
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ],
        vec![2usize, 2],
    );
    let b = TensorData::new(
        vec![
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
            f16::from_f32(8.0),
        ],
        vec![2usize, 2],
    );
    let x = Tensor::<B, 2>::from_data(a, (&dev(), DType::F16));
    let y = Tensor::<B, 2>::from_data(b, (&dev(), DType::F16));
    let out = x.matmul(y);
    // The result stays in f16 rather than being widened back to f32.
    assert_eq!(out.dtype(), DType::F16);
    let got: Vec<f16> = out.into_data().to_vec().unwrap();
    let got: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();
    assert_eq!(got, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn cast_to_f16_actually_casts() {
    use burn::tensor::DType;

    let a = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &dev());
    assert_eq!(a.dtype(), DType::F32);
    let half = a.cast(DType::F16);
    assert_eq!(half.dtype(), DType::F16);
    let back: Vec<f32> = half.clone().cast(DType::F32).into_data().to_vec().unwrap();
    assert_eq!(back, vec![1.0, 2.0, 3.0, 4.0]);
}
