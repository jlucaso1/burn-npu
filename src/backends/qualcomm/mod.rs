//! Qualcomm Hexagon NPU backend via the QNN (AI Engine Direct) SDK.
//!
//! `QnnFloatTensor` stores data as `Vec<f32>` with shape metadata. Matmul is
//! dispatched to the Hexagon Tensor Processor when a QAIRT SDK was present at
//! build time and the HTP backend library can be opened at runtime; everything
//! else runs on CPU or delegates to burn-flex.
//!
//! # Build requirements
//!
//! NPU dispatch is compiled in only when `QNN_SDK_ROOT` points at a Qualcomm AI
//! Runtime (QAIRT) SDK during the build -- see [`crate`]'s `build.rs`. Bindings
//! are generated from the SDK's own headers rather than hand-written, because
//! the QNN ABI uses versioned tagged unions that are unsafe to transcribe by
//! hand. Without the SDK the feature still builds and runs everything on CPU.
//!
//! # Status
//!
//! The NPU path is **implemented but awaiting hardware validation** -- see
//! [`qnn`] for what specifically still needs checking on a device.

use burn_tensor::{DType, Shape};

#[cfg(qnn_sdk)]
pub mod qnn;

/// Returned when Hexagon NPU dispatch is unavailable and the caller should fall
/// back to CPU.
///
/// Expected rather than exceptional: it is the normal result on any machine
/// built without the QNN SDK, or without a Snapdragon device present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QnnUnavailable;

impl std::fmt::Display for QnnUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("QNN runtime or Hexagon device unavailable; falling back to CPU")
    }
}

impl std::error::Error for QnnUnavailable {}

/// Attempt Hexagon NPU dispatch, falling back to CPU.
///
/// This is the single entry point the burn backend calls for matmul.
pub fn matmul(lhs: &QnnFloatTensor, rhs: &QnnFloatTensor) -> QnnFloatTensor {
    #[cfg(qnn_sdk)]
    {
        if let Ok(result) = qnn::qnn_matmul(lhs, rhs) {
            return result;
        }
    }
    cpu_matmul(lhs, rhs)
}

// ---------------------------------------------------------------------------
// QnnFloatTensor
// ---------------------------------------------------------------------------

/// A float tensor backed by a `Vec<f32>` with shape metadata.
///
/// When QNN SDK integration is complete, compute-heavy ops like matmul will
/// dispatch to the Hexagon NPU. Currently all ops execute on CPU.
#[derive(Debug, Clone)]
pub struct QnnFloatTensor {
    /// Flat f32 data in row-major order.
    pub data: Vec<f32>,
    /// Shape dimensions.
    pub shape: Vec<usize>,
}

impl QnnFloatTensor {
    /// Create a tensor from flat data and shape.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "QnnFloatTensor::new: data.len() != product(shape)"
        );
        Self { data, shape }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Self {
            data: vec![0.0; total],
            shape,
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Self {
            data: vec![1.0; total],
            shape,
        }
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: Vec<usize>, value: f32) -> Self {
        let total: usize = shape.iter().product();
        Self {
            data: vec![value; total],
            shape,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

impl burn_tensor::TensorMetadata for QnnFloatTensor {
    fn dtype(&self) -> DType {
        DType::F32
    }

    fn shape(&self) -> Shape {
        Shape::from(self.shape.clone())
    }
}

// ---------------------------------------------------------------------------
// CPU matmul
// ---------------------------------------------------------------------------

/// CPU matmul, used whenever NPU dispatch is unavailable.
///
/// Prefer [`matmul`], which tries the Hexagon NPU first.
pub fn cpu_matmul(lhs: &QnnFloatTensor, rhs: &QnnFloatTensor) -> QnnFloatTensor {
    // Support batched matmul: [..., M, K] x [..., K, N] -> [..., M, N]
    let lhs_ndim = lhs.shape.len();
    let rhs_ndim = rhs.shape.len();

    assert!(
        lhs_ndim >= 2 && rhs_ndim >= 2,
        "matmul requires at least 2D tensors"
    );

    let m = lhs.shape[lhs_ndim - 2];
    let k = lhs.shape[lhs_ndim - 1];
    let n = rhs.shape[rhs_ndim - 1];
    assert_eq!(
        rhs.shape[rhs_ndim - 2],
        k,
        "matmul inner dimensions mismatch"
    );

    // Compute batch dimensions.
    let lhs_batch: usize = lhs.shape[..lhs_ndim - 2].iter().product();
    let rhs_batch: usize = rhs.shape[..rhs_ndim - 2].iter().product();
    let batch = lhs_batch.max(rhs_batch);

    let mut out_shape: Vec<usize> = if lhs_ndim >= rhs_ndim {
        lhs.shape[..lhs_ndim - 2].to_vec()
    } else {
        rhs.shape[..rhs_ndim - 2].to_vec()
    };
    out_shape.push(m);
    out_shape.push(n);

    let mut result = vec![0.0f32; batch * m * n];

    for b in 0..batch {
        let lhs_offset = (b % lhs_batch) * m * k;
        let rhs_offset = (b % rhs_batch) * k * n;
        let out_offset = b * m * n;

        for i in 0..m {
            let out_row = &mut result[out_offset + i * n..out_offset + (i + 1) * n];
            for p in 0..k {
                let lhs_val = lhs.data[lhs_offset + i * k + p];
                if lhs_val == 0.0 {
                    continue;
                }
                let rhs_row = &rhs.data[rhs_offset + p * n..rhs_offset + (p + 1) * n];
                for (o, &r) in out_row.iter_mut().zip(rhs_row) {
                    *o += lhs_val * r;
                }
            }
        }
    }

    QnnFloatTensor::new(result, out_shape)
}

// ---------------------------------------------------------------------------
// Conversion helpers for Flex interop
// ---------------------------------------------------------------------------

/// Convert QnnFloatTensor -> FlexTensor (for delegating ops to burn-flex).
pub fn qnn_to_ndarray(tensor: &QnnFloatTensor) -> burn_flex::FlexTensor {
    burn_flex::FlexTensor::from_data(burn_tensor::TensorData::new(
        tensor.data.clone(),
        tensor.shape.clone(),
    ))
}

/// Convert FlexTensor (f32) -> QnnFloatTensor.
pub fn ndarray_to_qnn(tensor: &burn_flex::FlexTensor) -> QnnFloatTensor {
    assert_eq!(
        burn_tensor::TensorMetadata::dtype(tensor),
        DType::F32,
        "ndarray_to_qnn: expected an f32 tensor"
    );
    let contig = tensor.to_contiguous();
    let shape = burn_tensor::TensorMetadata::shape(tensor).to_vec();
    QnnFloatTensor::new(contig.storage::<f32>().to_vec(), shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straightforward triple-loop reference to check the blocked kernel against.
    fn reference(lhs: &QnnFloatTensor, rhs: &QnnFloatTensor) -> Vec<f32> {
        let ln = lhs.shape.len();
        let rn = rhs.shape.len();
        let (m, k, n) = (lhs.shape[ln - 2], lhs.shape[ln - 1], rhs.shape[rn - 1]);
        let batch: usize = lhs.shape[..ln - 2].iter().product::<usize>().max(1);
        let mut out = vec![0.0f32; batch * m * n];
        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k {
                        sum += lhs.data[b * m * k + i * k + p] * rhs.data[b * k * n + p * n + j];
                    }
                    out[b * m * n + i * n + j] = sum;
                }
            }
        }
        out
    }

    fn seeded(len: usize, seed: u32) -> Vec<f32> {
        // Small LCG; deterministic and dependency-free.
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1 << 24) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn cpu_matmul_2d_matches_reference() {
        let lhs = QnnFloatTensor::new(seeded(6 * 5, 1), vec![6, 5]);
        let rhs = QnnFloatTensor::new(seeded(5 * 4, 2), vec![5, 4]);
        let got = cpu_matmul(&lhs, &rhs);
        assert_eq!(got.shape, vec![6, 4]);
        for (a, b) in got.data.iter().zip(reference(&lhs, &rhs)) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn cpu_matmul_batched_matches_reference() {
        let lhs = QnnFloatTensor::new(seeded(3 * 4 * 7, 3), vec![3, 4, 7]);
        let rhs = QnnFloatTensor::new(seeded(3 * 7 * 2, 4), vec![3, 7, 2]);
        let got = cpu_matmul(&lhs, &rhs);
        assert_eq!(got.shape, vec![3, 4, 2]);
        for (a, b) in got.data.iter().zip(reference(&lhs, &rhs)) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    /// The kernel skips zero multipliers; make sure that shortcut cannot leave
    /// stale accumulator state behind.
    #[test]
    fn cpu_matmul_handles_zero_rows() {
        let lhs = QnnFloatTensor::new(vec![0.0, 0.0, 1.0, 2.0], vec![2, 2]);
        let rhs = QnnFloatTensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
        let got = cpu_matmul(&lhs, &rhs);
        assert_eq!(got.data, vec![0.0, 0.0, 13.0, 16.0]);
    }

    #[test]
    fn matmul_falls_back_to_cpu_without_sdk() {
        // Without cfg(qnn_sdk) this is the only path; with it, qnn_matmul
        // declines small shapes and the result must still be correct.
        let lhs = QnnFloatTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let rhs = QnnFloatTensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        assert_eq!(matmul(&lhs, &rhs).data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
