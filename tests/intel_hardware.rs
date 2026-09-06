//! Explicit OpenVINO hardware exercise (the general tests can use CPU fallback).
//! Run with BURN_NPU_TRACE=1 cargo test --release --features intel --test intel_hardware -- --ignored --nocapture
#![cfg(feature = "intel")]

use burn_npu::backends::intel::{openvino_matmul_npu as openvino_matmul, IntelFloatTensor};

#[test]
#[ignore = "requires an installed OpenVINO runtime; inspect trace for the selected device"]
fn openvino_matmul_matches_reference() {
    let (m, k, n) = (32, 64, 32);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 % 31) as f32 - 15.0) / 32.0)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 11 % 29) as f32 - 14.0) / 32.0)
        .collect();
    let result = openvino_matmul(
        &IntelFloatTensor::new(a.clone(), vec![m, k]),
        &IntelFloatTensor::new(b.clone(), vec![k, n]),
    )
    .expect("OpenVINO must execute; the Rust CPU fallback is not accepted here");
    assert_eq!(result.shape, vec![m, n]);
    let mut max_error = 0.0_f32;
    for i in 0..m {
        for j in 0..n {
            let expected: f32 = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
            let actual = result.data[i * n + j];
            assert!(actual.is_finite());
            max_error = max_error.max((actual - expected).abs());
            assert!(
                (actual - expected).abs() <= 0.005 + 0.005 * expected.abs(),
                "[{i},{j}] {actual} != {expected}"
            );
        }
    }
    println!("Maximum absolute error: {max_error}");
}

fn check_batched(shape: &[usize], salt: usize) {
    let (m, k, n) = (32, 64, 32);
    let batches: usize = shape.iter().product();
    let mut a_shape = shape.to_vec();
    a_shape.extend([m, k]);
    let mut b_shape = shape.to_vec();
    b_shape.extend([k, n]);
    let a: Vec<f32> = (0..batches * m * k)
        .map(|i| (((i * 7 + salt) % 31) as f32 - 15.0) / 37.0)
        .collect();
    let b: Vec<f32> = (0..batches * k * n)
        .map(|i| (((i * 11 + salt * 3) % 29) as f32 - 14.0) / 39.0)
        .collect();
    let result = openvino_matmul(
        &IntelFloatTensor::new(a.clone(), a_shape),
        &IntelFloatTensor::new(b.clone(), b_shape),
    )
    .expect("OpenVINO execution required");
    let mut expected_shape = shape.to_vec();
    expected_shape.extend([m, n]);
    assert_eq!(result.shape, expected_shape);
    for batch in 0..batches {
        for i in 0..m {
            for j in 0..n {
                let expected: f32 = (0..k)
                    .map(|p| a[batch * m * k + i * k + p] * b[batch * k * n + p * n + j])
                    .sum();
                let actual = result.data[batch * m * n + i * n + j];
                assert!(actual.is_finite());
                assert!(
                    (actual - expected).abs() <= 0.005 + 0.005 * expected.abs(),
                    "batch={batch}, [{i},{j}]: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires OpenVINO; inspect trace for NPU execution"]
fn batched_matmul_reuses_buffers_with_changing_inputs() {
    for salt in [0, 5, 17] {
        // Same flattened batch, different public shapes; plus separate cache keys.
        for shape in [&[][..], &[1], &[12], &[1, 12], &[2, 6], &[2]] {
            check_batched(shape, salt);
        }
    }
}

#[test]
#[ignore = "requires OpenVINO; inspect trace for NPU execution"]
fn cached_requests_do_not_mix_concurrent_inputs() {
    std::thread::scope(|scope| {
        for thread in 0..4 {
            scope.spawn(move || {
                for iteration in 0..3 {
                    check_batched(&[12], thread * 7 + iteration);
                }
            });
        }
    });
}

#[test]
#[ignore = "requires OpenVINO; tests the shared-storage Burn path"]
fn strided_and_offset_views_match_reference_on_openvino() {
    use burn_flex::FlexTensor;
    use burn_npu::backends::intel::openvino_matmul_flex;
    use burn_tensor::TensorData;
    let a: Vec<f32> = (0..12 * 64 * 35)
        .map(|i| (i % 37) as f32 / 47.0 - 0.3)
        .collect();
    let b: Vec<f32> = (0..12 * 64 * 35)
        .map(|i| (i % 31) as f32 / 43.0 - 0.3)
        .collect();
    let lhs = FlexTensor::from_data(TensorData::new(a, [12, 64, 35]))
        .narrow(2, 2, 32)
        .transpose(1, 2);
    let rhs = FlexTensor::from_data(TensorData::new(b, [12, 64, 35])).narrow(2, 1, 32);
    let actual = openvino_matmul_flex(&lhs, &rhs).expect("OpenVINO must execute");
    let lc = lhs.to_contiguous();
    let rc = rhs.to_contiguous();
    let (a, b, c) = (
        lc.storage::<f32>(),
        rc.storage::<f32>(),
        actual.storage::<f32>(),
    );
    for batch in 0..12 {
        for i in 0..32 {
            for j in 0..32 {
                let expected: f32 = (0..64)
                    .map(|p| a[batch * 32 * 64 + i * 64 + p] * b[batch * 64 * 32 + p * 32 + j])
                    .sum();
                let actual = c[batch * 32 * 32 + i * 32 + j];
                assert!(
                    actual.is_finite()
                        && (actual - expected).abs() < 0.005 + 0.005 * expected.abs()
                );
            }
        }
    }
}

#[test]
fn intel_tensor_clone_and_reshape_share_storage_without_aliasing_mutations() {
    use burn::tensor::Tensor;
    use burn_npu::{NpuBurnBackend as B, NpuBurnDevice};
    let raw = burn_npu::burn_backend::tensor::NpuFloatTensor::from_data(
        burn_tensor::TensorData::new(vec![1_f32, 2., 3., 4.], [2, 2]),
    );
    let view = raw.clone().reshape([4].into());
    assert!(std::sync::Arc::ptr_eq(&raw.data_arc(), &view.data_arc()));
    let device = NpuBurnDevice::Default;
    let a = Tensor::<B, 2>::from_floats([[1., 2.], [3., 4.]], &device);
    let b = a.clone().reshape([4]).add_scalar(10.0);
    assert_eq!(a.into_data().to_vec::<f32>().unwrap(), vec![1., 2., 3., 4.]);
    assert_eq!(
        b.into_data().to_vec::<f32>().unwrap(),
        vec![11., 12., 13., 14.]
    );
}

#[test]
#[ignore = "requires BURN_NPU_CONSTANT_WEIGHTS=1 and an Intel NPU"]
fn constant_weight_cache_tracks_copy_on_write_updates() {
    use burn_flex::FlexTensor;
    use burn_npu::backends::intel::openvino_matmul_flex;
    use burn_tensor::TensorData;
    assert_eq!(
        std::env::var("BURN_NPU_CONSTANT_WEIGHTS").as_deref(),
        Ok("1")
    );
    let a = FlexTensor::from_data(TensorData::new(
        (0..8 * 256)
            .map(|i| ((i * 7 % 31) as f32 - 15.) / 37.)
            .collect::<Vec<_>>(),
        [8, 256],
    ));
    let original = FlexTensor::from_data(TensorData::new(
        (0..256 * 256)
            .map(|i| ((i * 11 % 29) as f32 - 14.) / 39.)
            .collect::<Vec<_>>(),
        [256, 256],
    ));
    let verify = |weight: &FlexTensor| {
        let output = openvino_matmul_flex(&a, weight).expect("OpenVINO must execute");
        for i in 0..8 {
            for j in 0..256 {
                let expected: f32 = (0..256)
                    .map(|p| a.storage::<f32>()[i * 256 + p] * weight.storage::<f32>()[p * 256 + j])
                    .sum();
                let actual = output.storage::<f32>()[i * 256 + j];
                assert!(
                    actual.is_finite()
                        && (actual - expected).abs() < 0.005 + 0.005 * expected.abs()
                );
            }
        }
    };
    verify(&original);
    let mut updated = original.clone();
    for value in updated.storage_mut::<f32>() {
        *value *= -0.75;
    }
    verify(&updated);
    verify(&original);
    verify(&updated);
}

#[test]
#[ignore = "requires NPU attention graph support"]
fn fused_attention_matches_flex_with_bias_and_changed_inputs() {
    use burn_flex::{Flex, FlexTensor};
    use burn_npu::backends::intel::openvino_attention;
    use burn_tensor::{
        ops::{AttentionModuleOptions, ModuleOps},
        TensorData,
    };
    for salt in [0, 7] {
        let tensor = |offset: usize| {
            FlexTensor::from_data(TensorData::new(
                (0..2 * 16 * 32)
                    .map(|i| (((i * 7 + offset + salt) % 37) as f32 - 18.) / 29.)
                    .collect::<Vec<_>>(),
                [1, 2, 16, 32],
            ))
        };
        let (q, k, v) = (tensor(0), tensor(3), tensor(9));
        let mask = FlexTensor::from_data(TensorData::new(
            (0..16 * 16)
                .map(|i| if i % 16 > i / 16 { -1e9_f32 } else { 0. })
                .collect::<Vec<_>>(),
            [1, 1, 16, 16],
        ));
        for bias in [None, Some(mask)] {
            let options = AttentionModuleOptions {
                scale: Some(0.125),
                softcap: None,
                is_causal: false,
            };
            let expected = <Flex as ModuleOps<Flex>>::attention(
                q.clone(),
                k.clone(),
                v.clone(),
                None,
                bias.clone(),
                options,
            )
            .to_contiguous();
            let actual = openvino_attention(&q, &k, &v, bias.as_ref(), &options)
                .expect("NPU graph required");
            let mut max_error = 0_f32;
            for (&actual, &expected) in actual
                .storage::<f32>()
                .iter()
                .zip(expected.storage::<f32>())
            {
                max_error = max_error.max((actual - expected).abs());
                assert!(
                    actual.is_finite()
                        && (actual - expected).abs() < 0.005 + 0.005 * expected.abs(),
                    "{actual} != {expected}"
                );
            }
            println!("Fused attention maximum absolute error: {max_error}");
        }
    }
}

#[test]
fn attention_options_and_fully_masked_rows_keep_fallback_semantics() {
    use burn_flex::{Flex, FlexTensor};
    use burn_npu::NpuBurnBackend as B;
    use burn_tensor::{
        ops::{AttentionModuleOptions, ModuleOps},
        DType,
    };
    let q = FlexTensor::zeros([1, 1, 16, 32].into(), DType::F32);
    let k = q.clone();
    let v = FlexTensor::filled_typed([1, 1, 16, 32].into(), DType::F32, 0.25_f32);
    let bias = FlexTensor::filled_typed([1, 1, 16, 16].into(), DType::F32, f32::NEG_INFINITY);
    for (mask, bias, options) in [
        (None, Some(bias), AttentionModuleOptions::default()),
        (
            None,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: Some(2.),
                is_causal: true,
            },
        ),
        (
            Some(FlexTensor::zeros(
                [1, 1, 16, 16].into(),
                DType::Bool(burn_tensor::BoolStore::Native),
            )),
            None,
            AttentionModuleOptions::default(),
        ),
    ] {
        let expected = <Flex as ModuleOps<Flex>>::attention(
            q.clone(),
            k.clone(),
            v.clone(),
            mask.clone(),
            bias.clone(),
            options,
        )
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        let actual =
            <B as ModuleOps<B>>::attention(q.clone(), k.clone(), v.clone(), mask, bias, options)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn legacy_cpu_matmul_broadcasts_independent_batch_axes() {
    use burn_npu::backends::intel::cpu_matmul;
    let a = IntelFloatTensor::new(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 1, 2, 2]);
    let b = IntelFloatTensor::new(
        vec![1., 0., 0., 1., 2., 0., 0., 2., 3., 0., 0., 3.],
        vec![1, 3, 2, 2],
    );
    let c = cpu_matmul(&a, &b);
    assert_eq!(c.shape, vec![2, 3, 2, 2]);
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(
                    c.data[(i * 3 + j) * 4 + k],
                    a.data[i * 4 + k] * (j + 1) as f32
                );
            }
        }
    }
}

#[test]
fn invalid_low_level_shapes_are_rejected_without_runtime_work() {
    let invalid = IntelFloatTensor {
        data: vec![0.],
        shape: vec![usize::MAX, 2],
    };
    let rhs = IntelFloatTensor::new(vec![1.; 4], vec![2, 2]);
    assert!(openvino_matmul(&invalid, &rhs).is_err());
    let invalid = IntelFloatTensor {
        data: vec![0.],
        shape: vec![32, 64],
    };
    assert!(openvino_matmul(&invalid, &rhs).is_err());
}

#[test]
fn finite_large_matmul_keeps_cpu_semantics_if_npu_range_is_insufficient() {
    use burn::tensor::Tensor;
    use burn_npu::{NpuBurnBackend as B, NpuBurnDevice};
    let device = NpuBurnDevice::Default;
    let a = Tensor::<B, 2>::full([16, 64], 256., &device);
    let b = Tensor::<B, 2>::full([64, 16], 256., &device);
    assert!(a
        .matmul(b)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|&v| v == 4194304.));
}

#[test]
#[ignore = "requires NPU; verifies runtime linking and cache lifecycle on fresh threads"]
fn native_cache_hit_and_clear_work_on_fresh_threads() {
    let tensor = || IntelFloatTensor::new(vec![0.125; 32 * 32], vec![32, 32]);
    openvino_matmul(&tensor(), &tensor()).expect("NPU required");
    std::thread::spawn(move || {
        let out = openvino_matmul(&tensor(), &tensor()).expect("cache hit on fresh thread");
        assert!(out.data.iter().all(|&v| v == 0.5));
    })
    .join()
    .unwrap();
    std::thread::spawn(burn_npu::backends::intel::clear_caches)
        .join()
        .unwrap()
        .expect("clear on a fresh thread");
    assert_eq!(burn_npu::backends::intel::cache_stats().matmul.entries, 0);
}

#[test]
#[ignore = "requires NPU; verifies causal alignment and broadcast boolean masks"]
fn causal_cross_attention_and_boolean_masks_match_flex() {
    use burn_flex::{Flex, FlexTensor};
    use burn_npu::NpuBurnBackend as B;
    use burn_tensor::{
        ops::{AttentionModuleOptions, ModuleOps},
        TensorData,
    };
    let tensor = |seq, seed| {
        FlexTensor::from_data(TensorData::new(
            (0..2 * seq * 32)
                .map(|i| ((i * 7 + seed) % 37) as f32 / 37. - 0.5)
                .collect::<Vec<_>>(),
            [1, 2, seq, 32],
        ))
    };
    let (q, k, v) = (tensor(16, 1), tensor(32, 3), tensor(32, 5));
    let bias = FlexTensor::from_data(TensorData::new(
        (0..32).map(|i| i as f32 / 100.).collect::<Vec<_>>(),
        [1, 1, 1, 32],
    ));
    let mask = FlexTensor::from_data(TensorData::new(
        (0..16 * 32)
            .map(|i| i % 32 > 0 && i % 3 == 0)
            .collect::<Vec<_>>(),
        [1, 1, 16, 32],
    ));
    for causal in [false, true] {
        let options = AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: causal,
        };
        let expected = <Flex as ModuleOps<Flex>>::attention(
            q.clone(),
            k.clone(),
            v.clone(),
            Some(mask.clone()),
            Some(bias.clone()),
            options,
        )
        .to_contiguous();
        let before = burn_npu::backends::intel::execution_stats().npu_calls;
        let actual = <B as ModuleOps<B>>::attention(
            q.clone(),
            k.clone(),
            v.clone(),
            Some(mask.clone()),
            Some(bias.clone()),
            options,
        )
        .to_contiguous();
        assert!(
            burn_npu::backends::intel::execution_stats().npu_calls > before,
            "native attention was not used"
        );
        for (&a, &b) in actual
            .storage::<f32>()
            .iter()
            .zip(expected.storage::<f32>())
        {
            assert!((a - b).abs() < 0.005 + 0.005 * b.abs());
        }
    }
}

#[test]
#[ignore = "requires NPU; tests shared RHS broadcasting without host replication"]
fn shared_rhs_broadcast_runs_on_npu() {
    let a = IntelFloatTensor::new(
        (0..2 * 3 * 4 * 256)
            .map(|i| (i % 13) as f32 / 32.)
            .collect(),
        vec![2, 3, 4, 256],
    );
    let b = IntelFloatTensor::new(
        (0..256 * 256).map(|i| (i % 17) as f32 / 32.).collect(),
        vec![1, 1, 256, 256],
    );
    let out = openvino_matmul(&a, &b).expect("NPU required");
    assert_eq!(out.shape, vec![2, 3, 4, 256]);
    for i in 0..24 {
        for j in 0..256 {
            let expected: f32 = (0..256)
                .map(|k| a.data[i * 256 + k] * b.data[k * 256 + j])
                .sum();
            assert!((out.data[i * 256 + j] - expected).abs() < 0.005 + 0.005 * expected.abs());
        }
    }
    // Exercise the optional constant-weight path with the same batched shape.
    if std::env::var("BURN_NPU_CONSTANT_WEIGHTS").as_deref() == Ok("1") {
        use burn_tensor::{TensorData, TensorMetadata};
        let lhs = burn_flex::FlexTensor::from_data(TensorData::new(a.data, a.shape));
        let rhs = burn_flex::FlexTensor::from_data(TensorData::new(b.data, b.shape));
        let before = burn_npu::backends::intel::cache_stats().constant_weights;
        let actual = burn_npu::backends::intel::openvino_matmul_flex(&lhs, &rhs)
            .expect("shared constant RHS execution");
        let after = burn_npu::backends::intel::cache_stats().constant_weights;
        assert!(after.misses > before.misses);
        assert_eq!(after.build_failures, before.build_failures);
        assert_eq!(actual.shape().to_vec(), out.shape);
        for (&a, &b) in actual.storage::<f32>().iter().zip(&out.data) {
            assert!((a - b).abs() <= 0.005 + 0.005 * b.abs());
        }
    }
}

#[test]
fn attention_rejects_unrepresentable_operands_even_when_their_product_is_small() {
    use burn_flex::{Flex, FlexTensor};
    use burn_npu::NpuBurnBackend as B;
    use burn_tensor::{
        ops::{AttentionModuleOptions, ModuleOps},
        TensorData,
    };
    for (query_value, key_value, scale) in [(1e8_f32, 1e-8_f32, None), (0.0001, 0.0001, Some(1e8))]
    {
        let q = FlexTensor::from_data(TensorData::new(vec![query_value; 16 * 32], [1, 1, 16, 32]));
        let k = FlexTensor::from_data(TensorData::new(
            (0..16 * 32)
                .map(|i| if i < 8 * 32 { key_value } else { 0. })
                .collect::<Vec<_>>(),
            [1, 1, 16, 32],
        ));
        let v = FlexTensor::from_data(TensorData::new(
            (0..16 * 32)
                .map(|i| if i < 8 * 32 { 1_f32 } else { 0. })
                .collect::<Vec<_>>(),
            [1, 1, 16, 32],
        ));
        let options = AttentionModuleOptions {
            scale,
            softcap: None,
            is_causal: false,
        };
        let expected = <Flex as ModuleOps<Flex>>::attention(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            None,
            options,
        )
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        let actual = <B as ModuleOps<B>>::attention(q, k, v, None, None, options)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(actual, expected);
    }
}
