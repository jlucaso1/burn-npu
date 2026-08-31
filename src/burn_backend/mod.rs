//! Burn `Backend` implementation for NPU.
//!
//! Platform-specific float tensor primitives:
//!
//! - **`apple`**: `NpuFloatTensor` wraps an `i32` MLTensor handle. All float ops
//!   pass handles through FFI; no data leaves the NPU between ops.
//! - **`intel`**: `NpuFloatTensor` is `IntelFloatTensor` (`Vec<f32>` + shape).
//!   Matmul attempts OpenVINO NPU dispatch; all other ops run on CPU or delegate
//!   to burn-ndarray.
//! - **`qualcomm`**: `NpuFloatTensor` is `QnnFloatTensor` (`Vec<f32>` + shape).
//!   All ops currently run on CPU. Ready for QNN SDK integration.
//! - **no feature**: `NpuFloatTensor` is `NdArrayTensor` (pure CPU fallback).
//!
//! Int/Bool tensor primitives always remain `NdArrayTensor` (delegated to burn-ndarray).

extern crate alloc;

mod bool_ops;
mod ffi;
mod float_ops;
mod int_ops;
mod module_ops;
mod quantization_ops;
pub mod tensor;

use alloc::string::String;
use burn_ndarray::{NdArray, NdArrayDevice, NdArrayQTensor, NdArrayTensor};
use burn_tensor::backend::{Backend, DeviceId, DeviceOps};
use burn_tensor::ops::*;
use burn_tensor::DType;

#[cfg(any(feature = "apple", feature = "intel", feature = "qualcomm"))]
pub use tensor::NpuFloatTensor;

// ---------------------------------------------------------------------------
// Type alias for the NdArray backend we delegate to.
// ---------------------------------------------------------------------------
pub(super) type Nd = NdArray<f32, i64, i8>;

// ---------------------------------------------------------------------------
// NpuBurnDevice
// ---------------------------------------------------------------------------
/// Device type for the NPU burn backend. There is only one logical device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NpuBurnDevice {
    /// The default device (routes to ANE when available, falls back to CPU).
    #[default]
    Default,
}

impl DeviceOps for NpuBurnDevice {}

impl burn_tensor::backend::Device for NpuBurnDevice {
    fn from_id(_device_id: DeviceId) -> Self {
        Self::Default
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 1,
            index_id: 0,
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// NpuBurnBackend
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Default, Debug)]
pub struct NpuBurnBackend;

/// Helper: map NpuBurnDevice -> NdArrayDevice for forwarding.
#[inline(always)]
pub(super) fn nd_dev() -> NdArrayDevice {
    NdArrayDevice::Cpu
}

// ===========================================================================
// Backend impl — apple feature: FloatTensorPrimitive = NpuFloatTensor
// ===========================================================================
#[cfg(feature = "apple")]
impl Backend for NpuBurnBackend {
    type Device = NpuBurnDevice;

    type FloatTensorPrimitive = NpuFloatTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = NdArrayTensor;
    type IntElem = i64;

    type BoolTensorPrimitive = NdArrayTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = NdArrayQTensor;

    fn ad_enabled() -> bool {
        false
    }

    fn name(_device: &Self::Device) -> String {
        String::from("Apple ANE")
    }

    fn seed(_device: &Self::Device, seed: u64) {
        <Nd as Backend>::seed(&NdArrayDevice::Cpu, seed);
    }

    fn supports_dtype(_device: &Self::Device, dtype: DType) -> bool {
        <Nd as Backend>::supports_dtype(&NdArrayDevice::Cpu, dtype)
    }
}

// ===========================================================================
// Backend impl — no feature: full NdArray delegation
// ===========================================================================
#[cfg(not(any(feature = "apple", feature = "intel", feature = "qualcomm")))]
impl Backend for NpuBurnBackend {
    type Device = NpuBurnDevice;

    type FloatTensorPrimitive = NdArrayTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = NdArrayTensor;
    type IntElem = i64;

    type BoolTensorPrimitive = NdArrayTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = NdArrayQTensor;

    fn ad_enabled() -> bool {
        false
    }

    fn name(_device: &Self::Device) -> String {
        String::from("CPU fallback")
    }

    fn seed(_device: &Self::Device, seed: u64) {
        <Nd as Backend>::seed(&NdArrayDevice::Cpu, seed);
    }

    fn supports_dtype(_device: &Self::Device, dtype: DType) -> bool {
        <Nd as Backend>::supports_dtype(&NdArrayDevice::Cpu, dtype)
    }
}

// ===========================================================================
// Backend impl — intel/qualcomm: FloatTensorPrimitive = NpuFloatTensor (Vec<f32>)
// ===========================================================================
#[cfg(any(feature = "intel", feature = "qualcomm"))]
impl Backend for NpuBurnBackend {
    type Device = NpuBurnDevice;

    type FloatTensorPrimitive = NpuFloatTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = NdArrayTensor;
    type IntElem = i64;

    type BoolTensorPrimitive = NdArrayTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = NdArrayQTensor;

    fn ad_enabled() -> bool {
        false
    }

    fn name(_device: &Self::Device) -> String {
        #[cfg(feature = "intel")]
        {
            String::from("Intel NPU")
        }
        #[cfg(feature = "qualcomm")]
        {
            String::from("Qualcomm Hexagon")
        }
    }

    fn seed(_device: &Self::Device, seed: u64) {
        <Nd as Backend>::seed(&NdArrayDevice::Cpu, seed);
    }

    fn supports_dtype(_device: &Self::Device, dtype: DType) -> bool {
        <Nd as Backend>::supports_dtype(&NdArrayDevice::Cpu, dtype)
    }
}

// ===========================================================================
// ActivationOps (all methods have defaults)
// ===========================================================================
impl ActivationOps<Self> for NpuBurnBackend {}

// ===========================================================================
// TransactionOps (all methods have defaults)
// ===========================================================================
impl TransactionOps<Self> for NpuBurnBackend {}
