//! Hexagon NPU dispatch via the Qualcomm AI Engine Direct (QNN) C API.
//!
//! # Status
//!
//! **Implemented but not yet validated on hardware.** This module compiles only
//! when `build.rs` found a QAIRT SDK (`cfg(qnn_sdk)`), which requires
//! `QNN_SDK_ROOT` to be set at build time. It has been written against the
//! documented QNN C API but has never been compiled against real headers nor
//! executed on a Snapdragon device, because neither was available to the
//! author. Treat the first build on an SDK machine as the real bring-up: in
//! particular, the versioned-union accessors in [`interface_v1`] and
//! [`tensor_v1_mut`] are the places where bindgen's generated field names are
//! most likely to need adjusting.
//!
//! Every failure path returns [`QnnUnavailable`] so the caller falls back to
//! CPU. Nothing here can make the backend produce wrong answers; the worst case
//! is that it never leaves the CPU path.
//!
//! # Design
//!
//! Mirrors the OpenVINO integration: the backend library is opened at runtime
//! (so a build without a Snapdragon device still runs), one backend/device/
//! context is created lazily per process, and finalized graphs are cached by
//! matmul shape so repeated calls skip graph construction.

use super::{QnnFloatTensor, QnnUnavailable};
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{LazyLock, Mutex};

/// Bindings generated from the SDK headers by `build.rs`.
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    clippy::all
)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/qnn_bindings.rs"));
}

/// Candidate file names for the HTP (Hexagon Tensor Processor) backend library.
///
/// HTP is the NPU proper. The CPU backend is deliberately not in this list: if
/// HTP cannot be opened we want our own fallback, which keeps data in the
/// tensor representation the rest of the backend already uses.
const HTP_LIBRARIES: &[&str] = &["libQnnHtp.so", "QnnHtp.dll", "libQnnHtp.dylib"];

/// A loaded QNN backend, ready to build graphs.
struct QnnRuntime {
    /// Function-pointer table taken from the selected provider.
    iface: sys::QNN_INTERFACE_VER_TYPE,
    backend: sys::Qnn_BackendHandle_t,
    device: sys::Qnn_DeviceHandle_t,
    context: sys::Qnn_ContextHandle_t,
    /// Finalized matmul graphs, keyed by `(m, k, n)`.
    graphs: Mutex<HashMap<(usize, usize, usize), sys::Qnn_GraphHandle_t>>,
    /// Keeps the backend library mapped for as long as the handles live.
    _lib: libloading::Library,
}

// The QNN handles are opaque pointers owned by this struct and only reached
// through `&self` plus the graph mutex; the underlying API is documented as
// thread-safe for graph execution on a shared context.
unsafe impl Send for QnnRuntime {}
unsafe impl Sync for QnnRuntime {}

static RUNTIME: LazyLock<Option<QnnRuntime>> = LazyLock::new(|| match QnnRuntime::load() {
    Ok(rt) => Some(rt),
    Err(()) => None,
});

/// Extract the v1 function table from a provider's versioned union.
///
/// The QNN headers expose this through the `QNN_INTERFACE_VER_NAME` macro,
/// which bindgen cannot follow. If the field name below does not match the
/// generated bindings, this is the single place to correct it.
///
/// # Safety
/// `provider` must point at a live `QnnInterface_t` returned by
/// `QnnInterface_getProviders`.
unsafe fn interface_v1(provider: *const sys::QnnInterface_t) -> sys::QNN_INTERFACE_VER_TYPE {
    unsafe { (*provider).__bindgen_anon_1.v2_31 }
}

/// Mutable view of a tensor's v1 fields, for the same reason as above.
///
/// # Safety
/// `tensor` must point at a live, zero-initialised `Qnn_Tensor_t`.
unsafe fn tensor_v1_mut(tensor: *mut sys::Qnn_Tensor_t) -> *mut sys::Qnn_TensorV1_t {
    unsafe { &raw mut (*tensor).__bindgen_anon_1.v1 }
}

impl QnnRuntime {
    fn load() -> Result<Self, ()> {
        let lib = HTP_LIBRARIES
            .iter()
            .find_map(|name| unsafe { libloading::Library::new(name) }.ok())
            .ok_or(())?;

        // SAFETY: the symbol's signature is fixed by the QNN ABI and the
        // generated bindings describe the types it operates on.
        let get_providers: libloading::Symbol<
            unsafe extern "C" fn(*mut *const *const sys::QnnInterface_t, *mut u32) -> u64,
        > = unsafe { lib.get(b"QnnInterface_getProviders\0") }.map_err(|_| ())?;

        let mut providers: *const *const sys::QnnInterface_t = std::ptr::null();
        let mut count: u32 = 0;
        let status = unsafe { get_providers(&mut providers, &mut count) };
        if status != sys::QNN_SUCCESS as u64 || providers.is_null() || count == 0 {
            return Err(());
        }

        // Pick the first provider whose core API major version matches what we
        // generated bindings for. A major mismatch means the struct layouts
        // differ and using the table would be undefined behaviour.
        let iface = (0..count as usize)
            .filter_map(|i| {
                let provider = unsafe { *providers.add(i) };
                if provider.is_null() {
                    return None;
                }
                let version = unsafe { (*provider).apiVersion.coreApiVersion };
                (version.major == sys::QNN_API_VERSION_MAJOR)
                    .then(|| unsafe { interface_v1(provider) })
            })
            .next()
            .ok_or(())?;

        let mut backend: sys::Qnn_BackendHandle_t = std::ptr::null_mut();
        let mut device: sys::Qnn_DeviceHandle_t = std::ptr::null_mut();
        let mut context: sys::Qnn_ContextHandle_t = std::ptr::null_mut();

        unsafe {
            let create_backend = iface.backendCreate.ok_or(())?;
            if create_backend(std::ptr::null_mut(), std::ptr::null_mut(), &mut backend)
                != sys::QNN_SUCCESS as u64
            {
                return Err(());
            }

            // A device handle is optional on some backends; tolerate failure
            // and pass null through to contextCreate.
            if let Some(create_device) = iface.deviceCreate {
                if create_device(std::ptr::null_mut(), std::ptr::null_mut(), &mut device)
                    != sys::QNN_SUCCESS as u64
                {
                    device = std::ptr::null_mut();
                }
            }

            let create_context = iface.contextCreate.ok_or(())?;
            if create_context(backend, device, std::ptr::null_mut(), &mut context)
                != sys::QNN_SUCCESS as u64
            {
                return Err(());
            }
        }

        Ok(Self {
            iface,
            backend,
            device,
            context,
            graphs: Mutex::new(HashMap::new()),
            _lib: lib,
        })
    }
}

/// Fill in a float32 `Qnn_Tensor_t` describing one matmul operand.
///
/// # Safety
/// `tensor` must point at a zero-initialised `Qnn_Tensor_t` that outlives the
/// graph call it is passed to, as must `name` and `dims`.
unsafe fn describe_tensor(
    tensor: *mut sys::Qnn_Tensor_t,
    name: &CString,
    dims: &mut [u32],
    tensor_type: sys::Qnn_TensorType_t,
) {
    unsafe {
        let v1 = tensor_v1_mut(tensor);
        (*v1).id = 0;
        (*v1).name = name.as_ptr();
        (*v1).type_ = tensor_type;
        (*v1).dataFormat = sys::Qnn_TensorDataFormat_t::default();
        (*v1).dataType = sys::Qnn_DataType_t_QNN_DATATYPE_FLOAT_32;
        (*v1).rank = dims.len() as u32;
        (*v1).dimensions = dims.as_mut_ptr();
        (*v1).memType = sys::Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW;
    }
}

/// Point a tensor's client buffer at `data`.
///
/// # Safety
/// `tensor` must be a tensor already filled in by [`describe_tensor`], and
/// `data` must outlive the graph execution.
unsafe fn set_client_buf(tensor: *mut sys::Qnn_Tensor_t, data: &mut [f32]) {
    unsafe {
        let v1 = tensor_v1_mut(tensor);
        (*v1).clientBuf.data = data.as_mut_ptr().cast();
        (*v1).clientBuf.dataSize = std::mem::size_of_val(data) as u32;
    }
}

/// Run a 2D or batched matmul on the Hexagon NPU.
///
/// Batched inputs are executed as one graph per 2D slice, matching the shape
/// key used for caching. Returns [`QnnUnavailable`] whenever the runtime, the
/// device, or graph construction is not usable, so the caller falls back to
/// CPU.
pub fn qnn_matmul(
    lhs: &QnnFloatTensor,
    rhs: &QnnFloatTensor,
) -> Result<QnnFloatTensor, QnnUnavailable> {
    let rt = RUNTIME.as_ref().ok_or(QnnUnavailable)?;

    let lhs_ndim = lhs.shape.len();
    let rhs_ndim = rhs.shape.len();
    if lhs_ndim < 2 || rhs_ndim < 2 {
        return Err(QnnUnavailable);
    }

    let m = lhs.shape[lhs_ndim - 2];
    let k = lhs.shape[lhs_ndim - 1];
    let n = rhs.shape[rhs_ndim - 1];
    if rhs.shape[rhs_ndim - 2] != k {
        return Err(QnnUnavailable);
    }

    // Graph setup dominates for small problems; leave those on the CPU. Same
    // threshold as the OpenVINO path.
    if m * k * n < 4096 {
        return Err(QnnUnavailable);
    }

    let lhs_batch: Vec<usize> = lhs.shape[..lhs_ndim - 2].to_vec();
    let rhs_batch: Vec<usize> = rhs.shape[..rhs_ndim - 2].to_vec();
    if lhs_batch != rhs_batch {
        return Err(QnnUnavailable);
    }
    let batch: usize = lhs_batch.iter().product::<usize>().max(1);

    let graph = graph_for(rt, m, k, n)?;

    let mut out = vec![0.0f32; batch * m * n];
    for b in 0..batch {
        execute_slice(
            rt,
            graph,
            &lhs.data[b * m * k..(b + 1) * m * k],
            &rhs.data[b * k * n..(b + 1) * k * n],
            &mut out[b * m * n..(b + 1) * m * n],
            (m, k, n),
        )?;
    }

    let mut shape = lhs_batch;
    shape.push(m);
    shape.push(n);
    Ok(QnnFloatTensor::new(out, shape))
}

/// Build and finalize a matmul graph for one shape, or return the cached one.
fn graph_for(
    rt: &QnnRuntime,
    m: usize,
    k: usize,
    n: usize,
) -> Result<sys::Qnn_GraphHandle_t, QnnUnavailable> {
    let mut cache = rt.graphs.lock().map_err(|_| QnnUnavailable)?;
    if let Some(graph) = cache.get(&(m, k, n)) {
        return Ok(*graph);
    }

    let graph_name = CString::new(format!("matmul_{m}x{k}x{n}")).map_err(|_| QnnUnavailable)?;
    let lhs_name = CString::new("lhs").map_err(|_| QnnUnavailable)?;
    let rhs_name = CString::new("rhs").map_err(|_| QnnUnavailable)?;
    let out_name = CString::new("out").map_err(|_| QnnUnavailable)?;
    let node_name = CString::new("matmul").map_err(|_| QnnUnavailable)?;
    // Every built-in QNN op lives in this package.
    let package = CString::new("qti.aisw").map_err(|_| QnnUnavailable)?;
    let op_type = CString::new("MatMul").map_err(|_| QnnUnavailable)?;

    let mut lhs_dims = [m as u32, k as u32];
    let mut rhs_dims = [k as u32, n as u32];
    let mut out_dims = [m as u32, n as u32];

    let mut graph: sys::Qnn_GraphHandle_t = std::ptr::null_mut();

    unsafe {
        let create_graph = rt.iface.graphCreate.ok_or(QnnUnavailable)?;
        if create_graph(
            rt.context,
            graph_name.as_ptr(),
            std::ptr::null_mut(),
            &mut graph,
        ) != sys::QNN_SUCCESS as u64
        {
            return Err(QnnUnavailable);
        }

        let mut lhs_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        let mut rhs_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        let mut out_t: sys::Qnn_Tensor_t = std::mem::zeroed();

        describe_tensor(
            &mut lhs_t,
            &lhs_name,
            &mut lhs_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        );
        describe_tensor(
            &mut rhs_t,
            &rhs_name,
            &mut rhs_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        );
        describe_tensor(
            &mut out_t,
            &out_name,
            &mut out_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
        );

        let create_tensor = rt.iface.tensorCreateGraphTensor.ok_or(QnnUnavailable)?;
        for t in [&mut lhs_t, &mut rhs_t, &mut out_t] {
            if create_tensor(graph, t) != sys::QNN_SUCCESS as u64 {
                return Err(QnnUnavailable);
            }
        }

        let mut inputs = [lhs_t, rhs_t];
        let mut outputs = [out_t];

        let mut op: sys::Qnn_OpConfig_t = std::mem::zeroed();
        let op_v1 = &raw mut op.__bindgen_anon_1.v1;
        (*op_v1).name = node_name.as_ptr();
        (*op_v1).packageName = package.as_ptr();
        (*op_v1).typeName = op_type.as_ptr();
        (*op_v1).numOfParams = 0;
        (*op_v1).params = std::ptr::null_mut();
        (*op_v1).numOfInputs = inputs.len() as u32;
        (*op_v1).inputTensors = inputs.as_mut_ptr();
        (*op_v1).numOfOutputs = outputs.len() as u32;
        (*op_v1).outputTensors = outputs.as_mut_ptr();

        let add_node = rt.iface.graphAddNode.ok_or(QnnUnavailable)?;
        if add_node(graph, op) != sys::QNN_SUCCESS as u64 {
            return Err(QnnUnavailable);
        }

        let finalize = rt.iface.graphFinalize.ok_or(QnnUnavailable)?;
        if finalize(graph, std::ptr::null_mut(), std::ptr::null_mut()) != sys::QNN_SUCCESS as u64 {
            return Err(QnnUnavailable);
        }
    }

    cache.insert((m, k, n), graph);
    Ok(graph)
}

/// Execute one finalized graph over a single 2D slice.
fn execute_slice(
    rt: &QnnRuntime,
    graph: sys::Qnn_GraphHandle_t,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    (m, k, n): (usize, usize, usize),
) -> Result<(), QnnUnavailable> {
    let lhs_name = CString::new("lhs").map_err(|_| QnnUnavailable)?;
    let rhs_name = CString::new("rhs").map_err(|_| QnnUnavailable)?;
    let out_name = CString::new("out").map_err(|_| QnnUnavailable)?;

    let mut lhs_dims = [m as u32, k as u32];
    let mut rhs_dims = [k as u32, n as u32];
    let mut out_dims = [m as u32, n as u32];

    // QNN writes through the client buffers, so the inputs need to be owned
    // and mutable for the duration of the call.
    let mut lhs_buf = lhs.to_vec();
    let mut rhs_buf = rhs.to_vec();

    unsafe {
        let mut lhs_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        let mut rhs_t: sys::Qnn_Tensor_t = std::mem::zeroed();
        let mut out_t: sys::Qnn_Tensor_t = std::mem::zeroed();

        describe_tensor(
            &mut lhs_t,
            &lhs_name,
            &mut lhs_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        );
        describe_tensor(
            &mut rhs_t,
            &rhs_name,
            &mut rhs_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE,
        );
        describe_tensor(
            &mut out_t,
            &out_name,
            &mut out_dims,
            sys::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ,
        );

        set_client_buf(&mut lhs_t, &mut lhs_buf);
        set_client_buf(&mut rhs_t, &mut rhs_buf);
        set_client_buf(&mut out_t, out);

        let inputs = [lhs_t, rhs_t];
        let mut outputs = [out_t];

        let execute = rt.iface.graphExecute.ok_or(QnnUnavailable)?;
        if execute(
            graph,
            inputs.as_ptr(),
            inputs.len() as u32,
            outputs.as_mut_ptr(),
            outputs.len() as u32,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ) != sys::QNN_SUCCESS as u64
        {
            return Err(QnnUnavailable);
        }
    }

    Ok(())
}
