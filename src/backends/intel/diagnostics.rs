//! Process-wide counters; snapshots are approximate while other threads execute.
use super::OpenVinoUnavailable;
use std::fmt::Display;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Target {
    Npu,
    Gpu,
    Cpu,
}
static NPU: AtomicU64 = AtomicU64::new(0);
static GPU: AtomicU64 = AtomicU64::new(0);
static CPU: AtomicU64 = AtomicU64::new(0);
static FLEX: AtomicU64 = AtomicU64::new(0);
static ERRORS: AtomicU64 = AtomicU64::new(0);
static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);
/// Successful native inference calls and fallback decisions. A matmul/batched
/// graph counts as one call, regardless of the number of internal kernels.
#[derive(Clone, Debug, Default)]
pub struct ExecutionStats {
    pub npu_calls: u64,
    pub gpu_calls: u64,
    pub openvino_cpu_calls: u64,
    pub flex_fallbacks: u64,
    pub runtime_errors: u64,
    pub last_error: Option<String>,
}
pub(super) fn snapshot() -> ExecutionStats {
    ExecutionStats {
        npu_calls: NPU.load(Ordering::Relaxed),
        gpu_calls: GPU.load(Ordering::Relaxed),
        openvino_cpu_calls: CPU.load(Ordering::Relaxed),
        flex_fallbacks: FLEX.load(Ordering::Relaxed),
        runtime_errors: ERRORS.load(Ordering::Relaxed),
        last_error: LAST_ERROR.lock().unwrap_or_else(|e| e.into_inner()).clone(),
    }
}
pub(super) fn executed(target: Target) {
    match target {
        Target::Npu => &NPU,
        Target::Gpu => &GPU,
        Target::Cpu => &CPU,
    }
    .fetch_add(1, Ordering::Relaxed);
}
pub(crate) fn fallback() {
    FLEX.fetch_add(1, Ordering::Relaxed);
}
pub(super) fn failure(context: &str, error: impl Display) -> OpenVinoUnavailable {
    let message = format!("{context}: {error}");
    ERRORS.fetch_add(1, Ordering::Relaxed);
    *LAST_ERROR.lock().unwrap_or_else(|e| e.into_inner()) = Some(message.clone());
    if super::trace() {
        eprintln!("{message}");
    }
    OpenVinoUnavailable
}
