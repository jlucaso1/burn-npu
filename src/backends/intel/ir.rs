//! Shared OpenVINO IR-XML builders and model compilation. Graph topologies
//! stay at their call sites; only the envelope, ports, edges and the
//! Core/read/compile sequence are shared.
use super::{diagnostics, OpenVinoUnavailable};

pub(super) fn dims(shape: &[usize]) -> String {
    shape.iter().map(|d| format!("<dim>{d}</dim>")).collect()
}

pub(super) fn edge(from_layer: usize, from_port: usize, to_layer: usize, to_port: usize) -> String {
    format!(
        r#"<edge from-layer="{from_layer}" from-port="{from_port}" to-layer="{to_layer}" to-port="{to_port}"/>"#
    )
}

pub(super) fn net(name: &str, layers: &str, edges: &str) -> String {
    format!(
        r#"<?xml version="1.0"?><net name="{name}" version="11"><layers>{layers}</layers><edges>{edges}</edges></net>"#
    )
}

/// Generic FP32 layer. `out` is `None` for `Result`-style sinks.
pub(super) fn layer(
    id: usize,
    name: &str,
    kind: &str,
    version: &str,
    data: &str,
    inputs: &[&[usize]],
    out: Option<&[usize]>,
) -> String {
    let mut xml = format!(
        r#"<layer id="{id}" name="{name}" type="{kind}" version="{version}"><data {data}/><input>"#
    );
    for (port, shape) in inputs.iter().enumerate() {
        xml += &format!(
            r#"<port id="{port}" precision="FP32">{}</port>"#,
            dims(shape)
        );
    }
    xml += "</input>";
    if let Some(shape) = out {
        xml += &format!(
            r#"<output><port id="{}" precision="FP32" names="{name}">{}</port></output>"#,
            inputs.len(),
            dims(shape)
        );
    }
    xml += "</layer>";
    xml
}

pub(super) fn compile_on_device(
    core: &mut openvino::Core,
    xml: &str,
    weights: Option<&openvino::Tensor>,
    device: openvino::DeviceType,
    fail_ctx: &str,
) -> Result<openvino::CompiledModel, OpenVinoUnavailable> {
    let model = core
        .read_model_from_buffer(xml.as_bytes(), weights)
        .map_err(|_| OpenVinoUnavailable)?;
    core.compile_model(&model, device)
        .map_err(|err| diagnostics::failure(fail_ctx, err))
}
