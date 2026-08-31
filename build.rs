fn main() {
    // `qnn_sdk` is set by build_qnn() below when a QAIRT SDK is available.
    println!("cargo::rustc-check-cfg=cfg(qnn_sdk)");

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=framework=Accelerate");

    #[cfg(feature = "apple")]
    build_npu_sys();

    #[cfg(feature = "qualcomm")]
    build_qnn();
}

#[cfg(feature = "apple")]
fn build_npu_sys() {
    use std::process::Command;

    let npu_dir = std::path::Path::new("npu-sys");
    let src = npu_dir.join("Sources/npu_sys.swift");
    let lib = npu_dir.join("libnpu_sys.a");

    println!("cargo:rerun-if-changed={}", src.display());

    if !src.exists() {
        println!("cargo:warning=npu-sys/Sources/npu_sys.swift not found");
        return;
    }

    // Compile Swift to static library
    let status = Command::new("swiftc")
        .args([
            "-parse-as-library",
            "-emit-library",
            "-static",
            "-O",
            "-module-name",
            "npu_sys",
            &src.to_string_lossy(),
            "-o",
            &lib.to_string_lossy(),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            // Link the static library
            println!(
                "cargo:rustc-link-search=native={}",
                npu_dir.canonicalize().unwrap().display()
            );
            println!("cargo:rustc-link-lib=static=npu_sys");

            // Link Swift runtime (system) and CoreML
            println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
            println!("cargo:rustc-link-lib=framework=CoreML");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        _ => {
            println!("cargo:warning=Failed to compile npu-sys Swift library");
        }
    }
}

/// Generate QNN bindings when a Qualcomm AI Runtime (QAIRT) SDK is present.
///
/// The `qualcomm` feature deliberately does not vendor or hand-write the QNN
/// ABI. `Qnn_Tensor_t` is a versioned tagged union and `QnnInterface_t` is a
/// large function-pointer table; transcribing either by hand risks silent
/// memory corruption rather than a clean error. Instead bindings are generated
/// from the SDK's own headers, so they are correct by construction for
/// whichever SDK version the user builds against.
///
/// Set `QNN_SDK_ROOT` to the SDK root (the directory containing `include/QNN`).
/// Without it the feature still compiles and simply runs on CPU.
#[cfg(feature = "qualcomm")]
fn build_qnn() {
    use std::path::PathBuf;

    println!("cargo:rerun-if-env-changed=QNN_SDK_ROOT");

    let Some(root) = std::env::var_os("QNN_SDK_ROOT").map(PathBuf::from) else {
        println!(
            "cargo:warning=QNN_SDK_ROOT not set; the qualcomm feature will run on CPU. \
             Set it to a Qualcomm AI Runtime (QAIRT) SDK root to enable Hexagon NPU dispatch."
        );
        return;
    };

    let include = root.join("include").join("QNN");
    let header = include.join("QnnInterface.h");
    if !header.exists() {
        println!(
            "cargo:warning=QNN_SDK_ROOT is set but {} was not found; falling back to CPU.",
            header.display()
        );
        return;
    }
    println!("cargo:rerun-if-changed={}", header.display());

    // One translation unit pulling in the pieces the runtime touches. The
    // backend library itself is opened with libloading at runtime, so nothing
    // is linked here -- the SDK is a build-time input only.
    let wrapper = "\
        #include <QnnInterface.h>\n\
        #include <QnnBackend.h>\n\
        #include <QnnContext.h>\n\
        #include <QnnGraph.h>\n\
        #include <QnnTensor.h>\n\
        #include <QnnTypes.h>\n\
        #include <QnnCommon.h>\n";

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let wrapper_path = out_dir.join("qnn_wrapper.h");
    if let Err(e) = std::fs::write(&wrapper_path, wrapper) {
        println!("cargo:warning=could not write QNN wrapper header: {e}");
        return;
    }

    let bindings = bindgen::Builder::default()
        .header(wrapper_path.to_string_lossy())
        .clang_arg(format!("-I{}", include.display()))
        .allowlist_type("Qnn.*")
        .allowlist_function("Qnn.*")
        .allowlist_var("QNN_.*")
        .derive_debug(false)
        .layout_tests(false)
        .generate();

    match bindings {
        Ok(b) => match b.write_to_file(out_dir.join("qnn_bindings.rs")) {
            Ok(()) => {
                // Gates the runtime module in src/backends/qualcomm/qnn.rs.
                println!("cargo:rustc-cfg=qnn_sdk");
            }
            Err(e) => println!("cargo:warning=could not write QNN bindings: {e}"),
        },
        Err(e) => println!("cargo:warning=bindgen failed on the QNN headers: {e}"),
    }
}
