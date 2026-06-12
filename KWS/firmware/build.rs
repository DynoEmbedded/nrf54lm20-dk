use std::env;
use std::path::PathBuf;

// Axon buffer sizes (bytes). Must cover the model's reported
// `interlayer_buffer_needed` / `psum_buffer_needed` (printed by the Axon
// compiler). Keep in sync with the matching constants in src/main.rs (the C
// side declares the extern arrays at this size; Rust defines the storage).
// DS-CNN kws model: compiler reports interlayer needed 13548, psum needed 0.
const INTERLAYER_BUFFER_SIZE: &str = "16384";
const PSUM_BUFFER_SIZE: &str = "4096";

// Nordic driver blob + open C wrappers + headers, shared with the npu/ crate
// (vendor/ there is populated from the sdk-edge-ai add-on; see npu/README.md).
const VENDOR: &str = "../../npu/vendor";

/// Return the model name from the first `nrf_axon_model_<name>_.h` in `dir`,
/// matching the `model_<name>` symbol the Axon compiler emits.
fn find_generated_model(dir: &str) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_str()?;
        if let Some(inner) = name
            .strip_prefix("nrf_axon_model_")
            .and_then(|s| s.strip_suffix("_.h"))
        {
            if !inner.is_empty() {
                return Some(inner.to_string());
            }
        }
    }
    None
}

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let vendor = crate_dir.join(VENDOR);
    let vendor_inc = vendor.join("include");
    let vendor_drv = vendor.join("include/drivers");

    // Make memory.x available to the linker (cortex-m-rt's link.x includes it).
    std::fs::copy("memory.x", out.join("memory.x")).unwrap();
    // cortex-m-rt's "device" feature INCLUDEs a device.x (normally from a PAC).
    // Our vector table in main.rs is self-contained, so an empty one is fine.
    std::fs::write(
        out.join("device.x"),
        "/* interrupt vectors resolved via __INTERRUPTS in src/main.rs */\n",
    )
    .unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");

    // Compile the open-source Axon driver wrappers (high-level inference API +
    // CPU op extensions) plus the variadic-printf glue.
    cc::Build::new()
        .compiler("arm-none-eabi-gcc")
        .flag("-ffreestanding")
        .opt_level_str("s")
        .include(&vendor_inc)
        .include(&vendor_drv)
        .define("NRF_AXON_INTERLAYER_BUFFER_SIZE", INTERLAYER_BUFFER_SIZE)
        .define("NRF_AXON_PSUM_BUFFER_SIZE", PSUM_BUFFER_SIZE)
        .file(vendor.join("src/nrf_axon_nn_infer.c"))
        .file(vendor.join("src/nrf_axon_nn_op_extensions.c"))
        .file("csrc/glue.c")
        .compile("axon_src");

    // Link Nordic's pre-compiled low-level driver blob.
    println!("cargo:rustc-link-search=native={}", vendor.join("lib").display());
    println!("cargo:rustc-link-lib=static=nrf-axon-driver-internal-fpu");

    // Compiled model header (from npu/tools/compile-model.sh, copied into
    // generated/): build a glue TU exposing `axon_active_model()`.
    println!("cargo:rustc-check-cfg=cfg(has_model)");
    println!("cargo:rerun-if-changed=generated");
    if let Some(model_name) = find_generated_model("generated") {
        let glue = out.join("model_glue.c");
        std::fs::write(
            &glue,
            format!(
                "#include <assert.h>\n\
                 #include <stddef.h>\n\
                 #include \"axon/nrf_axon_platform.h\"\n\
                 #include \"drivers/axon/nrf_axon_nn_infer.h\"\n\
                 #include \"generated/nrf_axon_model_{model_name}_.h\"\n\
                 const nrf_axon_nn_compiled_model_s *axon_active_model(void) {{\n\
                 \treturn &model_{model_name};\n\
                 }}\n"
            ),
        )
        .unwrap();
        cc::Build::new()
            .compiler("arm-none-eabi-gcc")
            .flag("-ffreestanding")
            .flag("-std=c11")
            // Nordic's generated headers use `const static` ordering.
            .flag("-Wno-old-style-declaration")
            .opt_level_str("s")
            .include(&vendor_inc)
            .include(&vendor_drv)
            .include(&crate_dir)
            .define("NRF_AXON_INTERLAYER_BUFFER_SIZE", INTERLAYER_BUFFER_SIZE)
            .define("NRF_AXON_PSUM_BUFFER_SIZE", PSUM_BUFFER_SIZE)
            .file(&glue)
            .compile("axon_model");
        println!("cargo:rustc-cfg=has_model");
        println!("cargo:warning=Axon model linked: {model_name}");
    }

    // Self-test vectors emitted by training/convert.py.
    println!("cargo:rustc-check-cfg=cfg(has_testvec)");
    if crate_dir.join("generated/testvec.rs").exists() {
        println!("cargo:rustc-cfg=has_testvec");
    }

    // Generate Rust bindings for the API we call.
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", vendor_inc.display()))
        .clang_arg(format!("-I{}", vendor_drv.display()))
        .clang_arg("--target=thumbv8m.main-none-eabihf")
        .clang_arg("-ffreestanding")
        .clang_arg(format!("-DNRF_AXON_INTERLAYER_BUFFER_SIZE={INTERLAYER_BUFFER_SIZE}"))
        .clang_arg(format!("-DNRF_AXON_PSUM_BUFFER_SIZE={PSUM_BUFFER_SIZE}"))
        .use_core()
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .allowlist_function("nrf_axon_.*")
        .allowlist_type("nrf_axon_.*")
        .allowlist_var("NRF_AXON_[A-Z].*")
        .generate()
        .expect("bindgen failed to generate Axon bindings");
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=csrc/glue.c");
}
