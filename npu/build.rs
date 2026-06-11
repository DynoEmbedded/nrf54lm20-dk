use std::env;
use std::path::PathBuf;

// Axon buffer sizes (bytes). Must be large enough for your largest model's
// `interlayer_buffer_needed` / `psum_buffer_needed`. Keep in sync with the
// matching constants in src/main.rs (the C side declares the extern arrays at
// this size; Rust defines the actual storage).
const INTERLAYER_BUFFER_SIZE: &str = "2048";
const PSUM_BUFFER_SIZE: &str = "2048";

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
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Make memory.x available to the linker (cortex-m-rt's link.x includes it).
    std::fs::copy("memory.x", out.join("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");

    // Compile the open-source Axon driver wrappers (high-level inference API +
    // CPU op extensions) plus the variadic-printf glue.
    // cc derives -march/-mthumb/float-abi from the thumbv8m.main-none-eabihf
    // target triple; we only add freestanding.
    cc::Build::new()
        .compiler("arm-none-eabi-gcc")
        .flag("-ffreestanding")
        .opt_level_str("s")
        .include("vendor/include")
        .include("vendor/include/drivers")
        .define("NRF_AXON_INTERLAYER_BUFFER_SIZE", INTERLAYER_BUFFER_SIZE)
        .define("NRF_AXON_PSUM_BUFFER_SIZE", PSUM_BUFFER_SIZE)
        .file("vendor/src/nrf_axon_nn_infer.c")
        .file("vendor/src/nrf_axon_nn_op_extensions.c")
        .file("csrc/glue.c")
        .compile("axon_src");

    // Link Nordic's pre-compiled low-level driver blob (the register + ISA logic
    // that is not publicly documented).
    println!("cargo:rustc-link-search=native={crate_dir}/vendor/lib");
    println!("cargo:rustc-link-lib=static=nrf-axon-driver-internal-fpu");

    // If a compiled model header has been dropped into vendor/include/generated/
    // (by tools/compile-model.sh), generate a small glue TU that includes it and
    // exposes a stable `axon_active_model()` accessor, then compile it. The
    // generated header relies on the includer for the Axon headers + static_assert,
    // so it must be compiled this way rather than bound directly.
    println!("cargo:rustc-check-cfg=cfg(has_model)");
    println!("cargo:rerun-if-changed=vendor/include/generated");
    if let Some(model_name) = find_generated_model("vendor/include/generated") {
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
            .include("vendor/include")
            .include("vendor/include/drivers")
            .define("NRF_AXON_INTERLAYER_BUFFER_SIZE", INTERLAYER_BUFFER_SIZE)
            .define("NRF_AXON_PSUM_BUFFER_SIZE", PSUM_BUFFER_SIZE)
            .file(&glue)
            .compile("axon_model");
        println!("cargo:rustc-cfg=has_model");
        println!("cargo:warning=Axon model linked: {model_name}");
    }

    // Generate Rust bindings for the API we call.
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-Ivendor/include")
        .clang_arg("-Ivendor/include/drivers")
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
    println!("cargo:rerun-if-changed=vendor/src/nrf_axon_nn_infer.c");
    println!("cargo:rerun-if-changed=vendor/src/nrf_axon_nn_op_extensions.c");
}
