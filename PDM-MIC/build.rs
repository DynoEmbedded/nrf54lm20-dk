use std::env;
use std::path::PathBuf;

fn main() {
    // Make memory.x available to the linker (cortex-m-rt's link.x includes it).
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::copy("memory.x", out.join("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");
}
