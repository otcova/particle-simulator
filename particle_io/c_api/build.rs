use std::{env, path::PathBuf};

use cbindgen::Language;

fn main() {
    // println!("cargo:rerun-if-changed=src/");

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .include_item("DataStructure")
        .include_item("Device")
        .with_parse_include(&["particle_io"])
        .with_parse_deps(true)
        .with_language(Language::C)
        .with_no_includes()
        .with_pragma_once(true)
        .with_sys_include("stdint.h")
        .with_sys_include("stdbool.h")
        .with_cpp_compat(true)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(crate_dir.join("include/particle_io.h"));
}
