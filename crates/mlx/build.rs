// build.rs — build the `mlx/` Swift static library and link it in.
//
// The Swift package lives at the workspace root at `mlx/`; this file
// shells out to `swift build -c release --package-path <repo-root>/mlx`
// every cargo build, and emits link directives for the resulting
// `libCrabllmMlx.a` plus the Swift runtime lookup path.
//
// The build is pinned to the `release` SwiftPM configuration regardless
// of the Rust profile — debug Swift builds link fine against Rust
// release binaries, and swapping configs per Cargo profile would force
// a full Swift rebuild on every `cargo build` flip.
//
// On non-Apple targets the whole thing no-ops. `src/lib.rs` gates the
// real FFI on the same target predicate and falls back to a stub so the
// workspace still builds on Linux CI.

use std::{env, fs, path::Path, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SDKROOT");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" && target_os != "ios" {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("crate should live at <workspace>/crates/<name>");
    let mlx_dir = workspace_root.join("mlx");
    let build_dir = mlx_dir.join(".build").join("release");

    // Explicitly emit rerun-if-changed for every tracked input. Cargo's
    // directory-level rerun is *not* recursive — it only checks mtime
    // of the directory entry itself, which does not change when a file
    // inside is edited. Globbing the Swift sources is the only way to
    // actually retrigger a rebuild on edit.
    println!(
        "cargo:rerun-if-changed={}",
        mlx_dir.join("Package.swift").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        mlx_dir.join("include").join("crabllm_mlx.h").display()
    );
    for entry in walk_dir(&mlx_dir.join("Sources").join("CrabllmMlx")) {
        println!("cargo:rerun-if-changed={}", entry.display());
    }

    let status = Command::new("swift")
        .args(["build", "-c", "release"])
        .current_dir(&mlx_dir)
        .status()
        .expect("failed to invoke `swift build` — is the Swift toolchain installed?");
    if !status.success() {
        panic!("swift build -c release failed in {}", mlx_dir.display());
    }

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=CrabllmMlx");

    // Swift runtime: the dylibs live under the platform SDK's
    // usr/lib/swift. Pick the right SDK for the target OS — macOS uses
    // the default SDK, iOS needs `--sdk iphoneos` to get the device
    // runtime (and `iphonesimulator` for the simulator, which we
    // surface here if the Rust target triple says so).
    let sdk_flag = match (target_os.as_str(), sim_target()) {
        ("ios", true) => Some("iphonesimulator"),
        ("ios", false) => Some("iphoneos"),
        _ => None,
    };
    let mut xcrun = Command::new("xcrun");
    if let Some(sdk) = sdk_flag {
        xcrun.args(["--sdk", sdk]);
    }
    let sdk_output = xcrun
        .args(["--show-sdk-path"])
        .output()
        .expect("failed to run `xcrun --show-sdk-path`");
    if !sdk_output.status.success() {
        panic!("`xcrun --show-sdk-path` failed: {sdk_output:?}");
    }
    let sdk_path = String::from_utf8(sdk_output.stdout)
        .expect("xcrun returned non-UTF-8 path")
        .trim()
        .to_string();
    println!("cargo:rustc-link-search=native={sdk_path}/usr/lib/swift");

    // Foundation pulls in the Swift runtime symbols the static library
    // references. Phase 5 will additionally need Metal and
    // MetalPerformanceShaders once mlx-swift-lm is wired in.
    println!("cargo:rustc-link-lib=framework=Foundation");
}

/// True if the current target is an iOS simulator (not a device). We
/// detect this from the Rust target triple rather than a CARGO_CFG var
/// because Cargo does not expose a dedicated "simulator" cfg.
fn sim_target() -> bool {
    let target = env::var("TARGET").unwrap_or_default();
    target.ends_with("-apple-ios-sim") || target.contains("ios-sim")
}

/// Recursively walk a directory and yield every regular file. Used to
/// emit an accurate rerun-if-changed list for the Swift sources.
fn walk_dir(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if let Ok(file_type) = entry.file_type() {
                if file_type.is_dir() {
                    stack.push(path);
                } else if file_type.is_file() {
                    out.push(path);
                }
            }
        }
    }
    out
}
