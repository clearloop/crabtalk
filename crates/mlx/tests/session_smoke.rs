//! Error-path smoke tests for the safe `Session` FFI wrapper.
//!
//! Phase 5 replaced the Phase 2 dummy Swift body with real
//! mlx-swift-lm model loading, so the canned happy-path tests this
//! file used to carry no longer apply — loading a real model takes
//! real weights. End-to-end generation is covered by Rust integration
//! tests that pull a cached mlx-community model from disk (see
//! `crates/mlx/tests/model_smoke.rs`, gated on `MLX_LIVE=1`).
//!
//! What we keep here: the cheap invariants that must hold for every
//! build of the Swift package, regardless of whether a real model is
//! on disk. Anything that requires weights belongs in the live test.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use crabllm_mlx::Session;

#[test]
fn session_new_rejects_empty_path() {
    let err = match Session::new("") {
        Ok(_) => panic!("session_new(\"\") unexpectedly succeeded"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(msg.contains("mlx"), "error should mention mlx: {msg}");
}

#[test]
fn session_new_rejects_missing_directory() {
    let err = match Session::new("/definitely/does/not/exist/crabllm-mlx-smoke") {
        Ok(_) => panic!("session_new(nonexistent) unexpectedly succeeded"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(msg.contains("mlx"), "error should mention mlx: {msg}");
}

#[test]
fn session_new_rejects_file_not_directory() {
    // Any file in /tmp is not a directory; we use /etc/hosts because it
    // is present on every macOS and iOS simulator.
    let err = match Session::new("/etc/hosts") {
        Ok(_) => panic!("session_new(file) unexpectedly succeeded"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(msg.contains("mlx"), "error should mention mlx: {msg}");
}
