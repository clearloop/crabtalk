//! Round-trip the Rust safe wrapper against the Phase 2 Swift dummy
//! static library. Proves the FFI bindings, struct layout, trampoline,
//! and ResultGuard drop ordering all behave.
//!
//! Phase 5 replaces the dummy Swift body with real mlx-swift-lm, at
//! which point `/tmp/dummy-model` stops working and the test either
//! moves to a cached mlx-community model or gets rewritten.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use crabllm_mlx::{GenerateOptions, GenerateRequest, Session};

const DUMMY_MESSAGES: &str = r#"[{"role":"user","content":"hi"}]"#;

fn dummy_session() -> Session {
    Session::new("/tmp/dummy-model").expect("session_new against dummy stub")
}

fn dummy_request<'a>(messages: &'a str) -> GenerateRequest<'a> {
    GenerateRequest {
        messages_json: messages,
        tools_json: None,
        options: GenerateOptions::default(),
        cancel_flag: None,
    }
}

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
fn generate_returns_dummy_output() {
    let session = dummy_session();
    let req = dummy_request(DUMMY_MESSAGES);
    let out = session.generate(&req).expect("generate");
    assert!(
        out.text.starts_with("dummy: echoing"),
        "unexpected text: {}",
        out.text
    );
    assert!(out.text.contains(r#""role":"user""#));
    assert_eq!(out.prompt_tokens, 4);
    assert!(out.completion_tokens > 0);
    assert!(out.tool_calls_json.is_none());
}

#[test]
fn generate_stream_fires_canned_tokens() {
    let session = dummy_session();
    let req = dummy_request(DUMMY_MESSAGES);
    let mut tokens = Vec::new();
    let out = session
        .generate_stream(&req, |tok| {
            tokens.push(tok.to_string());
            false
        })
        .expect("generate_stream");
    assert_eq!(tokens, vec!["hello", " from", " swift", " stub"]);
    assert_eq!(out.prompt_tokens, 4);
    assert!(out.completion_tokens > 0);
}

#[test]
fn generate_stream_honors_early_stop() {
    let session = dummy_session();
    let req = dummy_request(DUMMY_MESSAGES);
    let mut count = 0;
    let _ = session
        .generate_stream(&req, |_| {
            count += 1;
            count >= 2
        })
        .expect("generate_stream early stop");
    assert_eq!(count, 2, "callback should have been called exactly twice");
}
