//! Non-Apple stub. Every call returns `Error::not_implemented` so the
//! crate compiles on Linux / Windows / etc. and downstream code that
//! references `crabllm_mlx::Session` behind a `cfg` still type-checks.
//!
//! The public surface must stay in lockstep with `session.rs` — any
//! type added there needs a mirror here, including the `Send + Sync`
//! bounds, or downstream code that stores an `Arc<Session>` in a
//! tokio task compiles on macOS and breaks on Linux CI.

use crabllm_core::Error;
use std::{path::Path, sync::atomic::AtomicU32};

#[derive(Debug, Clone, Copy, Default)]
pub struct GenerateOptions {
    pub seed: u64,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}

pub struct GenerateRequest<'a> {
    pub messages_json: &'a str,
    pub tools_json: Option<&'a str>,
    pub options: GenerateOptions,
    pub cancel_flag: Option<&'a AtomicU32>,
}

#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub text: String,
    pub tool_calls_json: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct StreamOutput {
    pub tool_calls_json: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

pub struct Session;

// Vacuously true: the stub never constructs a live session, so
// Send/Sync are safe. The bounds exist to match `session.rs` so the
// public API compiles identically on every target.
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    pub fn new(_model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        Err(Error::not_implemented(
            "mlx: only macOS and iOS (Apple Silicon) are supported",
        ))
    }

    pub fn generate(&self, _req: &GenerateRequest<'_>) -> Result<GenerateOutput, Error> {
        Err(Error::not_implemented("mlx: stub build"))
    }

    pub fn generate_stream<F>(
        &self,
        _req: &GenerateRequest<'_>,
        _on_token: F,
    ) -> Result<StreamOutput, Error>
    where
        F: FnMut(&str) -> bool,
    {
        Err(Error::not_implemented("mlx: stub build"))
    }
}
