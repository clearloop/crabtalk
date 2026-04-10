//! Non-Apple stub. Every call returns `Error::not_implemented`.

use crabllm_core::{
    BoxStream, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Error, Provider,
};
use std::{path::Path, sync::Arc, sync::atomic::AtomicU32};

const STUB_MSG: &str = "mlx: only macOS and iOS (Apple Silicon) are supported";

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
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    pub fn new(_model_dir: impl AsRef<Path>) -> Result<Self, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    pub fn generate(&self, _req: &GenerateRequest<'_>) -> Result<GenerateOutput, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    pub fn generate_stream<F>(
        &self,
        _req: &GenerateRequest<'_>,
        _on_token: F,
    ) -> Result<StreamOutput, Error>
    where
        F: FnMut(&str) -> bool,
    {
        Err(Error::not_implemented(STUB_MSG))
    }
}

pub struct MlxPool;
unsafe impl Send for MlxPool {}
unsafe impl Sync for MlxPool {}

impl MlxPool {
    pub fn new(_idle_timeout_secs: u64) -> Result<Self, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    pub fn evict(&self, _model_dir: &str) {}
    pub fn stop_all(&self) {}
}

#[derive(Clone, Debug)]
pub struct MlxProvider {
    _pool: Arc<MlxPool>,
}

impl MlxProvider {
    pub fn new(pool: Arc<MlxPool>) -> Self {
        Self { _pool: pool }
    }
}

impl Provider for MlxProvider {
    async fn chat_completion(
        &self,
        _request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }

    async fn chat_completion_stream(
        &self,
        _request: &ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Error>>, Error> {
        Err(Error::not_implemented(STUB_MSG))
    }
}
