//! Translate between Anthropic Messages API and internal ChatCompletion types.
//!
//! The actual conversions are `From<AnthropicRequest> for ChatCompletionRequest`
//! and `TryFrom<ChatCompletionResponse> for AnthropicResponse`, both implemented
//! in `crabllm_core`. This module re-exports thin wrappers so existing handler
//! call-sites compile unchanged.

use crabllm_core::{AnthropicRequest, AnthropicResponse, ChatCompletionRequest, ChatCompletionResponse, Error};

pub fn to_chat_completion(req: AnthropicRequest) -> ChatCompletionRequest {
    ChatCompletionRequest::from(req)
}

pub fn from_chat_completion(resp: ChatCompletionResponse) -> Result<AnthropicResponse, Error> {
    AnthropicResponse::try_from(resp)
}
