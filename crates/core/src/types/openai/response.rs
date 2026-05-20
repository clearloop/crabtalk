use crate::types::openai::{FinishReason, Message, ToolType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

impl ChatCompletionResponse {
    /// First choice's message, if present.
    pub fn message(&self) -> Option<&Message> {
        self.choices.first().map(|c| &c.message)
    }

    /// Text content from the first choice's message, if non-empty.
    pub fn content(&self) -> Option<&str> {
        self.choices.first()?.message.content_str()
    }

    /// Reasoning content from the first choice's message, if non-empty.
    pub fn reasoning_content(&self) -> Option<&str> {
        self.choices.first()?.message.thinking()
    }

    /// Finish reason from the first choice, if present.
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.choices.first()?.finish_reason.as_ref()
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_hit_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_miss_tokens: Option<u32>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionCall,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}
