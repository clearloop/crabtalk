//! Translate Anthropic Messages API requests into the internal
//! `ChatCompletionRequest` shape.
//!
//! # Known lossiness
//!
//! Fields we deliberately do not preserve yet:
//! - `metadata.user_id` and other top-level `metadata` fields.
//! - Thinking block `signature` (only load-bearing when replaying extended
//!   thinking back to the Anthropic upstream, which the inbound path does not
//!   do — our providers receive OpenAI-shaped reasoning_content).
//! - Non-text content inside `system` blocks (images get dropped).
//! - Non-text content inside `tool_result.content` when sent as blocks.
//! - `disable_parallel_tool_use` on `tool_choice`.

use crabllm_core::{
    AnthropicContent, AnthropicContentBlock, AnthropicMessage, AnthropicRequest, AnthropicResponse,
    AnthropicSystem, AnthropicTool, AnthropicUsage, ChatCompletionRequest, ChatCompletionResponse,
    Error, FinishReason, FunctionDef, Message, Role, Stop, Tool, ToolChoice, ToolType, Usage,
};

pub fn to_chat_completion(req: AnthropicRequest) -> ChatCompletionRequest {
    let mut messages = Vec::new();

    if let Some(system) = req.system {
        let text = match system {
            AnthropicSystem::Text(s) => s,
            AnthropicSystem::Blocks(blocks) => flatten_text_blocks(&blocks),
        };
        if !text.is_empty() {
            messages.push(Message::system(text));
        }
    }

    for msg in req.messages {
        append_message(&mut messages, msg);
    }

    let stop = req.stop_sequences.map(|mut seqs| {
        if seqs.len() == 1 {
            Stop::Single(seqs.pop().unwrap())
        } else {
            Stop::Multiple(seqs)
        }
    });

    let tools = req
        .tools
        .map(|tools| tools.into_iter().map(convert_tool).collect());
    let tool_choice = req.tool_choice.as_ref().and_then(translate_tool_choice);

    ChatCompletionRequest {
        model: req.model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        // Populate both fields. The Anthropic-direct upstream reads
        // `anthropic_max_tokens` (preserves the request-required
        // semantics on the way back out); every OpenAI-shape provider
        // reads `max_tokens`. Dropping the latter meant Anthropic-
        // translated requests reached non-Anthropic providers with no
        // explicit budget — fine for hosted APIs that have their own
        // defaults, brittle for local providers.
        max_tokens: Some(req.max_tokens),
        stream: req.stream,
        stop,
        tools,
        tool_choice,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        user: None,
        reasoning_effort: None,
        thinking: req.thinking,
        anthropic_max_tokens: Some(req.max_tokens),
        extra: serde_json::Map::new(),
    }
}

fn append_message(out: &mut Vec<Message>, msg: AnthropicMessage) {
    let role = match msg.role.as_str() {
        "assistant" => Role::Assistant,
        "user" => Role::User,
        _ => Role::Custom(msg.role),
    };
    let blocks = match msg.content {
        AnthropicContent::Text(s) => vec![AnthropicContentBlock::text(s)],
        AnthropicContent::Blocks(b) => b,
    };
    out.push(Message {
        role,
        content: blocks,
    });
}

fn flatten_text_blocks(blocks: &[AnthropicContentBlock]) -> String {
    let mut out = String::new();
    for block in blocks {
        if let AnthropicContentBlock::Text { text, .. } = block {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

fn convert_tool(t: AnthropicTool) -> Tool {
    let mut schema = t.input_schema;
    crabllm_provider::schema::inline_refs(&mut schema);
    crabllm_provider::schema::strip_fields(
        &mut schema,
        &[
            "propertyNames",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "const",
        ],
    );
    Tool {
        kind: ToolType::Function,
        function: FunctionDef {
            name: t.name,
            description: t.description,
            parameters: Some(schema),
        },
        strict: None,
    }
}

fn translate_tool_choice(value: &serde_json::Value) -> Option<ToolChoice> {
    let kind = value.get("type")?.as_str()?;
    match kind {
        "auto" => Some(ToolChoice::Auto),
        "any" => Some(ToolChoice::Required),
        "tool" => value
            .get("name")
            .and_then(|n| n.as_str())
            .map(|name| ToolChoice::Function {
                name: name.to_string(),
            }),
        "none" => Some(ToolChoice::Disabled),
        _ => None,
    }
}

/// Translate an internal `ChatCompletionResponse` back to the Anthropic
/// Messages API response wire shape.
///
/// Errors when the response has zero choices or missing usage — both indicate
/// a provider bug or transport failure, and papering over them with empty
/// defaults would corrupt billing and hide real problems.
pub fn from_chat_completion(resp: ChatCompletionResponse) -> Result<AnthropicResponse, Error> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| Error::Internal("provider returned zero choices".into()))?;
    let usage = resp
        .usage
        .ok_or_else(|| Error::Internal("provider returned no usage".into()))?;

    let stop_reason = choice.finish_reason.as_ref().map(finish_reason_to_stop);
    let mut content = choice.message.content;
    if content.is_empty() {
        content.push(AnthropicContentBlock::text(""));
    }

    Ok(AnthropicResponse {
        id: resp.id,
        r#type: "message".to_string(),
        role: "assistant".to_string(),
        model: resp.model,
        content,
        stop_reason,
        stop_sequence: None,
        usage: usage_to_anthropic(usage),
    })
}

fn finish_reason_to_stop(reason: &FinishReason) -> String {
    match reason {
        FinishReason::Stop => "end_turn".to_string(),
        FinishReason::Length => "max_tokens".to_string(),
        FinishReason::ToolCalls => "tool_use".to_string(),
        // Not a documented Anthropic value, but honesty beats papering over a
        // safety event. SDKs treat unknown stop_reason as a plain string.
        FinishReason::ContentFilter => "content_filter".to_string(),
        FinishReason::Custom(s) => s.clone(),
    }
}

fn usage_to_anthropic(u: Usage) -> AnthropicUsage {
    AnthropicUsage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        cache_read_input_tokens: u.prompt_cache_hit_tokens,
        cache_creation_input_tokens: u.prompt_cache_miss_tokens,
    }
}
