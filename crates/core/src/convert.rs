use crate::{
    AnthropicContent, AnthropicContentBlock, AnthropicRequest, AnthropicResponse, AnthropicSystem,
    AnthropicUsage, ChatCompletionRequest, ChatCompletionResponse, Error, FinishReason, FunctionDef,
    Message, Role, Stop, Tool, ToolChoice, ToolType,
};

impl From<AnthropicRequest> for ChatCompletionRequest {
    fn from(req: AnthropicRequest) -> Self {
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
            let role = match msg.role.as_str() {
                "assistant" => Role::Assistant,
                "user" => Role::User,
                _ => Role::Custom(msg.role),
            };
            let blocks = match msg.content {
                AnthropicContent::Text(s) => vec![AnthropicContentBlock::text(s)],
                AnthropicContent::Blocks(b) => b,
            };
            messages.push(Message {
                role,
                content: blocks,
            });
        }

        let stop = req.stop_sequences.map(|mut seqs| {
            if seqs.len() == 1 {
                Stop::Single(seqs.pop().unwrap())
            } else {
                Stop::Multiple(seqs)
            }
        });

        let tools = req.tools.map(|tools| {
            tools
                .into_iter()
                .map(|t| Tool {
                    kind: ToolType::Function,
                    function: FunctionDef {
                        name: t.name,
                        description: t.description,
                        parameters: Some(t.input_schema),
                    },
                    strict: None,
                })
                .collect()
        });
        let tool_choice = req.tool_choice.as_ref().and_then(translate_tool_choice);

        ChatCompletionRequest {
            model: req.model,
            messages,
            temperature: req.temperature,
            top_p: req.top_p,
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
}

impl TryFrom<ChatCompletionResponse> for AnthropicResponse {
    type Error = Error;

    fn try_from(resp: ChatCompletionResponse) -> Result<Self, Self::Error> {
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
            usage: AnthropicUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                cache_read_input_tokens: usage.prompt_cache_hit_tokens,
                cache_creation_input_tokens: usage.prompt_cache_miss_tokens,
            },
        })
    }
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

fn finish_reason_to_stop(reason: &FinishReason) -> String {
    match reason {
        FinishReason::Stop => "end_turn".to_string(),
        FinishReason::Length => "max_tokens".to_string(),
        FinishReason::ToolCalls => "tool_use".to_string(),
        FinishReason::ContentFilter => "content_filter".to_string(),
        FinishReason::Custom(s) => s.clone(),
    }
}
