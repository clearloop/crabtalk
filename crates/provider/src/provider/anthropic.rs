use crabtalk_core::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice, Delta,
    Error, Message, Usage,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};

const DEFAULT_MAX_TOKENS: u32 = 4096;
const BASE_URL: &str = "https://api.anthropic.com/v1";

// ── Anthropic-native request types ──

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

// ── Anthropic-native response types ──

#[derive(Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ── Anthropic SSE event types ──

#[derive(Deserialize)]
struct SseEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    delta: Option<SseDelta>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct SseDelta {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    stop_reason: Option<String>,
}

// ── Translation ──

fn translate_request(request: &ChatCompletionRequest) -> AnthropicRequest {
    let mut system_parts = Vec::new();
    let mut messages = Vec::new();

    for msg in &request.messages {
        if msg.role == "system" {
            if let Some(content) = &msg.content
                && let Some(s) = content.as_str()
            {
                system_parts.push(s.to_string());
            }
        } else {
            let content = msg
                .content
                .as_ref()
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            messages.push(AnthropicMessage {
                role: msg.role.clone(),
                content,
            });
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    };

    AnthropicRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        system,
        temperature: request.temperature,
        top_p: request.top_p,
        stream: request.stream,
    }
}

fn map_stop_reason(stop_reason: &Option<String>) -> Option<String> {
    stop_reason.as_ref().map(|r| match r.as_str() {
        "end_turn" => "stop".to_string(),
        "max_tokens" => "length".to_string(),
        other => other.to_string(),
    })
}

fn translate_response(resp: AnthropicResponse) -> ChatCompletionResponse {
    let content_text: String = resp
        .content
        .iter()
        .filter(|b| b.kind == "text")
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join("");

    ChatCompletionResponse {
        id: resp.id,
        object: "chat.completion".to_string(),
        created: 0,
        model: resp.model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: Some(serde_json::Value::String(content_text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            finish_reason: map_stop_reason(&resp.stop_reason),
        }],
        usage: Some(Usage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        }),
    }
}

pub fn not_implemented(name: &str) -> Error {
    Error::Internal(format!("anthropic {name} not supported"))
}

// ── Public API ──

pub async fn chat_completion(
    client: &reqwest::Client,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Error> {
    let anthropic_req = translate_request(request);
    let url = format!("{BASE_URL}/messages");

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&anthropic_req)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    let anthropic_resp: AnthropicResponse = resp
        .json()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    Ok(translate_response(anthropic_resp))
}

pub async fn chat_completion_stream(
    client: &reqwest::Client,
    api_key: &str,
    request: &ChatCompletionRequest,
    model: &str,
) -> Result<impl Stream<Item = Result<ChatCompletionChunk, Error>> + use<>, Error> {
    let mut anthropic_req = translate_request(request);
    anthropic_req.stream = Some(true);
    let url = format!("{BASE_URL}/messages");

    let resp = client
        .post(&url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&anthropic_req)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    let model = model.to_string();
    Ok(anthropic_sse_stream(resp, model))
}

fn anthropic_sse_stream(
    resp: reqwest::Response,
    model: String,
) -> impl Stream<Item = Result<ChatCompletionChunk, Error>> {
    let byte_stream = resp.bytes_stream();

    stream::unfold(
        (byte_stream, String::new(), model, 0u64),
        |(mut byte_stream, mut buffer, model, mut chunk_idx)| async move {
            use futures::TryStreamExt;

            loop {
                if let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                    buffer = buffer[newline_pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    let Some(data) = line.strip_prefix("data: ") else {
                        // Skip event: lines and other non-data lines.
                        continue;
                    };
                    let data = data.trim();

                    let event: SseEvent = match serde_json::from_str(data) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };

                    match event.kind.as_str() {
                        "content_block_delta" => {
                            if let Some(delta) = &event.delta
                                && delta.kind == "text_delta"
                            {
                                chunk_idx += 1;
                                let chunk = ChatCompletionChunk {
                                    id: format!("chatcmpl-{chunk_idx}"),
                                    object: "chat.completion.chunk".to_string(),
                                    created: 0,
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: if chunk_idx == 1 {
                                                Some("assistant".to_string())
                                            } else {
                                                None
                                            },
                                            content: Some(delta.text.clone()),
                                            tool_calls: None,
                                        },
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };
                                return Some((Ok(chunk), (byte_stream, buffer, model, chunk_idx)));
                            }
                        }
                        "message_delta" => {
                            if let Some(delta) = &event.delta {
                                let finish_reason = map_stop_reason(&delta.stop_reason);
                                chunk_idx += 1;
                                let chunk = ChatCompletionChunk {
                                    id: format!("chatcmpl-{chunk_idx}"),
                                    object: "chat.completion.chunk".to_string(),
                                    created: 0,
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: None,
                                            tool_calls: None,
                                        },
                                        finish_reason,
                                    }],
                                    usage: event.usage.map(|u| Usage {
                                        prompt_tokens: u.input_tokens,
                                        completion_tokens: u.output_tokens,
                                        total_tokens: u.input_tokens + u.output_tokens,
                                    }),
                                };
                                return Some((Ok(chunk), (byte_stream, buffer, model, chunk_idx)));
                            }
                        }
                        "message_stop" => return None,
                        // Ignore: message_start, content_block_start, content_block_stop, ping
                        _ => continue,
                    }
                }

                match byte_stream.try_next().await {
                    Ok(Some(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                    }
                    Ok(None) => return None,
                    Err(e) => {
                        return Some((
                            Err(Error::Internal(format!("stream error: {e}"))),
                            (byte_stream, buffer, model, chunk_idx),
                        ));
                    }
                }
            }
        },
    )
}
