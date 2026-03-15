use crabtalk_core::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice, Delta,
    Error, Message, Usage,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MAX_TOKENS: u32 = 4096;

// ── Gemini-native request types ──

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Serialize, Deserialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

// ── Gemini-native response types ──

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiContent>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsage {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
    #[serde(default)]
    total_token_count: u32,
}

// ── Translation ──

fn translate_request(request: &ChatCompletionRequest) -> GeminiRequest {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();

    for msg in &request.messages {
        if msg.role == "system" {
            if let Some(content) = &msg.content
                && let Some(s) = content.as_str()
            {
                system_parts.push(GeminiPart {
                    text: Some(s.to_string()),
                });
            }
        } else {
            let role = match msg.role.as_str() {
                "assistant" => "model",
                other => other,
            };
            let text = msg
                .content
                .as_ref()
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            contents.push(GeminiContent {
                role: Some(role.to_string()),
                parts: vec![GeminiPart { text: Some(text) }],
            });
        }
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(GeminiContent {
            role: None,
            parts: system_parts,
        })
    };

    let stop_sequences = request.stop.as_ref().map(|s| match s {
        crabtalk_core::Stop::Single(s) => vec![s.clone()],
        crabtalk_core::Stop::Multiple(v) => v.clone(),
    });

    let generation_config = Some(GenerationConfig {
        max_output_tokens: Some(request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
        temperature: request.temperature,
        top_p: request.top_p,
        stop_sequences,
    });

    GeminiRequest {
        contents,
        system_instruction,
        generation_config,
    }
}

fn map_finish_reason(reason: &Option<String>) -> Option<String> {
    reason.as_ref().map(|r| match r.as_str() {
        "STOP" => "stop".to_string(),
        "MAX_TOKENS" => "length".to_string(),
        "SAFETY" => "content_filter".to_string(),
        other => other.to_lowercase(),
    })
}

fn translate_response(resp: GeminiResponse, model: &str) -> ChatCompletionResponse {
    let (content_text, finish_reason) = resp
        .candidates
        .first()
        .map(|c| {
            let text = c
                .content
                .as_ref()
                .map(|content| {
                    content
                        .parts
                        .iter()
                        .filter_map(|p| p.text.as_deref())
                        .collect::<Vec<_>>()
                        .join("")
                })
                .unwrap_or_default();
            (text, map_finish_reason(&c.finish_reason))
        })
        .unwrap_or_default();

    ChatCompletionResponse {
        id: String::new(),
        object: "chat.completion".to_string(),
        created: 0,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: Some(serde_json::Value::String(content_text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            finish_reason,
        }],
        usage: resp.usage_metadata.map(|u| Usage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: u.total_token_count,
        }),
    }
}

pub fn not_implemented(name: &str) -> Error {
    Error::Internal(format!("google {name} not supported"))
}

// ── Public API ──

pub async fn chat_completion(
    client: &reqwest::Client,
    api_key: &str,
    request: &ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Error> {
    let gemini_req = translate_request(request);
    let url = format!("{}/models/{}:generateContent", BASE_URL, request.model);

    let resp = client
        .post(&url)
        .header("x-goog-api-key", api_key)
        .header("content-type", "application/json")
        .json(&gemini_req)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    let gemini_resp: GeminiResponse = resp
        .json()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    Ok(translate_response(gemini_resp, &request.model))
}

pub async fn chat_completion_stream(
    client: &reqwest::Client,
    api_key: &str,
    request: &ChatCompletionRequest,
    model: &str,
) -> Result<impl Stream<Item = Result<ChatCompletionChunk, Error>> + use<>, Error> {
    let gemini_req = translate_request(request);
    let url = format!(
        "{}/models/{}:streamGenerateContent?alt=sse",
        BASE_URL, request.model
    );

    let resp = client
        .post(&url)
        .header("x-goog-api-key", api_key)
        .header("content-type", "application/json")
        .json(&gemini_req)
        .send()
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    let status = resp.status().as_u16();
    if status >= 400 {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider { status, body });
    }

    let model = model.to_string();
    Ok(gemini_sse_stream(resp, model))
}

fn gemini_sse_stream(
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
                        continue;
                    };
                    let data = data.trim();

                    let gemini_resp: GeminiResponse = match serde_json::from_str(data) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };

                    // Extract text delta from first candidate.
                    let text = gemini_resp
                        .candidates
                        .first()
                        .and_then(|c| c.content.as_ref())
                        .and_then(|content| content.parts.first())
                        .and_then(|p| p.text.clone());

                    let finish_reason = gemini_resp
                        .candidates
                        .first()
                        .and_then(|c| map_finish_reason(&c.finish_reason));

                    // Skip chunks with no content and no finish reason.
                    if text.is_none() && finish_reason.is_none() {
                        continue;
                    }

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
                                content: text,
                                tool_calls: None,
                            },
                            finish_reason,
                        }],
                        usage: gemini_resp.usage_metadata.map(|u| Usage {
                            prompt_tokens: u.prompt_token_count,
                            completion_tokens: u.candidates_token_count,
                            total_tokens: u.total_token_count,
                        }),
                    };
                    return Some((Ok(chunk), (byte_stream, buffer, model, chunk_idx)));
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
