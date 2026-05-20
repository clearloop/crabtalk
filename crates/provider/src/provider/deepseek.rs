use crate::{ByteStream, HttpClient};
use bytes::Bytes;
use crabllm_core::Error;

pub const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";

pub fn not_implemented(name: &str) -> Error {
    Error::Internal(format!("deepseek {name} not supported"))
}

/// Forward raw Anthropic-format JSON bytes to DeepSeek's Anthropic-
/// compatible endpoint. Uses `Authorization: Bearer` (not `x-api-key`)
/// because DeepSeek unifies auth across both endpoints.
pub async fn anthropic_messages_raw(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    raw_body: Bytes,
) -> Result<Bytes, Error> {
    let url = format!("{}/messages", base_url.trim_end_matches('/'));
    let bearer = format!("Bearer {api_key}");
    let headers = [
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
        ("authorization", bearer.as_str()),
    ];
    let resp = client
        .post(&url, &headers, raw_body)
        .await
        .map_err(|e| Error::Internal(e.to_string()))?;

    if resp.status >= 400 {
        let body = String::from_utf8_lossy(&resp.body).into_owned();
        return Err(Error::Provider {
            status: resp.status,
            body,
        });
    }

    Ok(resp.body)
}

/// Stream raw Anthropic SSE bytes from DeepSeek's Anthropic-compatible endpoint.
pub async fn anthropic_messages_stream(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    raw_body: Bytes,
) -> Result<ByteStream, Error> {
    let url = format!("{}/messages", base_url.trim_end_matches('/'));
    let bearer = format!("Bearer {api_key}");
    let headers = [
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
        ("authorization", bearer.as_str()),
    ];
    client.post_stream(&url, &headers, raw_body).await
}
