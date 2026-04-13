use serde::{Deserialize, Serialize};

/// Per-key rate limit (mirrors server-side KeyRateLimit).
#[derive(Debug, Deserialize, Serialize)]
pub struct KeyRateLimit {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requests_per_minute: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_minute: Option<u64>,
}

/// POST /v1/admin/keys request body.
#[derive(Serialize)]
pub struct CreateKeyRequest {
    pub name: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<KeyRateLimit>,
}

/// POST /v1/admin/keys response (full key visible once).
#[derive(Deserialize, Serialize)]
pub struct KeyResponse {
    pub name: String,
    pub key: String,
    pub models: Vec<String>,
    pub rate_limit: Option<KeyRateLimit>,
}

/// Format rate limit fields as (rpm, tpm) display strings.
pub fn format_rate_limit(rl: &Option<KeyRateLimit>) -> (String, String) {
    let rpm = rl
        .as_ref()
        .and_then(|r| r.requests_per_minute)
        .map_or("-".into(), |v| v.to_string());
    let tpm = rl
        .as_ref()
        .and_then(|r| r.tokens_per_minute)
        .map_or("-".into(), |v| v.to_string());
    (rpm, tpm)
}

/// GET /v1/admin/keys and GET /v1/admin/keys/{name} response.
#[derive(Deserialize, Serialize)]
pub struct KeySummary {
    pub name: String,
    pub key_prefix: String,
    pub models: Vec<String>,
    pub rate_limit: Option<KeyRateLimit>,
    pub source: String,
}

/// POST /v1/admin/providers/reload response.
#[derive(Deserialize, Serialize)]
pub struct ReloadResponse {
    pub status: String,
    pub models: usize,
    pub providers: usize,
}

/// GET /v1/admin/usage entry.
#[derive(Deserialize, Serialize)]
pub struct UsageEntry {
    pub name: String,
    pub model: String,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
}

/// GET /v1/budget entry.
#[derive(Deserialize, Serialize)]
pub struct BudgetEntry {
    pub key: String,
    pub spent_usd: f64,
    pub budget_usd: f64,
    pub remaining_usd: f64,
}

/// GET /v1/admin/logs entry.
#[derive(Deserialize, Serialize)]
pub struct AuditRecord {
    pub request_id: String,
    pub timestamp: i64,
    pub key_name: String,
    pub model: String,
    pub provider: String,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub cost_micros: i64,
    pub latency_ms: u64,
    pub status: u16,
}
