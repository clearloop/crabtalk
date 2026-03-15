use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use crabtalk_core::ApiError;

use crate::AppState;

/// Auth middleware: validates Bearer token against configured virtual keys.
/// If no keys are configured, all requests pass through.
pub async fn auth(State(state): State<AppState>, request: Request, next: Next) -> Response {
    // If no keys configured, skip auth entirely.
    if state.config.keys.is_empty() {
        return next.run(request).await;
    }

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    let token = match auth_header.and_then(|h| h.strip_prefix("Bearer ")) {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ApiError::new(
                    "missing or invalid Authorization header",
                    "authentication_error",
                )),
            )
                .into_response();
        }
    };

    let key_config = match state.config.keys.iter().find(|k| k.key == token) {
        Some(k) => k,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ApiError::new("invalid API key", "authentication_error")),
            )
                .into_response();
        }
    };

    // Check model access. We can't parse the body here without consuming it,
    // so model-level access control is deferred to the handler layer.
    // For now, key existence is sufficient.
    let _ = key_config;

    next.run(request).await
}
