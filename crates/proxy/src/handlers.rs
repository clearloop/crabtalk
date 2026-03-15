use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use crabtalk_core::{ApiError, ChatCompletionRequest, EmbeddingRequest, Model, ModelList};
use futures::StreamExt;

use crate::AppState;

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let provider = match state.registry.get(&request.model) {
        Some(p) => p,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("model '{}' not found", request.model),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    // Branch on stream field.
    if request.stream == Some(true) {
        match provider
            .chat_completion_stream(&state.client, &request)
            .await
        {
            Ok(stream) => {
                let sse_stream = stream.map(|result| match result {
                    Ok(chunk) => {
                        let json = serde_json::to_string(&chunk).unwrap_or_default();
                        Ok(Event::default().data(json))
                    }
                    Err(e) => {
                        let json =
                            serde_json::to_string(&ApiError::new(e.to_string(), "server_error"))
                                .unwrap_or_default();
                        Ok(Event::default().data(json))
                    }
                });

                // Append [DONE] sentinel after the stream ends.
                let done = futures::stream::once(async {
                    Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
                });
                let full_stream = sse_stream.chain(done);

                Sse::new(full_stream)
                    .keep_alive(axum::response::sse::KeepAlive::new())
                    .into_response()
            }
            Err(e) => error_response(e),
        }
    } else {
        match provider.chat_completion(&state.client, &request).await {
            Ok(resp) => Json(resp).into_response(),
            Err(e) => error_response(e),
        }
    }
}

/// POST /v1/embeddings
pub async fn embeddings(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingRequest>,
) -> Response {
    let provider = match state.registry.get(&request.model) {
        Some(p) => p,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiError::new(
                    format!("model '{}' not found", request.model),
                    "invalid_request_error",
                )),
            )
                .into_response();
        }
    };

    match provider.embedding(&state.client, &request).await {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => error_response(e),
    }
}

/// GET /v1/models
pub async fn models(State(state): State<AppState>) -> Json<ModelList> {
    let data: Vec<Model> = state
        .config
        .models
        .keys()
        .map(|name| Model {
            id: name.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "crabtalk".to_string(),
        })
        .collect();

    Json(ModelList {
        object: "list".to_string(),
        data,
    })
}

/// Map a provider Error to an HTTP error response.
fn error_response(e: crabtalk_core::Error) -> Response {
    let (status, api_error) = match &e {
        crabtalk_core::Error::Provider { status, body } => (
            StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY),
            ApiError::new(body.clone(), "upstream_error"),
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::new(e.to_string(), "server_error"),
        ),
    };
    (status, Json(api_error)).into_response()
}
