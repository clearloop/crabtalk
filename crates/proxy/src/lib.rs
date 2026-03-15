use axum::{
    Router, middleware,
    routing::{get, post},
};

pub use state::AppState;

mod auth;
mod handlers;
mod state;

/// Build the Axum router with all API routes.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/models", get(handlers::models))
        .layer(middleware::from_fn_with_state(state.clone(), auth::auth))
        .with_state(state)
}
