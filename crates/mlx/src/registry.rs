//! Predefined model registry, auto-generated from mlx-swift-lm's
//! `LLMModelFactory.swift` at build time.
//!
//! Each entry is a `(alias, hf_repo_id, default_prompt)` tuple. The
//! alias is a lowercase version of the HuggingFace model name (e.g.
//! `"qwen3.5-2b-mlx-4bit"` for `mlx-community/Qwen3.5-2B-MLX-4bit`).
//!
//! Use [`resolve`] to map an alias or full repo id to the canonical
//! HF repo path. Use [`list`] to get all available models for UI
//! display.

include!(concat!(env!("OUT_DIR"), "/model_registry.rs"));

/// A model entry from the registry.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Lowercase alias (e.g. `"qwen3.5-2b-mlx-4bit"`).
    pub alias: &'static str,
    /// Full HuggingFace repo id (e.g. `"mlx-community/Qwen3.5-2B-MLX-4bit"`).
    pub repo_id: &'static str,
    /// Example prompt for the model.
    pub default_prompt: &'static str,
}

/// List all predefined models.
pub fn list() -> Vec<ModelEntry> {
    MODEL_REGISTRY
        .iter()
        .map(|&(alias, repo_id, default_prompt)| ModelEntry {
            alias,
            repo_id,
            default_prompt,
        })
        .collect()
}

/// Resolve a model name to a HuggingFace repo id.
///
/// Accepts:
///   * A full repo id (`"mlx-community/Qwen3.5-2B-MLX-4bit"`) — returned as-is.
///   * A lowercase alias (`"qwen3.5-2b-mlx-4bit"`) — looked up in the registry.
///   * A local directory path — returned as-is.
///
/// Returns `None` if the input is not a path, not a full repo id
/// (no `/`), and not a known alias.
pub fn resolve(model: &str) -> Option<&'static str> {
    // Full repo id
    if model.contains('/') {
        return Some(
            MODEL_REGISTRY
                .iter()
                .find(|(_, id, _)| *id == model)
                .map(|(_, id, _)| *id)
                .unwrap_or_else(|| {
                    // Not in registry but looks like a repo id — pass through.
                    // Leak is fine: this is called once per model load.
                    Box::leak(model.to_string().into_boxed_str())
                }),
        );
    }

    // Alias lookup (case-insensitive)
    let lower = model.to_lowercase();
    MODEL_REGISTRY
        .iter()
        .find(|(alias, _, _)| *alias == lower)
        .map(|(_, id, _)| *id)
}
