use crate::PricingConfig;
use serde::{Deserialize, Serialize};

/// Per-model metadata: context window and token pricing.
///
/// Every field is `Option` so partial overrides work — a config entry
/// that sets only `context_length` can leave pricing unset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ModelInfo {
    /// Maximum context window in tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// Token pricing (input + output cost per million tokens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<PricingConfig>,
    /// Whether the model accepts image/video input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,
}

impl ModelInfo {
    /// Compute cost in USD for the given token counts. Returns 0.0 if
    /// pricing is not set.
    pub fn cost(&self, prompt_tokens: u32, completion_tokens: u32) -> f64 {
        let Some(ref p) = self.pricing else {
            return 0.0;
        };
        (prompt_tokens as f64 * p.prompt_cost_per_million
            + completion_tokens as f64 * p.completion_cost_per_million)
            / 1_000_000.0
    }
}
