use crabtalk_core::GatewayConfig;
use crabtalk_provider::ProviderRegistry;

/// Shared application state passed to all handlers.
#[derive(Clone)]
pub struct AppState {
    pub registry: ProviderRegistry,
    pub client: reqwest::Client,
    pub config: GatewayConfig,
}
