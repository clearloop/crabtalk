use clap::Parser;
use crabtalk_core::GatewayConfig;
use crabtalk_provider::ProviderRegistry;
use crabtalk_proxy::AppState;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "crabtalk", about = "High-performance LLM API gateway")]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "crabtalk.toml")]
    config: PathBuf,

    /// Override listen address (e.g. 0.0.0.0:8080)
    #[arg(short, long)]
    bind: Option<String>,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let mut config = match GatewayConfig::from_file(&cli.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to load config: {e}");
            std::process::exit(1);
        }
    };

    if let Some(bind) = cli.bind {
        config.listen = bind;
    }

    let registry = match ProviderRegistry::from_config(&config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: failed to build provider registry: {e}");
            std::process::exit(1);
        }
    };

    let addr = config.listen.clone();
    let model_count = config.models.len();
    let provider_count = config.providers.len();

    let state = AppState {
        registry,
        client: reqwest::Client::new(),
        config,
    };

    let app = crabtalk_proxy::router(state);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: failed to bind to {addr}: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("crabtalk listening on {addr} ({model_count} models, {provider_count} providers)");

    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("error: server failed: {e}");
        std::process::exit(1);
    }
}
