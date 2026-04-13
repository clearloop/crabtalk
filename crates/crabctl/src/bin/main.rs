use clap::{Parser, Subcommand};
use crabctl::{
    client::AdminClient,
    config::Config,
    error::Error,
    output::{print_kv, print_table},
    types::{CreateKeyRequest, KeyRateLimit, format_rate_limit},
};

#[derive(Parser)]
#[command(name = "crabctl", about = "Manage a running crabllm gateway")]
struct Cli {
    /// Gateway URL (e.g. http://localhost:8080)
    #[arg(long)]
    url: Option<String>,

    /// Admin bearer token
    #[arg(long)]
    token: Option<String>,

    /// Output raw JSON instead of formatted tables
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage API keys
    Keys {
        #[command(subcommand)]
        command: KeyCommands,
    },
    /// Manage providers
    Providers {
        #[command(subcommand)]
        command: ProviderCommands,
    },
    /// Query token usage
    Usage {
        /// Filter by key name
        #[arg(long)]
        key: Option<String>,
        /// Filter by model
        #[arg(long)]
        model: Option<String>,
    },
    /// View budget status
    Budget,
    /// Query audit logs
    Logs {
        /// Filter by key name
        #[arg(long)]
        key: Option<String>,
        /// Filter by model
        #[arg(long)]
        model: Option<String>,
        /// Start timestamp (epoch milliseconds)
        #[arg(long)]
        since: Option<i64>,
        /// End timestamp (epoch milliseconds)
        #[arg(long)]
        until: Option<i64>,
        /// Maximum number of records
        #[arg(long, default_value = "100")]
        limit: usize,
    },
    /// Cache operations
    Cache {
        #[command(subcommand)]
        command: CacheCommands,
    },
}

#[derive(Subcommand)]
enum KeyCommands {
    /// List all keys
    List,
    /// Create a new key
    Create {
        /// Key name
        name: String,
        /// Allowed models (comma-separated, default: all)
        #[arg(long, value_delimiter = ',')]
        models: Option<Vec<String>>,
        /// Requests per minute limit
        #[arg(long)]
        rpm: Option<u64>,
        /// Tokens per minute limit
        #[arg(long)]
        tpm: Option<u64>,
    },
    /// Get key details
    Get {
        /// Key name
        name: String,
    },
    /// Update a key (JSON Merge Patch)
    Update {
        /// Key name
        name: String,
        /// Allowed models (comma-separated)
        #[arg(long, value_delimiter = ',')]
        models: Option<Vec<String>>,
        /// Requests per minute limit
        #[arg(long)]
        rpm: Option<u64>,
        /// Tokens per minute limit
        #[arg(long)]
        tpm: Option<u64>,
    },
    /// Delete a key
    Delete {
        /// Key name
        name: String,
    },
}

#[derive(Subcommand)]
enum ProviderCommands {
    /// Reload providers from config file
    Reload,
}

#[derive(Subcommand)]
enum CacheCommands {
    /// Clear all cached responses
    Clear,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let config = match Config::resolve(cli.url, cli.token) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let client = AdminClient::new(config.url, config.token);

    let result = match cli.command {
        Commands::Keys { command } => run_keys(&client, command, cli.json).await,
        Commands::Providers { command } => run_providers(&client, command, cli.json).await,
        Commands::Usage { key, model } => run_usage(&client, key, model, cli.json).await,
        Commands::Budget => run_budget(&client, cli.json).await,
        Commands::Logs {
            key,
            model,
            since,
            until,
            limit,
        } => run_logs(&client, key, model, since, until, limit, cli.json).await,
        Commands::Cache { command } => run_cache(&client, command, cli.json).await,
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run_keys(client: &AdminClient, cmd: KeyCommands, json: bool) -> Result<(), Error> {
    match cmd {
        KeyCommands::List => {
            let keys = client.list_keys().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&keys).unwrap());
                return Ok(());
            }
            let rows: Vec<Vec<String>> = keys
                .iter()
                .map(|k| {
                    let (rpm, tpm) = format_rate_limit(&k.rate_limit);
                    vec![
                        k.name.clone(),
                        k.key_prefix.clone(),
                        k.models.join(", "),
                        rpm,
                        tpm,
                        k.source.clone(),
                    ]
                })
                .collect();
            print_table(&["NAME", "PREFIX", "MODELS", "RPM", "TPM", "SOURCE"], &rows);
        }
        KeyCommands::Create {
            name,
            models,
            rpm,
            tpm,
        } => {
            let rate_limit = rate_limit_from_flags(rpm, tpm);
            let req = CreateKeyRequest {
                name,
                models: models.unwrap_or_default(),
                rate_limit,
            };
            let resp = client.create_key(&req).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                return Ok(());
            }
            let models = resp.models.join(", ");
            let (rpm, tpm) = format_rate_limit(&resp.rate_limit);
            print_kv(&[
                ("Name", &resp.name),
                ("Key", &resp.key),
                ("Models", &models),
                ("RPM", &rpm),
                ("TPM", &tpm),
            ]);
        }
        KeyCommands::Get { name } => {
            let key = client.get_key(&name).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&key).unwrap());
                return Ok(());
            }
            let models = key.models.join(", ");
            let (rpm, tpm) = format_rate_limit(&key.rate_limit);
            print_kv(&[
                ("Name", &key.name),
                ("Prefix", &key.key_prefix),
                ("Models", &models),
                ("RPM", &rpm),
                ("TPM", &tpm),
                ("Source", &key.source),
            ]);
        }
        KeyCommands::Update {
            name,
            models,
            rpm,
            tpm,
        } => {
            let mut patch = serde_json::Map::new();
            if let Some(m) = models {
                patch.insert("models".into(), serde_json::json!(m));
            }
            let rate_limit = rate_limit_from_flags(rpm, tpm);
            if let Some(rl) = rate_limit {
                patch.insert("rate_limit".into(), serde_json::to_value(rl).unwrap());
            }
            if patch.is_empty() {
                eprintln!("error: nothing to update (pass --models, --rpm, or --tpm)");
                std::process::exit(1);
            }
            let key = client
                .update_key(&name, &serde_json::Value::Object(patch))
                .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&key).unwrap());
                return Ok(());
            }
            let models = key.models.join(", ");
            let (rpm, tpm) = format_rate_limit(&key.rate_limit);
            print_kv(&[
                ("Name", &key.name),
                ("Prefix", &key.key_prefix),
                ("Models", &models),
                ("RPM", &rpm),
                ("TPM", &tpm),
                ("Source", &key.source),
            ]);
        }
        KeyCommands::Delete { name } => {
            client.delete_key(&name).await?;
            println!("Key '{name}' deleted.");
        }
    }
    Ok(())
}

fn rate_limit_from_flags(rpm: Option<u64>, tpm: Option<u64>) -> Option<KeyRateLimit> {
    if rpm.is_some() || tpm.is_some() {
        Some(KeyRateLimit {
            requests_per_minute: rpm,
            tokens_per_minute: tpm,
        })
    } else {
        None
    }
}

async fn run_providers(
    client: &AdminClient,
    cmd: ProviderCommands,
    json: bool,
) -> Result<(), Error> {
    match cmd {
        ProviderCommands::Reload => {
            let resp = client.reload_providers().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                return Ok(());
            }
            println!(
                "Providers reloaded: {} models, {} providers.",
                resp.models, resp.providers
            );
        }
    }
    Ok(())
}

async fn run_usage(
    client: &AdminClient,
    key: Option<String>,
    model: Option<String>,
    json: bool,
) -> Result<(), Error> {
    let entries = client.usage(key.as_deref(), model.as_deref()).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&entries).unwrap());
        return Ok(());
    }
    let rows: Vec<Vec<String>> = entries
        .iter()
        .map(|e| {
            vec![
                e.name.clone(),
                e.model.clone(),
                e.prompt_tokens.to_string(),
                e.completion_tokens.to_string(),
            ]
        })
        .collect();
    print_table(
        &["KEY", "MODEL", "PROMPT TOKENS", "COMPLETION TOKENS"],
        &rows,
    );
    Ok(())
}

async fn run_budget(client: &AdminClient, json: bool) -> Result<(), Error> {
    let entries = client.budget().await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&entries).unwrap());
        return Ok(());
    }
    let rows: Vec<Vec<String>> = entries
        .iter()
        .map(|e| {
            vec![
                e.key.clone(),
                format!("{:.6}", e.spent_usd),
                format!("{:.6}", e.budget_usd),
                format!("{:.6}", e.remaining_usd),
            ]
        })
        .collect();
    print_table(
        &["KEY", "SPENT (USD)", "BUDGET (USD)", "REMAINING (USD)"],
        &rows,
    );
    Ok(())
}

async fn run_logs(
    client: &AdminClient,
    key: Option<String>,
    model: Option<String>,
    since: Option<i64>,
    until: Option<i64>,
    limit: usize,
    json: bool,
) -> Result<(), Error> {
    let records = client
        .logs(key.as_deref(), model.as_deref(), since, until, limit)
        .await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&records).unwrap());
        return Ok(());
    }
    let rows: Vec<Vec<String>> = records
        .iter()
        .map(|r| {
            let tokens = match (r.prompt_tokens, r.completion_tokens) {
                (Some(p), Some(c)) => format!("{p}/{c}"),
                _ => "-".into(),
            };
            let cost = format!("${:.6}", r.cost_micros as f64 / 1_000_000.0);
            let latency = format!("{}ms", r.latency_ms);
            vec![
                r.timestamp.to_string(),
                r.request_id.clone(),
                r.key_name.clone(),
                r.model.clone(),
                r.provider.clone(),
                tokens,
                cost,
                latency,
                r.status.to_string(),
            ]
        })
        .collect();
    print_table(
        &[
            "TIMESTAMP",
            "REQUEST ID",
            "KEY",
            "MODEL",
            "PROVIDER",
            "TOKENS",
            "COST",
            "LATENCY",
            "STATUS",
        ],
        &rows,
    );
    Ok(())
}

async fn run_cache(client: &AdminClient, cmd: CacheCommands, _json: bool) -> Result<(), Error> {
    match cmd {
        CacheCommands::Clear => {
            client.clear_cache().await?;
            println!("Cache cleared.");
        }
    }
    Ok(())
}
