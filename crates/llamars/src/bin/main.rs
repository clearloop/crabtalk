use clap::{Parser, Subcommand};
use llamars::registry;

#[derive(Parser)]
#[command(
    name = "llamars",
    about = "Managed llama.cpp server with Ollama registry"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start a llama-server for a model
    Serve {
        /// Model name (e.g. llama3.2:3b) or path to a GGUF file
        model: String,

        /// Port to listen on (default: 8080)
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
    /// Pull a model from the Ollama registry
    Pull {
        /// Model name (e.g. llama3.2:3b)
        model: String,
    },
    /// List available tags for a model
    Tags {
        /// Model name (e.g. llama3.2)
        model: String,
    },
    /// Download the llama-server binary for this platform
    Download {
        /// Release tag (e.g. b4567). Defaults to latest.
        #[arg(short, long)]
        tag: Option<String>,
    },
    /// Check that llama-server is installed and reachable
    Check,
    /// Show the resolved path to the llama-server binary
    Which,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { model, port } => serve(&model, port),
        Commands::Pull { model } => pull(&model),
        Commands::Tags { model } => tags(&model),
        Commands::Download { tag } => download(tag.as_deref()),
        Commands::Check => check(),
        Commands::Which => which(),
    }
}

fn serve(model: &str, port: u16) {
    let bin = match llamars::find_server_binary() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    // Resolve model: Ollama name → cached GGUF, or direct file path.
    let model_path = if std::path::Path::new(model).exists() {
        std::path::PathBuf::from(model)
    } else {
        // Ensure model is pulled.
        let cache_dir = match registry::default_cache_dir() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
        match registry::cached_model_path(model, &cache_dir) {
            Some(p) => p,
            None => {
                eprintln!("model not cached, pulling...");
                match registry::pull_model(model, &cache_dir, &|_, _| {}) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("error: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }
    };

    let config = llamars::LlamaCppConfig {
        model_path,
        n_gpu_layers: 999,
        n_ctx: 4096,
        n_threads: None,
    };

    let (name, tag) = registry::parse_model_name(model);
    eprintln!("starting llama-server for {name}:{tag} on port {port}...");

    // Spawn llama-server with the requested port.
    let mut cmd = std::process::Command::new(&bin);
    cmd.arg("--model")
        .arg(&config.model_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg(config.n_ctx.to_string())
        .arg("--n-gpu-layers")
        .arg(config.n_gpu_layers.to_string());

    if let Some(threads) = config.n_threads {
        cmd.arg("--threads").arg(threads.to_string());
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to start llama-server: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("llama-server running on http://127.0.0.1:{port}/v1");

    // Wait for the process — Ctrl+C will kill it.
    match child.wait() {
        Ok(status) => {
            if !status.success() {
                std::process::exit(status.code().unwrap_or(1));
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn pull(model: &str) {
    let cache_dir = match registry::default_cache_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let (name, tag) = registry::parse_model_name(model);
    eprintln!("pulling {name}:{tag}...");

    let last_pct = std::cell::Cell::new(0u8);
    match registry::pull_model(model, &cache_dir, &|downloaded, total| {
        if total == 0 {
            return;
        }
        let pct = (downloaded * 100 / total) as u8;
        if pct != last_pct.get() {
            last_pct.set(pct);
            let downloaded_mb = downloaded / (1024 * 1024);
            let total_mb = total / (1024 * 1024);
            eprint!("\r  {downloaded_mb} MB / {total_mb} MB ({pct}%)    ");
        }
    }) {
        Ok(path) => {
            eprintln!("\r  done: {}", path.display());
        }
        Err(e) => {
            eprintln!("\nerror: {e}");
            std::process::exit(1);
        }
    }
}

fn tags(model: &str) {
    let (name, _) = registry::parse_model_name(model);
    match registry::fetch_tags(name) {
        Ok(tags) => {
            for tag in &tags {
                println!("{name}:{tag}");
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn download(tag: Option<&str>) {
    match llamars::download(tag) {
        Ok(path) => {
            eprintln!("llama-server ready at {}", path.display());
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn check() {
    match llamars::find_server_binary() {
        Ok(path) => {
            eprintln!("llama-server found: {}", path.display());
            let output = std::process::Command::new(&path).arg("--version").output();
            match output {
                Ok(out) => {
                    let version = String::from_utf8_lossy(&out.stdout);
                    let version = version.trim();
                    if !version.is_empty() {
                        eprintln!("{version}");
                    } else {
                        let version = String::from_utf8_lossy(&out.stderr);
                        let version = version.trim();
                        if !version.is_empty() {
                            eprintln!("{version}");
                        }
                    }
                }
                Err(_) => eprintln!("(could not determine version)"),
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn which() {
    match llamars::find_server_binary() {
        Ok(path) => println!("{}", path.display()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
