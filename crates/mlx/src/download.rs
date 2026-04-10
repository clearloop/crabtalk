//! HuggingFace model downloader backed by the `hf-hub` crate.

use crabllm_core::Error;
use hf_hub::api::Progress;
use hf_hub::api::sync::{Api, ApiBuilder};
use std::path::PathBuf;

/// File name suffixes matched by the wildcard part of the allowlist.
const ALLOWED_SUFFIXES: &[&str] = &[".safetensors", ".jinja"];

/// Exact filenames matched in addition to the suffix list.
const ALLOWED_EXACT: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "tokenizer.model",
    "model.safetensors.index.json",
];

fn build_api() -> Result<Api, Error> {
    ApiBuilder::from_env()
        .build()
        .map_err(|e| Error::Internal(format!("mlx: failed to build HF API client: {e}")))
}

/// Check whether a model is already cached. Returns the snapshot
/// directory if `config.json` is present.
pub fn cached_model_path(repo: &str) -> Option<PathBuf> {
    let api = build_api().ok()?;
    let repo_handle = api.model(repo.to_string());
    repo_handle
        .get("config.json")
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
}

/// Per-file progress event.
#[derive(Debug, Clone)]
pub enum DownloadEvent {
    /// A new file started downloading. `total_bytes` is the file size.
    FileStart {
        filename: String,
        total_bytes: usize,
    },
    /// `bytes` more bytes have been written for the current file.
    FileProgress { bytes: usize },
    /// The current file finished downloading.
    FileDone,
    /// All files finished. `model_dir` is the local path.
    AllDone { model_dir: PathBuf },
}

/// Download all mlx-compatible files from a HuggingFace repo,
/// reporting progress through a `tokio::sync::mpsc` channel.
///
/// Blocking. tokio callers must wrap in `spawn_blocking`.
///
/// Returns the local snapshot directory on success.
pub fn download_model(
    repo: &str,
    tx: &std::sync::mpsc::Sender<DownloadEvent>,
) -> Result<PathBuf, Error> {
    let api = build_api()?;
    let repo_handle = api.model(repo.to_string());

    let info = repo_handle
        .info()
        .map_err(|e| Error::Internal(format!("mlx: model info for {repo}: {e}")))?;

    let wanted: Vec<&str> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|name| is_wanted_filename(name))
        .collect();

    if wanted.is_empty() {
        return Err(Error::Internal(format!(
            "no mlx-compatible files in repo {repo}"
        )));
    }

    let mut model_dir: Option<PathBuf> = None;
    for filename in &wanted {
        let progress = ChannelProgress {
            tx: tx.clone(),
            filename: filename.to_string(),
        };
        let path = repo_handle
            .download_with_progress(filename, progress)
            .map_err(|e| Error::Internal(format!("mlx: download {filename}: {e}")))?;
        if model_dir.is_none() {
            model_dir = path.parent().map(|p| p.to_path_buf());
        }
    }

    let dir = model_dir.ok_or_else(|| Error::Internal("mlx: no files downloaded".to_string()))?;
    let _ = tx.send(DownloadEvent::AllDone {
        model_dir: dir.clone(),
    });
    Ok(dir)
}

/// Adapter from hf-hub's `Progress` trait to our channel-based events.
struct ChannelProgress {
    tx: std::sync::mpsc::Sender<DownloadEvent>,
    filename: String,
}

impl Progress for ChannelProgress {
    fn init(&mut self, size: usize, _filename: &str) {
        let _ = self.tx.send(DownloadEvent::FileStart {
            filename: self.filename.clone(),
            total_bytes: size,
        });
    }

    fn update(&mut self, size: usize) {
        let _ = self.tx.send(DownloadEvent::FileProgress { bytes: size });
    }

    fn finish(&mut self) {
        let _ = self.tx.send(DownloadEvent::FileDone);
    }
}

fn is_wanted_filename(name: &str) -> bool {
    let basename = name.rsplit('/').next().unwrap_or(name);
    if ALLOWED_EXACT.contains(&basename) {
        return true;
    }
    ALLOWED_SUFFIXES.iter().any(|ext| basename.ends_with(ext))
}
