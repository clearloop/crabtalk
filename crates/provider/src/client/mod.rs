#[cfg(all(feature = "hyper", feature = "reqwest"))]
compile_error!("crabllm-provider: features `hyper` and `reqwest` are mutually exclusive");

#[cfg(not(any(feature = "hyper", feature = "reqwest")))]
compile_error!("crabllm-provider: enable exactly one of `hyper` or `reqwest`");

#[cfg(all(feature = "native-tls", feature = "rustls"))]
compile_error!("crabllm-provider: features `native-tls` and `rustls` are mutually exclusive");

#[cfg(not(any(feature = "native-tls", feature = "rustls")))]
compile_error!("crabllm-provider: enable exactly one of `native-tls` or `rustls`");

use bytes::Bytes;
use futures::stream::Stream;
use std::pin::Pin;

/// Raw HTTP response — status + body bytes + optional content-type.
pub struct RawResponse {
    pub status: u16,
    pub body: Bytes,
    pub content_type: Option<String>,
}

/// A boxed byte stream suitable for SSE parsing.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>;

#[cfg(feature = "hyper")]
mod hyper;
#[cfg(feature = "hyper")]
pub use hyper::HttpClient;

#[cfg(feature = "reqwest")]
mod reqwest;
#[cfg(feature = "reqwest")]
pub use reqwest::HttpClient;
