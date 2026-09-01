//! Tessera gateway: HTTP front door for the inference workers.
//!
//! Responsibilities kept here rather than in the worker: request validation,
//! least-in-flight routing, and load shedding. The worker stays focused on
//! decoding.

mod pool;
mod routes;

/// Generated from proto/inference.proto by build.rs.
pub mod pb {
    // tonic returns Result<_, tonic::Status> everywhere; Status is ~176 bytes,
    // which trips result_large_err. Not ours to fix -- scoped to generated code.
    #![allow(clippy::result_large_err)]

    tonic::include_proto!("inference");
}

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tracing_subscriber::EnvFilter;

use crate::pool::WorkerPool;
use crate::routes::AppState;

struct Config {
    bind: String,
    worker_endpoints: Vec<String>,
    max_in_flight: usize,
    max_tokens_cap: i32,
}

impl Config {
    fn from_env() -> Self {
        let port = std::env::var("TESSERA_GATEWAY_PORT").unwrap_or_else(|_| "8080".into());
        let workers = std::env::var("TESSERA_WORKER_ENDPOINTS")
            .unwrap_or_else(|_| "http://127.0.0.1:50051".into());

        Self {
            bind: format!("0.0.0.0:{port}"),
            worker_endpoints: workers
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            max_in_flight: std::env::var("TESSERA_MAX_IN_FLIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            max_tokens_cap: std::env::var("TESSERA_MAX_TOKENS_CAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("TESSERA_LOG").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env();
    if config.worker_endpoints.is_empty() {
        return Err("TESSERA_WORKER_ENDPOINTS resolved to an empty list".into());
    }

    tracing::info!(
        workers = ?config.worker_endpoints,
        max_in_flight = config.max_in_flight,
        "connecting to workers"
    );

    let pool = Arc::new(WorkerPool::connect(
        &config.worker_endpoints,
        config.max_in_flight,
    )?);

    let state = Arc::new(AppState {
        pool,
        max_tokens_cap: config.max_tokens_cap,
    });

    let app = Router::new()
        .route("/v1/completions", post(routes::completions))
        .route("/health", get(routes::health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&config.bind).await?;
    tracing::info!(bind = %config.bind, "gateway listening");
    axum::serve(listener, app).await?;

    Ok(())
}
