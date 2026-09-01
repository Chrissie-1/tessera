//! HTTP surface: OpenAI-shaped completions plus health.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::pb::{GenerateRequest, HealthRequest};
use crate::pool::WorkerPool;

pub struct AppState {
    pub pool: Arc<WorkerPool>,
    pub max_tokens_cap: i32,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i32,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> i32 {
    16
}

fn default_top_p() -> f32 {
    1.0
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// Error type that renders as a JSON body rather than a bare status line.
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({ "error": { "message": self.message } })),
        )
            .into_response()
    }
}

/// Map a worker-side gRPC failure onto the closest HTTP status.
fn map_status(status: tonic::Status) -> ApiError {
    let http = match status.code() {
        tonic::Code::InvalidArgument => StatusCode::BAD_REQUEST,
        tonic::Code::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        tonic::Code::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
        tonic::Code::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::BAD_GATEWAY,
    };
    ApiError::new(http, status.message().to_string())
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ApiError> {
    if request.stream {
        return Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "streaming arrives with continuous batching (Phase 2)",
        ));
    }
    if request.max_tokens <= 0 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "max_tokens must be positive",
        ));
    }
    if request.max_tokens > state.max_tokens_cap {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("max_tokens exceeds cap {}", state.max_tokens_cap),
        ));
    }

    // Shedding load here, before the request reaches a worker, keeps queueing
    // out of the worker's decode loop where it would inflate tail latency.
    let mut lease = state.pool.acquire().ok_or_else(|| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "all workers are at capacity",
        )
    })?;

    let request_id = uuid::Uuid::new_v4().to_string();
    tracing::info!(request_id = %request_id, worker = %lease.endpoint, "dispatching");

    let response = lease
        .client
        .generate(GenerateRequest {
            prompt: request.prompt,
            max_tokens: request.max_tokens,
            stream: false,
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
            request_id: request_id.clone(),
        })
        .await
        .map_err(map_status)?
        .into_inner();

    Ok(Json(CompletionResponse {
        id: request_id,
        object: "text_completion",
        choices: vec![Choice {
            index: 0,
            text: response.text,
            finish_reason: response.finish_reason,
        }],
        usage: Usage {
            prompt_tokens: response.prompt_tokens,
            completion_tokens: response.completion_tokens,
            total_tokens: response.prompt_tokens + response.completion_tokens,
        },
    }))
}

/// Gateway liveness plus a per-worker readiness probe.
pub async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let endpoints = state.pool.endpoints();
    let in_flight = state.pool.in_flight();

    let mut workers = Vec::new();
    for (i, endpoint) in endpoints.iter().enumerate() {
        let mut entry = json!({
            "endpoint": endpoint,
            "in_flight": in_flight[i],
        });

        match state.pool.client_at(i) {
            Some(mut client) => match client.health(HealthRequest {}).await {
                Ok(response) => {
                    let health = response.into_inner();
                    entry["ready"] = json!(health.ready);
                    entry["model"] = json!(health.model);
                    entry["device"] = json!(health.device);
                    entry["worker_in_flight"] = json!(health.in_flight);
                }
                Err(status) => {
                    entry["ready"] = json!(false);
                    entry["error"] = json!(status.message());
                }
            },
            None => {
                entry["ready"] = json!(false);
                entry["error"] = json!("no such worker");
            }
        }
        workers.push(entry);
    }

    Json(json!({ "status": "ok", "workers": workers }))
}
