//! HTTP surface: OpenAI-shaped completions plus health.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::pb::{GenerateRequest, HealthRequest};
use crate::pool::{Lease, WorkerPool};

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

/// Reject a request before it costs a worker slot.
///
/// Split out so the rules can be tested without a live worker behind them.
pub fn validate(request: &CompletionRequest, max_tokens_cap: i32) -> Result<(), ApiError> {
    if request.max_tokens <= 0 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "max_tokens must be positive",
        ));
    }
    if request.max_tokens > max_tokens_cap {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("max_tokens exceeds cap {max_tokens_cap}"),
        ));
    }
    Ok(())
}

fn upstream(request: &CompletionRequest, request_id: &str, stream: bool) -> GenerateRequest {
    GenerateRequest {
        prompt: request.prompt.clone(),
        max_tokens: request.max_tokens,
        stream,
        temperature: request.temperature,
        top_p: request.top_p,
        seed: request.seed,
        request_id: request_id.to_string(),
    }
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    validate(&request, state.max_tokens_cap)?;

    // Shedding load here, before the request reaches a worker, keeps queueing
    // out of the worker's decode loop where it would inflate tail latency.
    let lease = state.pool.acquire().ok_or_else(|| {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "all workers are at capacity",
        )
    })?;

    let request_id = uuid::Uuid::new_v4().to_string();
    tracing::info!(
        request_id = %request_id,
        worker = %lease.endpoint,
        stream = request.stream,
        "dispatching"
    );

    if request.stream {
        return stream_completions(request, lease, request_id).await;
    }
    unary_completions(request, lease, request_id).await
}

async fn unary_completions(
    request: CompletionRequest,
    mut lease: Lease,
    request_id: String,
) -> Result<Response, ApiError> {
    let response = lease
        .client
        .generate(upstream(&request, &request_id, false))
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
    })
    .into_response())
}

/// Forward the worker's token stream to the client as server-sent events.
///
/// The lease moves into the forwarding task rather than being dropped when
/// this function returns: the request still occupies the worker for as long
/// as it streams, and releasing capacity at the response headers would let
/// the gateway oversubscribe every worker it owns.
async fn stream_completions(
    request: CompletionRequest,
    mut lease: Lease,
    request_id: String,
) -> Result<Response, ApiError> {
    let mut stream = lease
        .client
        .generate_stream(upstream(&request, &request_id, true))
        .await
        .map_err(map_status)?
        .into_inner();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

    tokio::spawn(async move {
        let _lease = lease;
        loop {
            match stream.message().await {
                Ok(Some(message)) => {
                    let finish_reason = if message.finished {
                        json!(message.finish_reason)
                    } else {
                        json!(null)
                    };
                    let chunk = json!({
                        "id": request_id,
                        "object": "text_completion.chunk",
                        "choices": [{
                            "index": 0,
                            "text": message.text,
                            "finish_reason": finish_reason,
                        }],
                    });
                    if tx
                        .send(Ok(Event::default().data(chunk.to_string())))
                        .await
                        .is_err()
                    {
                        // The client hung up; stop pulling tokens for nobody.
                        break;
                    }
                }
                Ok(None) => {
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                    break;
                }
                Err(status) => {
                    tracing::warn!(request_id = %request_id, error = %status, "stream failed");
                    let error = json!({ "error": { "message": status.message() } });
                    let _ = tx.send(Ok(Event::default().data(error.to_string()))).await;
                    break;
                }
            }
        }
    });

    Ok(Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx)).into_response())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn request(max_tokens: i32) -> CompletionRequest {
        CompletionRequest {
            prompt: "hello".to_string(),
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: 0,
            stream: false,
        }
    }

    #[test]
    fn accepts_a_request_within_the_cap() {
        assert!(validate(&request(16), 64).is_ok());
    }

    #[test]
    fn accepts_a_request_exactly_at_the_cap() {
        assert!(validate(&request(64), 64).is_ok());
    }

    #[test]
    fn rejects_non_positive_max_tokens() {
        for max_tokens in [0, -1] {
            let error = validate(&request(max_tokens), 64).expect_err("should reject");
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
        }
    }

    #[test]
    fn rejects_max_tokens_above_the_cap() {
        let error = validate(&request(65), 64).expect_err("should reject");

        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        // The cap is named in the message so a client can correct itself.
        assert!(error.message.contains("64"));
    }

    #[test]
    fn upstream_marks_the_streaming_flag() {
        assert!(upstream(&request(4), "req-1", true).stream);
        assert!(!upstream(&request(4), "req-1", false).stream);
    }

    #[test]
    fn upstream_forwards_sampling_and_identity() {
        let mut source = request(8);
        source.temperature = 0.7;
        source.top_p = 0.9;
        source.seed = 42;

        let forwarded = upstream(&source, "req-2", false);

        assert_eq!(forwarded.prompt, "hello");
        assert_eq!(forwarded.max_tokens, 8);
        assert_eq!(forwarded.temperature, 0.7);
        assert_eq!(forwarded.top_p, 0.9);
        assert_eq!(forwarded.seed, 42);
        assert_eq!(forwarded.request_id, "req-2");
    }

    #[test]
    fn worker_status_maps_onto_http() {
        let cases = [
            (tonic::Code::InvalidArgument, StatusCode::BAD_REQUEST),
            (tonic::Code::Unavailable, StatusCode::SERVICE_UNAVAILABLE),
            (tonic::Code::DeadlineExceeded, StatusCode::GATEWAY_TIMEOUT),
            (
                tonic::Code::ResourceExhausted,
                StatusCode::TOO_MANY_REQUESTS,
            ),
            (tonic::Code::Internal, StatusCode::BAD_GATEWAY),
        ];

        for (code, expected) in cases {
            let error = map_status(tonic::Status::new(code, "boom"));
            assert_eq!(error.status, expected, "for {code:?}");
        }
    }
}
