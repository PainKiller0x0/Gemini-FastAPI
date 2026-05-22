mod config;
mod gemini;
mod history;
mod openai;

use std::sync::Arc;

use async_stream::stream;
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{get, post},
};
use chrono::Utc;
use config::Config;
use gemini::GeminiClient;
use history::{HistoryRecord, HistoryStore, started, timestamp};
use openai::{
    AssistantMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, ModelData,
    ModelListResponse, ResponseCreateRequest, StreamChoice, StreamChunk, Usage,
    chat_extra_instructions, estimate_tokens, messages_to_prompt, response_extra_instructions,
};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    config: Config,
    gemini: Arc<GeminiClient>,
    history: HistoryStore,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let config = Config::load()?;
    let first_client = config
        .gemini
        .clients
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("at least one Gemini client is required"))?;
    let append_builtin = config.gemini.model_strategy != "overwrite";
    let gemini = Arc::new(GeminiClient::new(
        first_client,
        config.gemini.models.clone(),
        config.gemini.timeout,
        config.gemini.refresh_interval,
        append_builtin,
    )?);
    let addr = config.server.addr()?;
    if let Err(error) = gemini.refresh_runtime_models().await {
        tracing::warn!(
            ?error,
            "Gemini runtime model discovery failed; continuing with configured models"
        );
    }
    let history = HistoryStore::new(config.storage.path.clone());
    let state = Arc::new(AppState {
        config,
        gemini,
        history,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(create_response))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    tracing::info!("starting gemini-fastapi-rs at http://{}", addr);
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "status": "ok",
        "implementation": "rust",
        "clients": state.config.gemini.clients.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
    }))
}

async fn list_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<ModelListResponse>, ApiError> {
    verify_auth(&state, &headers)?;
    let created = Utc::now().timestamp();
    let data = state
        .gemini
        .models()
        .await
        .into_iter()
        .map(|model| ModelData {
            id: model.model_name,
            object: "model",
            created,
            owned_by: model.owned_by,
        })
        .collect();
    Ok(Json(ModelListResponse {
        object: "list",
        data,
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    verify_auth(&state, &headers)?;
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages are required"));
    }

    let mut prompt = messages_to_prompt(&request.messages);
    let extra = chat_extra_instructions(&request);
    if !extra.is_empty() {
        prompt.push_str("\n\n[system]\n");
        prompt.push_str(&extra);
    }
    let model_name = request.model.clone();
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = Utc::now().timestamp();
    let start = started();
    let output = match state.gemini.generate(&model_name, &prompt).await {
        Ok(output) => {
            state.history.append(&HistoryRecord {
                ts: timestamp(),
                kind: "chat.completions",
                model: &model_name,
                prompt_chars: prompt.chars().count(),
                output_chars: output.chars().count(),
                latency_ms: start.elapsed().as_millis(),
                ok: true,
                error: None,
            });
            output
        }
        Err(error) => {
            let detail = error.to_string();
            state.history.append(&HistoryRecord {
                ts: timestamp(),
                kind: "chat.completions",
                model: &model_name,
                prompt_chars: prompt.chars().count(),
                output_chars: 0,
                latency_ms: start.elapsed().as_millis(),
                ok: false,
                error: Some(&detail),
            });
            return Err(ApiError::from(error));
        }
    };

    if request.stream.unwrap_or(false) {
        let prompt_tokens = estimate_tokens(&prompt);
        let completion_tokens = estimate_tokens(&output);
        let id = completion_id.clone();
        let model = model_name.clone();
        let s = stream! {
            let role = StreamChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: json!({"role": "assistant"}),
                    finish_reason: None,
                }],
            };
            yield Ok::<_, std::convert::Infallible>(Event::default().data(serde_json::to_string(&role).unwrap()));

            let content = StreamChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: json!({"content": output}),
                    finish_reason: None,
                }],
            };
            yield Ok(Event::default().data(serde_json::to_string(&content).unwrap()));

            let done = StreamChunk {
                id,
                object: "chat.completion.chunk",
                created,
                model,
                choices: vec![StreamChoice {
                    index: 0,
                    delta: json!({}),
                    finish_reason: Some("stop"),
                }],
            };
            yield Ok(Event::default().data(serde_json::to_string(&done).unwrap()));
            yield Ok(Event::default().data("[DONE]"));

            let _ = (prompt_tokens, completion_tokens);
        };
        return Ok(Sse::new(s).into_response());
    }

    let prompt_tokens = estimate_tokens(&prompt);
    let completion_tokens = estimate_tokens(&output);
    let payload = ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion",
        created,
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: output,
            },
            finish_reason: "stop",
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };
    Ok(Json(payload).into_response())
}

async fn create_response(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ResponseCreateRequest>,
) -> Result<Response, ApiError> {
    verify_auth(&state, &headers)?;
    let mut prompt = response_input_to_prompt(&request.input, request.instructions.as_deref());
    let extra = response_extra_instructions(&request);
    if !extra.is_empty() {
        prompt.push_str("\n\n[system]\n");
        prompt.push_str(&extra);
    }
    if prompt.trim().is_empty() {
        return Err(ApiError::bad_request("input is required"));
    }
    let response_id = format!("resp_{}", Uuid::new_v4());
    let created = Utc::now().timestamp();
    let start = started();
    let output = match state.gemini.generate(&request.model, &prompt).await {
        Ok(output) => {
            state.history.append(&HistoryRecord {
                ts: timestamp(),
                kind: "responses",
                model: &request.model,
                prompt_chars: prompt.chars().count(),
                output_chars: output.chars().count(),
                latency_ms: start.elapsed().as_millis(),
                ok: true,
                error: None,
            });
            output
        }
        Err(error) => {
            let detail = error.to_string();
            state.history.append(&HistoryRecord {
                ts: timestamp(),
                kind: "responses",
                model: &request.model,
                prompt_chars: prompt.chars().count(),
                output_chars: 0,
                latency_ms: start.elapsed().as_millis(),
                ok: false,
                error: Some(&detail),
            });
            return Err(ApiError::from(error));
        }
    };

    if request.stream.unwrap_or(false) {
        let id = response_id.clone();
        let model = request.model.clone();
        let s = stream! {
            yield Ok::<_, std::convert::Infallible>(Event::default().event("response.created").data(serde_json::to_string(&json!({
                "type": "response.created",
                "response": {"id": id, "object": "response", "created_at": created, "model": model, "status": "in_progress"}
            })).unwrap()));
            yield Ok(Event::default().event("response.output_text.delta").data(serde_json::to_string(&json!({
                "type": "response.output_text.delta",
                "item_id": "msg_0",
                "output_index": 0,
                "content_index": 0,
                "delta": output,
            })).unwrap()));
            yield Ok(Event::default().event("response.completed").data(serde_json::to_string(&json!({
                "type": "response.completed",
                "response": response_payload(&id, created, &model, &output, estimate_tokens(&prompt), estimate_tokens(&output)),
            })).unwrap()));
            yield Ok(Event::default().data("[DONE]"));
        };
        return Ok(Sse::new(s).into_response());
    }

    let prompt_tokens = estimate_tokens(&prompt);
    let completion_tokens = estimate_tokens(&output);
    let payload = response_payload(
        &response_id,
        created,
        &request.model,
        &output,
        prompt_tokens,
        completion_tokens,
    );
    Ok(Json(payload).into_response())
}

fn response_payload(
    id: &str,
    created: i64,
    model: &str,
    output: &str,
    input_tokens: u64,
    output_tokens: u64,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model,
        "output": [{
            "id": "msg_0",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": output, "annotations": []}],
        }],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    })
}

fn response_input_to_prompt(input: &Value, instructions: Option<&str>) -> String {
    let mut parts = Vec::new();
    if let Some(instructions) = instructions.filter(|s| !s.trim().is_empty()) {
        parts.push(format!("[system]\n{}", instructions.trim()));
    }
    match input {
        Value::String(text) => parts.push(format!("[user]\n{}", text.trim())),
        Value::Array(items) => {
            for item in items {
                if let Some(text) = item.get("content").and_then(Value::as_str) {
                    let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                    parts.push(format!("[{role}]\n{}", text.trim()));
                } else if let Some(content) = item.get("content").and_then(Value::as_array) {
                    let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                    let text = content
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !text.trim().is_empty() {
                        parts.push(format!("[{role}]\n{}", text.trim()));
                    }
                }
            }
        }
        Value::Object(_) => {
            if let Some(text) = input.get("content").and_then(Value::as_str) {
                let role = input.get("role").and_then(Value::as_str).unwrap_or("user");
                parts.push(format!("[{role}]\n{}", text.trim()));
            }
        }
        _ => {}
    }
    parts.join("\n\n")
}

fn verify_auth(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(expected) = state
        .config
        .server
        .api_key
        .as_deref()
        .filter(|s| !s.is_empty())
    else {
        return Ok(());
    };
    let Some(value) = headers.get("authorization").and_then(|v| v.to_str().ok()) else {
        return Err(ApiError::unauthorized("missing authorization header"));
    };
    if value == format!("Bearer {expected}") {
        Ok(())
    } else {
        Err(ApiError::unauthorized("invalid api key"))
    }
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    detail: String,
}

impl ApiError {
    fn bad_request(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            detail: detail.into(),
        }
    }

    fn unauthorized(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            detail: detail.into(),
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            detail: error.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "detail": self.detail }))).into_response()
    }
}
