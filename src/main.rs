mod config;
mod gemini;
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
use openai::{
    AssistantMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, ModelData,
    ModelListResponse, StreamChoice, StreamChunk, Usage, estimate_tokens, messages_to_prompt,
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
        append_builtin,
    )?);
    let addr = config.server.addr()?;
    let state = Arc::new(AppState { config, gemini });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
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

    let prompt = messages_to_prompt(&request.messages);
    let model_name = request.model.clone();
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = Utc::now().timestamp();
    let output = state.gemini.generate(&model_name, &prompt).await?;

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
