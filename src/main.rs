mod config;
mod gemini;
mod history;
mod images;
mod openai;

use std::{collections::HashMap, path::Path as FsPath, sync::Arc};

use async_stream::stream;
use axum::{
    Json, Router,
    extract::{Path as AxPath, Query, State},
    http::{HeaderMap, StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{get, post},
};
use chrono::Utc;
use config::Config;
use gemini::{GeminiImage, GeminiPool};
use history::{HistoryRecord, HistoryStore, started, timestamp};
use images::{ImageData, ImageGenerationRequest, ImageGenerationResponse};
use openai::{
    AssistantMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, ModelData,
    ModelListResponse, ResponseCreateRequest, StreamChoice, StreamChunk, Usage,
    chat_extra_instructions, estimate_tokens, messages_to_gemini_input, process_output,
    response_extra_instructions, response_input_to_gemini_input,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    config: Config,
    gemini: Arc<GeminiPool>,
    history: HistoryStore,
    http: reqwest::Client,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let config = Config::load()?;
    let append_builtin = config.gemini.model_strategy != "overwrite";
    let gemini = Arc::new(GeminiPool::new(
        config.gemini.clients.clone(),
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
        http: reqwest::Client::new(),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(create_response))
        .route("/v1/images/generations", post(image_generations))
        .route("/images/{filename}", get(get_image))
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
        "clients": state.gemini.client_ids(),
    }))
}

async fn list_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<ModelListResponse>, ApiError> {
    verify_auth(&state, &headers)?;
    let created = Utc::now().timestamp();
    let mut data: Vec<ModelData> = state
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
    if state.config.image_generation.is_enabled() {
        for id in image_model_ids(&state.config.image_generation.model) {
            if !data.iter().any(|model| model.id == id) {
                data.push(ModelData {
                    id,
                    object: "model",
                    created,
                    owned_by: "image-generation".to_string(),
                });
            }
        }
    }
    Ok(Json(ModelListResponse {
        object: "list",
        data,
    }))
}

fn image_model_ids(configured: &str) -> Vec<String> {
    let mut ids = vec![
        configured.to_string(),
        "gemini-3.1-flash-image-preview".to_string(),
        "gemini-3-pro-image-preview".to_string(),
        "gemini-2.5-flash-image".to_string(),
        "imagen-4.0-generate-001".to_string(),
        "imagen-4.0-fast-generate-001".to_string(),
        "imagen-4.0-ultra-generate-001".to_string(),
    ];
    ids.retain(|id| !id.trim().is_empty());
    ids.sort();
    ids.dedup();
    ids
}

async fn get_image(
    State(state): State<Arc<AppState>>,
    AxPath(filename): AxPath<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    if filename.contains('/') || filename.contains('\\') || filename.contains("..") {
        return Err(ApiError::bad_request("invalid image filename"));
    }
    if let Some(expected) = image_token(&filename, state.config.server.api_key.as_deref()) {
        let provided = params.get("token").map(String::as_str).unwrap_or("");
        if provided != expected {
            return Err(ApiError::unauthorized("invalid image token"));
        }
    }
    let path = FsPath::new(&state.config.storage.images_path).join(&filename);
    let data = tokio::fs::read(&path)
        .await
        .map_err(|_| ApiError::not_found("image not found"))?;
    let content_type = match path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => "image/png",
    };
    Ok(([(CONTENT_TYPE, content_type)], data).into_response())
}

async fn append_image_markdown(
    state: &AppState,
    text: String,
    images: &[GeminiImage],
) -> Result<String, ApiError> {
    if images.is_empty() {
        return Ok(text);
    }
    let saved = state
        .gemini
        .save_images(images, &state.config.storage.images_path)
        .await
        .map_err(ApiError::from)?;
    if saved.is_empty() {
        return Ok(text);
    }
    let mut out = text;
    for image in saved {
        let token = image_token(&image.filename, state.config.server.api_key.as_deref())
            .map(|token| format!("?token={token}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "\n\n![{}](/images/{}{})",
            image.filename, image.filename, token
        ));
    }
    Ok(out)
}

async fn save_generated_images(
    state: &AppState,
    images: &[images::GeneratedImage],
) -> Result<Vec<ImageData>, ApiError> {
    tokio::fs::create_dir_all(&state.config.storage.images_path)
        .await
        .map_err(|error| ApiError::from(anyhow::anyhow!(error)))?;
    let mut out = Vec::new();
    let public_base_url = state
        .config
        .image_generation
        .public_base_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('/').to_string());
    for image in images {
        let data = images::decode_image_b64(image).map_err(ApiError::from)?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let sha256 = format!("{:x}", hasher.finalize());
        let filename = format!(
            "generated_{}.{}",
            &sha256[..24],
            images::image_ext(&image.mime_type)
        );
        let path = FsPath::new(&state.config.storage.images_path).join(&filename);
        if tokio::fs::metadata(&path).await.is_err() {
            tokio::fs::write(&path, data)
                .await
                .map_err(|error| ApiError::from(anyhow::anyhow!(error)))?;
        }
        let token = image_token(&filename, state.config.server.api_key.as_deref())
            .map(|token| format!("?token={token}"))
            .unwrap_or_default();
        if let Some(base) = public_base_url.as_ref() {
            out.push(ImageData {
                b64_json: None,
                url: Some(format!("{base}/images/{filename}{token}")),
                revised_prompt: image.revised_prompt.clone(),
            });
        } else {
            out.push(ImageData {
                b64_json: Some(image.b64_json.clone()),
                url: None,
                revised_prompt: image.revised_prompt.clone(),
            });
        }
    }
    Ok(out)
}

async fn image_generations(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ImageGenerationRequest>,
) -> Result<Response, ApiError> {
    verify_auth(&state, &headers)?;
    let start = started();
    let model = request
        .model
        .clone()
        .unwrap_or_else(|| state.config.image_generation.model.clone());
    match images::generate_images(&state.http, &state.config.image_generation, &request).await {
        Ok(images) => {
            let data = save_generated_images(&state, &images).await?;
            state.history.append(&HistoryRecord {
                ts: timestamp(),
                kind: "images.generations",
                model: &model,
                prompt_chars: request.prompt.chars().count(),
                output_chars: data.len(),
                latency_ms: start.elapsed().as_millis(),
                ok: true,
                error: None,
            });
            Ok(Json(ImageGenerationResponse {
                created: Utc::now().timestamp(),
                data,
            })
            .into_response())
        }
        Err(error) => {
            let detail = error.to_string();
            state.history.append(&HistoryRecord {
                ts: timestamp(),
                kind: "images.generations",
                model: &model,
                prompt_chars: request.prompt.chars().count(),
                output_chars: 0,
                latency_ms: start.elapsed().as_millis(),
                ok: false,
                error: Some(&detail),
            });
            Err(ApiError::from(error))
        }
    }
}

fn image_token(filename: &str, api_key: Option<&str>) -> Option<String> {
    let api_key = api_key.filter(|s| !s.is_empty())?;
    let mut hasher = Sha256::new();
    hasher.update(filename.as_bytes());
    hasher.update(b":");
    hasher.update(api_key.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    Some(digest[..24].to_string())
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

    let mut input = messages_to_gemini_input(&request.messages);
    let extra = chat_extra_instructions(&request);
    if !extra.is_empty() {
        input.prompt.push_str("\n\n[system]\n");
        input.prompt.push_str(&extra);
    }
    let prompt = input.prompt;
    let model_name = request.model.clone();
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = Utc::now().timestamp();
    let start = started();
    let output = match state
        .gemini
        .generate_output(&model_name, &prompt, &input.attachments)
        .await
    {
        Ok(output) => {
            let output_text = append_image_markdown(&state, output.text, &output.images).await?;
            state.history.append(&HistoryRecord {
                ts: timestamp(),
                kind: "chat.completions",
                model: &model_name,
                prompt_chars: prompt.chars().count(),
                output_chars: output_text.chars().count(),
                latency_ms: start.elapsed().as_millis(),
                ok: true,
                error: None,
            });
            output_text
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

    let processed = process_output(&output);
    let prompt_tokens = estimate_tokens(&prompt);
    let completion_tokens = estimate_tokens(&processed.storage_text);
    let usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        prompt_tokens_details: Some(json!({"cached_tokens": 0})),
        completion_tokens_details: Some(json!({"reasoning_tokens": 0})),
    };

    if request.stream.unwrap_or(false) {
        let prompt_tokens = estimate_tokens(&prompt);
        let id = completion_id.clone();
        let model = model_name.clone();
        let visible = processed.visible_text.clone().unwrap_or_default();
        let tool_calls = processed.tool_calls.clone();
        let finish_reason = processed.finish_reason.clone();
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
                usage: None,
            };
            yield Ok::<_, std::convert::Infallible>(Event::default().data(serde_json::to_string(&role).unwrap()));

            if !visible.is_empty() {
                let content = StreamChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: json!({"content": visible}),
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Ok(Event::default().data(serde_json::to_string(&content).unwrap()));
            }

            if !tool_calls.is_empty() {
                let tool_delta = tool_calls.iter().enumerate().map(|(index, call)| {
                    json!({
                        "index": index,
                        "id": &call.id,
                        "type": &call.kind,
                        "function": &call.function,
                    })
                }).collect::<Vec<_>>();
                let tools = StreamChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: json!({"tool_calls": tool_delta}),
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Ok(Event::default().data(serde_json::to_string(&tools).unwrap()));
            }

            let done = StreamChunk {
                id,
                object: "chat.completion.chunk",
                created,
                model,
                choices: vec![StreamChoice {
                    index: 0,
                    delta: json!({}),
                    finish_reason: Some(finish_reason),
                }],
                usage: Some(Usage {
                    prompt_tokens,
                    completion_tokens: estimate_tokens(&processed.storage_text),
                    total_tokens: prompt_tokens + estimate_tokens(&processed.storage_text),
                    prompt_tokens_details: Some(json!({"cached_tokens": 0})),
                    completion_tokens_details: Some(json!({"reasoning_tokens": 0})),
                }),
            };
            yield Ok(Event::default().data(serde_json::to_string(&done).unwrap()));
            yield Ok(Event::default().data("[DONE]"));
        };
        return Ok(Sse::new(s).into_response());
    }

    let payload = ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion",
        created,
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: processed.visible_text,
                tool_calls: if processed.tool_calls.is_empty() {
                    None
                } else {
                    Some(processed.tool_calls)
                },
                reasoning_content: None,
            },
            finish_reason: processed.finish_reason,
        }],
        usage,
    };
    Ok(Json(payload).into_response())
}

async fn create_response(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ResponseCreateRequest>,
) -> Result<Response, ApiError> {
    verify_auth(&state, &headers)?;
    let mut input = response_input_to_gemini_input(&request.input, request.instructions.as_ref());
    let extra = response_extra_instructions(&request);
    if !extra.is_empty() {
        input.prompt.push_str("\n\n[system]\n");
        input.prompt.push_str(&extra);
    }
    let prompt = input.prompt;
    if prompt.trim().is_empty() {
        return Err(ApiError::bad_request("input is required"));
    }
    let response_id = format!("resp_{}", Uuid::new_v4());
    let created = Utc::now().timestamp();
    let start = started();
    let output = match state
        .gemini
        .generate_output(&request.model, &prompt, &input.attachments)
        .await
    {
        Ok(output) => {
            let output_text = append_image_markdown(&state, output.text, &output.images).await?;
            state.history.append(&HistoryRecord {
                ts: timestamp(),
                kind: "responses",
                model: &request.model,
                prompt_chars: prompt.chars().count(),
                output_chars: output_text.chars().count(),
                latency_ms: start.elapsed().as_millis(),
                ok: true,
                error: None,
            });
            output_text
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

    let processed = process_output(&output);
    let response_output = processed.visible_text.clone().unwrap_or_default();

    if request.stream.unwrap_or(false) {
        let id = response_id.clone();
        let model = request.model.clone();
        let final_payload = response_payload(
            &id,
            created,
            &model,
            &response_output,
            estimate_tokens(&prompt),
            estimate_tokens(&processed.storage_text),
            &processed.tool_calls,
            request.metadata.clone(),
            request.tools.clone(),
            request.tool_choice.clone(),
        );
        let s = stream! {
            yield Ok::<_, std::convert::Infallible>(Event::default().event("response.created").data(serde_json::to_string(&json!({
                "type": "response.created",
                "response": {"id": id, "object": "response", "created_at": created, "model": model, "status": "in_progress", "metadata": request.metadata.clone().unwrap_or_else(|| json!({}))}
            })).unwrap()));
            if !response_output.is_empty() {
                yield Ok(Event::default().event("response.output_text.delta").data(serde_json::to_string(&json!({
                    "type": "response.output_text.delta",
                    "item_id": "msg_0",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": response_output,
                })).unwrap()));
            }
            yield Ok(Event::default().event("response.completed").data(serde_json::to_string(&json!({
                "type": "response.completed",
                "response": final_payload,
            })).unwrap()));
            yield Ok(Event::default().data("[DONE]"));
        };
        return Ok(Sse::new(s).into_response());
    }

    let prompt_tokens = estimate_tokens(&prompt);
    let completion_tokens = estimate_tokens(&processed.storage_text);
    let payload = response_payload(
        &response_id,
        created,
        &request.model,
        &response_output,
        prompt_tokens,
        completion_tokens,
        &processed.tool_calls,
        request.metadata,
        request.tools,
        request.tool_choice,
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
    tool_calls: &[openai::ToolCall],
    metadata: Option<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
) -> Value {
    let mut output_items = vec![json!({
        "id": "msg_0",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": output, "annotations": []}],
    })];
    for call in tool_calls {
        output_items.push(json!({
            "id": call.id,
            "type": "tool_call",
            "status": "completed",
            "function": &call.function,
        }));
    }
    json!({
        "id": id,
        "object": "response",
        "created_at": created,
        "completed_at": Utc::now().timestamp(),
        "status": "completed",
        "model": model,
        "output": output_items,
        "metadata": metadata.unwrap_or_else(|| json!({})),
        "tools": tools.unwrap_or_default(),
        "tool_choice": tool_choice.unwrap_or_else(|| json!("auto")),
        "text": {"format": {"type": "text"}},
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    })
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

    fn not_found(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
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
