mod config;
mod gemini;
mod history;
mod images;
mod openai;
mod routing;

use std::{
    collections::HashMap, env, fs as std_fs, path::Path as FsPath, sync::Arc, time::UNIX_EPOCH,
};

use async_stream::stream;
use axum::{
    Json, Router,
    extract::{Path as AxPath, Query, State},
    http::{HeaderMap, StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response, Sse, sse::Event},
    routing::{get, post},
};
use base64::{Engine as _, engine::general_purpose};
use chrono::{Local, Timelike, Utc};
use config::{Config, WarmGenerateConfig};
use gemini::{
    GeminiImage, GeminiOutput, GeminiPool, extract_cookie_header_from_text,
    extract_web_tool_nonce_from_text, short_fingerprint,
};
use history::{HistoryRecord, HistoryStore, started, timestamp};
use images::{ImageData, ImageGenerationRequest, ImageGenerationResponse};
use openai::{
    AssistantMessage, AttachmentSource, ChatCompletionRequest, ChatCompletionResponse, Choice,
    InputAttachment, ModelData, ModelListResponse, ResponseCreateRequest, StreamChoice,
    StreamChunk, Usage, chat_extra_instructions, estimate_tokens, latest_user_plain_text,
    messages_to_gemini_input, process_output, response_extra_instructions,
    response_input_to_gemini_input,
};
use routing::{GenerationRoute, RouteRequest, choose_generation_route, image_tool_prompt};
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
    let session_refresh_secs = config.gemini.refresh_interval;
    let warm_generate = config.gemini.warm_generate.clone();
    let gemini = Arc::new(GeminiPool::new(
        config.gemini.clients.clone(),
        config.gemini.models.clone(),
        config.gemini.timeout,
        config.gemini.refresh_interval,
        append_builtin,
        config.gemini.temporary_chat(),
    )?);
    let addr = config.server.addr()?;
    if let Err(error) = gemini.refresh_runtime_models().await {
        tracing::warn!(
            ?error,
            "Gemini runtime model discovery failed; continuing with configured models"
        );
    }
    spawn_session_warmup(gemini.clone(), session_refresh_secs);
    spawn_generate_warmup(gemini.clone(), warm_generate);
    let history = HistoryStore::new(config.storage.path.clone());
    let state = Arc::new(AppState {
        config,
        gemini,
        history,
        http: reqwest::Client::new(),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route(
            "/admin/session",
            get(admin_session_status).post(admin_session_reload),
        )
        .route("/admin/session/reload", post(admin_session_reload))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(create_response))
        .route("/v1/images/generations", post(image_generations))
        .route("/images/{filename}", get(get_image))
        .layer(CorsLayer::permissive().allow_private_network(true))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    tracing::info!("starting gemini-fastapi-rs at http://{}", addr);
    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

fn spawn_session_warmup(gemini: Arc<GeminiPool>, refresh_interval_secs: u64) {
    let period_secs = refresh_interval_secs.saturating_sub(60).max(60);
    tokio::spawn(async move {
        let period = std::time::Duration::from_secs(period_secs);
        let mut ticker = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
        loop {
            ticker.tick().await;
            match gemini.refresh_sessions().await {
                Ok(()) => tracing::info!(period_secs, "Gemini session proactive refresh completed"),
                Err(error) => tracing::warn!(
                    ?error,
                    period_secs,
                    "Gemini session proactive refresh failed"
                ),
            }
        }
    });
}

fn spawn_generate_warmup(gemini: Arc<GeminiPool>, config: WarmGenerateConfig) {
    if !config.enabled {
        return;
    }

    let interval_secs = config.interval.max(60);
    let initial_delay_secs = config.initial_delay.min(interval_secs);
    tracing::info!(
        model = config.model.as_str(),
        interval_secs,
        active_periods = ?config.active_periods,
        "Gemini generate warmup enabled"
    );
    tokio::spawn(async move {
        let period = std::time::Duration::from_secs(interval_secs);
        let first_tick =
            tokio::time::Instant::now() + std::time::Duration::from_secs(initial_delay_secs);
        let mut ticker = tokio::time::interval_at(first_tick, period);
        loop {
            ticker.tick().await;
            if !warm_generate_active_now(&config.active_periods) {
                tracing::debug!(
                    model = config.model.as_str(),
                    active_periods = ?config.active_periods,
                    "Gemini generate warmup skipped outside active period"
                );
                continue;
            }
            let started = std::time::Instant::now();
            match gemini
                .generate_output(&config.model, &config.prompt, &[])
                .await
            {
                Ok(output) => tracing::info!(
                    model = config.model.as_str(),
                    interval_secs,
                    elapsed_ms = started.elapsed().as_millis(),
                    output_chars = output.text.chars().count(),
                    "Gemini generate warmup completed"
                ),
                Err(error) => tracing::warn!(
                    ?error,
                    model = config.model.as_str(),
                    interval_secs,
                    "Gemini generate warmup failed"
                ),
            }
        }
    });
}

fn warm_generate_active_now(periods: &[String]) -> bool {
    if periods.is_empty() {
        return true;
    }
    let now = Local::now();
    let minute_of_day = now.hour() * 60 + now.minute();
    warm_generate_active_at(periods, minute_of_day)
}

fn warm_generate_active_at(periods: &[String], minute_of_day: u32) -> bool {
    periods.iter().any(|period| {
        parse_warm_period(period)
            .map(|(start, end)| minute_in_period(minute_of_day, start, end))
            .unwrap_or(false)
    })
}

fn parse_warm_period(period: &str) -> Option<(u32, u32)> {
    let (start, end) = period.split_once('-')?;
    Some((parse_hhmm(start.trim())?, parse_hhmm(end.trim())?))
}

fn parse_hhmm(value: &str) -> Option<u32> {
    let (hour, minute) = value.split_once(':')?;
    let hour: u32 = hour.parse().ok()?;
    let minute: u32 = minute.parse().ok()?;
    if hour > 23 || minute > 59 {
        return None;
    }
    Some(hour * 60 + minute)
}

fn minute_in_period(now: u32, start: u32, end: u32) -> bool {
    if start == end {
        return true;
    }
    if start < end {
        now >= start && now < end
    } else {
        now >= start || now < end
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "status": "ok",
        "implementation": "rust",
        "clients": state.gemini.client_ids(),
    }))
}

async fn admin_session_status(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    verify_auth(&state, &headers)?;
    Ok(Json(json!({
        "ok": true,
        "clients": state.gemini.session_status().await,
        "files": {
            "cookie": runtime_file_status("GEMINI_COOKIE_FILE"),
            "web_tool_nonce": runtime_file_status("GEMINI_WEB_TOOL_NONCE_FILE"),
        }
    })))
}

async fn admin_session_reload(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    verify_auth(&state, &headers)?;
    let text = payload.get("text").and_then(Value::as_str).unwrap_or("");
    let cookie = payload
        .get("cookie")
        .and_then(Value::as_str)
        .and_then(extract_cookie_header_from_text)
        .or_else(|| extract_cookie_header_from_text(text));
    let web_tool_nonce = payload
        .get("web_tool_nonce")
        .or_else(|| payload.get("tool_nonce"))
        .and_then(Value::as_str)
        .and_then(extract_web_tool_nonce_from_text)
        .or_else(|| extract_web_tool_nonce_from_text(text));

    let mut updated = serde_json::Map::new();
    if let Some(cookie) = cookie {
        let path = env::var("GEMINI_COOKIE_FILE").map_err(|_| {
            ApiError::bad_request("GEMINI_COOKIE_FILE is not configured; cannot hot-update cookie")
        })?;
        tokio::fs::write(&path, format!("{cookie}\n"))
            .await
            .map_err(|error| ApiError::from(anyhow::anyhow!(error)))?;
        updated.insert(
            "cookie".to_string(),
            json!({"path": path, "len": cookie.len(), "sha256": short_fingerprint(&cookie)}),
        );
    }
    if let Some(web_tool_nonce) = web_tool_nonce {
        let path = env::var("GEMINI_WEB_TOOL_NONCE_FILE").map_err(|_| {
            ApiError::bad_request(
                "GEMINI_WEB_TOOL_NONCE_FILE is not configured; cannot hot-update web tool nonce",
            )
        })?;
        tokio::fs::write(&path, format!("{web_tool_nonce}\n"))
            .await
            .map_err(|error| ApiError::from(anyhow::anyhow!(error)))?;
        updated.insert(
            "web_tool_nonce".to_string(),
            json!({
                "path": path,
                "len": web_tool_nonce.len(),
                "sha256": short_fingerprint(&web_tool_nonce),
            }),
        );
    }
    if updated.is_empty() {
        return Err(ApiError::bad_request(
            "no cookie or web_tool_nonce found in request payload",
        ));
    }

    state.gemini.clear_sessions().await;
    let verify = payload
        .get("verify")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let verify_result = if verify {
        match state.gemini.refresh_sessions().await {
            Ok(()) => json!({"ok": true}),
            Err(error) => json!({"ok": false, "error": error.to_string()}),
        }
    } else {
        json!({"ok": null, "skipped": true})
    };

    Ok(Json(json!({
        "ok": true,
        "updated": updated,
        "session_verify": verify_result,
        "clients": state.gemini.session_status().await,
        "files": {
            "cookie": runtime_file_status("GEMINI_COOKIE_FILE"),
            "web_tool_nonce": runtime_file_status("GEMINI_WEB_TOOL_NONCE_FILE"),
        }
    })))
}

fn runtime_file_status(env_name: &str) -> Value {
    let Ok(path) = env::var(env_name) else {
        return json!({"configured": false});
    };
    let metadata = std_fs::metadata(&path).ok();
    let content = std_fs::read_to_string(&path).ok();
    let modified_unix = metadata
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    json!({
        "configured": true,
        "path": path,
        "exists": metadata.is_some(),
        "len": metadata.as_ref().map(|m| m.len()),
        "modified_unix": modified_unix,
        "sha256": content.as_deref().map(short_fingerprint),
    })
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
    headers: &HeaderMap,
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
    let base_url = image_base_url_for_chat(state, headers);
    let cleaned_text = strip_generated_image_placeholders(&text);
    let mut out = if cleaned_text.trim().is_empty() {
        "Image generated.".to_string()
    } else {
        cleaned_text
    };
    for image in saved {
        append_image_link(&mut out, state, base_url.as_deref(), &image.filename);
    }
    Ok(out)
}

async fn append_generated_image_markdown(
    state: &AppState,
    headers: &HeaderMap,
    text: String,
    images: &[images::GeneratedImage],
) -> Result<String, ApiError> {
    if images.is_empty() {
        return Ok(text);
    }
    let saved = persist_generated_images(state, images).await?;
    if saved.is_empty() {
        return Ok(text);
    }
    let base_url = image_base_url_for_chat(state, headers);
    let mut out = if text.trim().is_empty() {
        "Image generated.".to_string()
    } else {
        text
    };
    for image in saved {
        append_image_link(&mut out, state, base_url.as_deref(), &image.filename);
    }
    Ok(out)
}

fn configured_image_base_url(state: &AppState) -> Option<String> {
    state
        .config
        .image_generation
        .public_base_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('/').to_string())
}

fn image_base_url_for_chat(state: &AppState, headers: &HeaderMap) -> Option<String> {
    configured_image_base_url(state).or_else(|| request_base_url(headers))
}

fn append_image_link(out: &mut String, state: &AppState, base_url: Option<&str>, filename: &str) {
    let token = image_token(filename, state.config.server.api_key.as_deref())
        .map(|token| format!("?token={token}"))
        .unwrap_or_default();
    let path = format!("/images/{filename}{token}");
    let url = base_url.map(|base| format!("{base}{path}")).unwrap_or(path);
    out.push_str(&format!(
        "

![{filename}]({url})

[Open image]({url})"
    ));
}

fn strip_generated_image_placeholders(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim();
            !(trimmed.starts_with("http://googleusercontent.com/image_generation_content/")
                || trimmed.starts_with("https://googleusercontent.com/image_generation_content/"))
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn stream_tail<'a>(streamed: &str, final_text: &'a str) -> Option<&'a str> {
    if final_text.is_empty() {
        None
    } else if streamed.is_empty() {
        Some(final_text)
    } else if final_text.starts_with(streamed) {
        let tail = &final_text[streamed.len()..];
        (!tail.is_empty()).then_some(tail)
    } else {
        None
    }
}

fn request_base_url(headers: &HeaderMap) -> Option<String> {
    let host = headers
        .get("host")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let proto = headers
        .get("x-forwarded-proto")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("http");
    Some(format!("{proto}://{host}"))
}

fn vision_worker_enabled(state: &AppState) -> bool {
    state.config.image_generation.worker_endpoint().is_some()
        && state
            .config
            .image_generation
            .resolved_worker_token()
            .is_some()
}

fn guess_attachment_content_type(filename: &str, explicit: Option<&str>) -> String {
    if let Some(value) = explicit.map(str::trim).filter(|value| !value.is_empty()) {
        return value.to_string();
    }
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg".to_string()
    } else if lower.ends_with(".webp") {
        "image/webp".to_string()
    } else if lower.ends_with(".gif") {
        "image/gif".to_string()
    } else {
        "image/png".to_string()
    }
}

async fn worker_image_payload(
    state: &AppState,
    attachment: &InputAttachment,
) -> anyhow::Result<Value> {
    let (data, content_type) = match &attachment.source {
        AttachmentSource::Data(data) => (
            data.clone(),
            guess_attachment_content_type(&attachment.filename, attachment.content_type.as_deref()),
        ),
        AttachmentSource::Url(url) => {
            let response = state
                .http
                .get(url)
                .timeout(std::time::Duration::from_millis(
                    state.config.image_generation.worker_timeout_ms.min(60_000),
                ))
                .send()
                .await?
                .error_for_status()?;
            let content_type = response
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .map(ToString::to_string)
                .unwrap_or_else(|| {
                    guess_attachment_content_type(
                        &attachment.filename,
                        attachment.content_type.as_deref(),
                    )
                });
            (response.bytes().await?.to_vec(), content_type)
        }
    };
    Ok(json!({
        "filename": attachment.filename,
        "mime": content_type,
        "data_base64": general_purpose::STANDARD.encode(data),
    }))
}

async fn generate_vision_worker_output(
    state: &AppState,
    prompt: &str,
    attachments: &[InputAttachment],
) -> anyhow::Result<GeminiOutput> {
    let worker_url = state
        .config
        .image_generation
        .worker_endpoint()
        .ok_or_else(|| anyhow::anyhow!("Gemini vision worker_url is not configured"))?;
    let token = state
        .config
        .image_generation
        .resolved_worker_token()
        .ok_or_else(|| anyhow::anyhow!("Gemini vision worker token is not configured"))?;
    let mut images = Vec::new();
    for attachment in attachments {
        images.push(worker_image_payload(state, attachment).await?);
    }
    if images.is_empty() {
        return Err(anyhow::anyhow!(
            "Gemini vision worker requires at least one image"
        ));
    }
    let body = json!({
        "prompt": prompt,
        "images": images,
        "timeout_ms": state.config.image_generation.worker_timeout_ms,
    });
    let response = state
        .http
        .post(format!("{worker_url}/vision"))
        .bearer_auth(token)
        .timeout(std::time::Duration::from_millis(
            state
                .config
                .image_generation
                .worker_timeout_ms
                .saturating_add(30_000),
        ))
        .json(&body)
        .send()
        .await?;
    let status = response.status();
    let value: Value = response.json().await.unwrap_or_else(|_| json!({}));
    if !status.is_success() || value.get("ok").and_then(Value::as_bool) == Some(false) {
        return Err(anyhow::anyhow!(
            "Gemini vision worker failed with status {status}: {}",
            value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        ));
    }
    let text = value
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if text.is_empty() {
        return Err(anyhow::anyhow!("Gemini vision worker returned empty text"));
    }
    Ok(GeminiOutput {
        text,
        images: Vec::new(),
    })
}

struct PersistedGeneratedImage {
    filename: String,
    b64_json: String,
    revised_prompt: Option<String>,
}

async fn persist_generated_images(
    state: &AppState,
    images: &[images::GeneratedImage],
) -> Result<Vec<PersistedGeneratedImage>, ApiError> {
    tokio::fs::create_dir_all(&state.config.storage.images_path)
        .await
        .map_err(|error| ApiError::from(anyhow::anyhow!(error)))?;
    let mut out = Vec::new();
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
        out.push(PersistedGeneratedImage {
            filename,
            b64_json: image.b64_json.clone(),
            revised_prompt: image.revised_prompt.clone(),
        });
    }
    Ok(out)
}

async fn save_generated_images(
    state: &AppState,
    images: &[images::GeneratedImage],
) -> Result<Vec<ImageData>, ApiError> {
    let public_base_url = configured_image_base_url(state);
    let saved = persist_generated_images(state, images).await?;
    let mut out = Vec::new();
    for image in saved {
        let token = image_token(&image.filename, state.config.server.api_key.as_deref())
            .map(|token| format!("?token={token}"))
            .unwrap_or_default();
        if let Some(base) = public_base_url.as_ref() {
            out.push(ImageData {
                b64_json: None,
                url: Some(format!("{base}/images/{}{}", image.filename, token)),
                revised_prompt: image.revised_prompt,
            });
        } else {
            out.push(ImageData {
                b64_json: Some(image.b64_json),
                url: None,
                revised_prompt: image.revised_prompt,
            });
        }
    }
    Ok(out)
}

async fn saved_image_to_openai_data(
    state: &AppState,
    filename: String,
    revised_prompt: Option<String>,
) -> Result<ImageData, ApiError> {
    let public_base_url = state
        .config
        .image_generation
        .public_base_url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('/').to_string());
    let token = image_token(&filename, state.config.server.api_key.as_deref())
        .map(|token| format!("?token={token}"))
        .unwrap_or_default();
    if let Some(base) = public_base_url.as_ref() {
        return Ok(ImageData {
            b64_json: None,
            url: Some(format!("{base}/images/{filename}{token}")),
            revised_prompt,
        });
    }
    let path = FsPath::new(&state.config.storage.images_path).join(&filename);
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|error| ApiError::from(anyhow::anyhow!(error)))?;
    Ok(ImageData {
        b64_json: Some(general_purpose::STANDARD.encode(bytes)),
        url: None,
        revised_prompt,
    })
}

async fn generate_web_images(
    state: &AppState,
    request: &ImageGenerationRequest,
) -> Result<Vec<ImageData>, ApiError> {
    let n = request.n.unwrap_or(1).clamp(1, 4);
    let prompt = if n > 1 {
        format!("请生成 {n} 张图片：{}", request.prompt.trim())
    } else {
        request.prompt.trim().to_string()
    };
    let output = state
        .gemini
        .generate_web_image_output(&state.config.image_generation.web_model, &prompt)
        .await
        .map_err(ApiError::from)?;
    if output.images.is_empty() {
        let refusal = output.text.trim();
        let detail = if refusal.is_empty() {
            "Gemini Web returned no generated images".to_string()
        } else {
            format!("Gemini Web returned no generated images: {refusal}")
        };
        return Err(ApiError::from(anyhow::anyhow!(detail)));
    }
    let saved = state
        .gemini
        .save_images(&output.images, &state.config.storage.images_path)
        .await
        .map_err(ApiError::from)?;
    let mut data = Vec::new();
    for image in saved {
        data.push(
            saved_image_to_openai_data(state, image.filename, Some(request.prompt.clone())).await?,
        );
    }
    if data.is_empty() {
        return Err(ApiError::from(anyhow::anyhow!(
            "Gemini Web generated images but they could not be downloaded"
        )));
    }
    Ok(data)
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
    let backend = state.config.image_generation.backend.as_str();
    let result: Result<Vec<ImageData>, ApiError> = match backend {
        "gemini_web" | "web" | "free_web" => generate_web_images(&state, &request).await,
        "auto" => match generate_web_images(&state, &request).await {
            Ok(data) => Ok(data),
            Err(web_error) => {
                match images::generate_images(&state.http, &state.config.image_generation, &request)
                    .await
                {
                    Ok(images) => save_generated_images(&state, &images).await,
                    Err(api_error) => Err(ApiError::from(anyhow::anyhow!(
                        "Gemini Web failed first: {}; official API fallback failed: {}",
                        web_error.detail,
                        api_error
                    ))),
                }
            }
        },
        _ => match images::generate_images(&state.http, &state.config.image_generation, &request)
            .await
        {
            Ok(images) => save_generated_images(&state, &images).await,
            Err(error) => Err(ApiError::from(error)),
        },
    };

    match result {
        Ok(data) => {
            record_history(
                &state,
                "images.generations",
                &model,
                request.prompt.chars().count(),
                data.len(),
                &start,
                true,
                None,
            );
            Ok(Json(ImageGenerationResponse {
                created: Utc::now().timestamp(),
                data,
            })
            .into_response())
        }
        Err(error) => {
            let detail = error.detail.clone();
            record_history(
                &state,
                "images.generations",
                &model,
                request.prompt.chars().count(),
                0,
                &start,
                false,
                Some(&detail),
            );
            Err(error)
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

fn append_text_only_image_guard(
    image_generation_enabled: bool,
    has_input_attachments: bool,
    prompt: &mut String,
) {
    if image_generation_enabled || has_input_attachments {
        return;
    }
    prompt.push_str("\n\n[system]\n");
    prompt.push_str(
        "This API bridge is text-only. Image creation/search/browsing tools are disabled. Never call or mention image creation availability, sign-in status, region availability, or image search. If the user talks about drawing, UI mockups, pictures, or images, answer as normal text. If the user explicitly asks you to create/generate/draw an image, say briefly that this endpoint only supports text and offer a text prompt, layout spec, or implementation plan instead."
    );
}

async fn generate_web_image_tool_output(
    state: &AppState,
    headers: &HeaderMap,
    prompt: &str,
) -> Result<String, ApiError> {
    let output = state
        .gemini
        .generate_web_image_output(&state.config.image_generation.web_model, prompt)
        .await
        .map_err(ApiError::from)?;
    if output.images.is_empty() {
        let refusal = output.text.trim();
        let detail = if refusal.is_empty() {
            "Gemini Web returned no generated images".to_string()
        } else {
            format!("Gemini Web returned no generated images: {refusal}")
        };
        return Err(ApiError::from(anyhow::anyhow!(detail)));
    }
    append_image_markdown(state, headers, output.text, &output.images).await
}

fn image_tool_failure_message(detail: &str) -> String {
    let lower = detail.to_ascii_lowercase();
    if lower.contains("no generated image found before timeout")
        || lower.contains("operation timed out")
        || lower.contains("request timed out")
    {
        return "\u{751f}\u{56fe}\u{5931}\u{8d25}\u{ff1a}Gemini Web \u{5728} 120 \u{79d2}\u{5185}\u{6ca1}\u{6709}\u{751f}\u{6210}\u{56fe}\u{7247}\u{ff0c}\u{53ef}\u{80fd}\u{662f}\u{8d26}\u{53f7}/\u{5730}\u{533a}\u{9650}\u{5236}\u{6216}\u{5f53}\u{524d}\u{670d}\u{52a1}\u{5361}\u{4f4f}\u{3002}\u{6211}\u{5df2}\u{7ecf}\u{505c}\u{6b62}\u{7b49}\u{5f85}\u{ff0c}\u{907f}\u{514d}\u{62d6}\u{6162}\u{804a}\u{5929}\u{3002}".to_string();
    }
    if lower.contains("worker busy") {
        return "\u{751f}\u{56fe}\u{5931}\u{8d25}\u{ff1a}\u{751f}\u{56fe} worker \u{6b63}\u{5fd9}\u{ff0c}\u{7a0d}\u{540e}\u{518d}\u{8bd5}\u{3002}".to_string();
    }
    format!("Image generation failed: {detail}")
}

async fn generate_backend_image_tool_output(
    state: &AppState,
    headers: &HeaderMap,
    prompt: &str,
) -> Result<String, ApiError> {
    let request = ImageGenerationRequest {
        model: Some(state.config.image_generation.model.clone()),
        prompt: image_tool_prompt(prompt),
        n: Some(1),
        size: None,
        quality: None,
        response_format: None,
        output_format: None,
        user: None,
    };
    let images = images::generate_images(&state.http, &state.config.image_generation, &request)
        .await
        .map_err(ApiError::from)?;
    append_generated_image_markdown(state, headers, "Image generated.".to_string(), &images).await
}

async fn generate_image_tool_output(
    state: &AppState,
    headers: &HeaderMap,
    prompt: &str,
) -> Result<String, ApiError> {
    match state.config.image_generation.backend.as_str() {
        "gemini_web" | "web" | "free_web" => {
            generate_web_image_tool_output(state, headers, prompt).await
        }
        "auto" => match generate_web_image_tool_output(state, headers, prompt).await {
            Ok(output) => Ok(output),
            Err(web_error) => {
                match generate_backend_image_tool_output(state, headers, prompt).await {
                    Ok(output) => Ok(output),
                    Err(backend_error) => Err(ApiError::from(anyhow::anyhow!(
                        "Gemini Web failed first: {}; configured image backend failed: {}",
                        web_error.detail,
                        backend_error.detail
                    ))),
                }
            }
        },
        _ => generate_backend_image_tool_output(state, headers, prompt).await,
    }
}

fn trace_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get("x-obp-request-id")
        .or_else(|| headers.get("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("-")
        .to_string()
}

fn record_history(
    state: &AppState,
    kind: &str,
    model: &str,
    prompt_chars: usize,
    output_chars: usize,
    start: &std::time::Instant,
    ok: bool,
    error: Option<&str>,
) {
    state.history.append(&HistoryRecord {
        ts: timestamp(),
        kind,
        model,
        prompt_chars,
        output_chars,
        latency_ms: start.elapsed().as_millis(),
        ok,
        error,
    });
}

fn usage_for(prompt: &str, storage_text: &str) -> Usage {
    let prompt_tokens = estimate_tokens(prompt);
    let completion_tokens = estimate_tokens(storage_text);
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        prompt_tokens_details: Some(json!({"cached_tokens": 0})),
        completion_tokens_details: Some(json!({"reasoning_tokens": 0})),
    }
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
    append_text_only_image_guard(
        state.config.image_generation.is_enabled(),
        !input.attachments.is_empty(),
        &mut input.prompt,
    );
    let latest_user_text = latest_user_plain_text(&request.messages);
    let prompt = input.prompt;
    let model_name = request.model.clone();
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = Utc::now().timestamp();
    let start = started();
    let trace_id = trace_id_from_headers(&headers);
    let route = choose_generation_route(RouteRequest {
        image_generation_enabled: state.config.image_generation.is_enabled(),
        vision_worker_enabled: vision_worker_enabled(&state),
        has_attachments: !input.attachments.is_empty(),
        model: &model_name,
        latest_user_text: &latest_user_text,
    });
    let image_tool = route == GenerationRoute::ImageTool;
    let use_worker_vision = route == GenerationRoute::VisionWorker;
    tracing::info!(
        trace_id = %trace_id,
        model = %model_name,
        stream = request.stream.unwrap_or(false),
        prompt_chars = prompt.chars().count(),
        image_tool = image_tool,
        vision_worker = use_worker_vision,
        "Gemini FastAPI chat request accepted"
    );
    if use_worker_vision && request.stream.unwrap_or(false) {
        let state_for_stream = state.clone();
        let prompt_for_stream = prompt.clone();
        let attachments_for_stream = input.attachments.clone();
        let model_for_stream = model_name.clone();
        let id = completion_id.clone();
        let s = stream! {
            let role = StreamChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_for_stream.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: json!({"role": "assistant"}),
                    finish_reason: None,
                }],
                usage: None,
            };
            yield Ok::<_, std::convert::Infallible>(Event::default().data(serde_json::to_string(&role).unwrap()));

            let vision_job = generate_vision_worker_output(
                &state_for_stream,
                &prompt_for_stream,
                &attachments_for_stream,
            );
            tokio::pin!(vision_job);
            let mut heartbeat = tokio::time::interval_at(
                tokio::time::Instant::now() + std::time::Duration::from_secs(5),
                std::time::Duration::from_secs(5),
            );
            let output_text = loop {
                tokio::select! {
                    result = &mut vision_job => {
                        match result {
                            Ok(output) => {
                                record_history(

                                    &state_for_stream,

                                    "chat.completions.vision_worker",

                                    &model_for_stream,

                                    prompt_for_stream.chars().count(),

                                    output.text.chars().count(),

                                    &start,

                                    true,

                                    None,

                                );
                                break output.text;
                            }
                            Err(error) => {
                                let detail = error.to_string();
                                record_history(

                                    &state_for_stream,

                                    "chat.completions.vision_worker",

                                    &model_for_stream,

                                    prompt_for_stream.chars().count(),

                                    0,

                                    &start,

                                    false,

                                    Some(&detail),

                                );
                                break format!("Gemini vision worker failed: {detail}");
                            }
                        }
                    }
                    _ = heartbeat.tick() => {
                        yield Ok(Event::default().comment("keepalive"));
                    }
                }
            };
            let processed = process_output(&output_text);
            let visible = processed.visible_text.unwrap_or_default();
            if !visible.is_empty() {
                let content = StreamChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_for_stream.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: json!({"content": visible}),
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Ok(Event::default().data(serde_json::to_string(&content).unwrap()));
            }
            let done = StreamChunk {
                id,
                object: "chat.completion.chunk",
                created,
                model: model_for_stream.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: json!({}),
                    finish_reason: Some(processed.finish_reason),
                }],
                usage: Some(usage_for(&prompt_for_stream, &processed.storage_text)),
            };
            yield Ok(Event::default().data(serde_json::to_string(&done).unwrap()));
            yield Ok(Event::default().data("[DONE]"));
        };
        return Ok(Sse::new(s).into_response());
    }
    if image_tool && request.stream.unwrap_or(false) {
        let state_for_stream = state.clone();
        let headers_for_stream = headers.clone();
        let prompt_for_stream = prompt.clone();
        let image_prompt_for_stream = latest_user_text.clone();
        let model_for_stream = model_name.clone();
        let id = completion_id.clone();
        let s = stream! {
            let role = StreamChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_for_stream.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: json!({"role": "assistant"}),
                    finish_reason: None,
                }],
                usage: None,
            };
            yield Ok::<_, std::convert::Infallible>(Event::default().data(serde_json::to_string(&role).unwrap()));

            let image_job = async {
                match generate_image_tool_output(&state_for_stream, &headers_for_stream, &image_prompt_for_stream).await {
                    Ok(output_text) => {
                        record_history(

                            &state_for_stream,

                            "chat.completions.image_tool",

                            &model_for_stream,

                            image_prompt_for_stream.chars().count(),

                            output_text.chars().count(),

                            &start,

                            true,

                            None,

                        );
                        output_text
                    }
                    Err(error) => {
                        let detail = error.detail;
                        record_history(

                            &state_for_stream,

                            "chat.completions.image_tool",

                            &model_for_stream,

                            image_prompt_for_stream.chars().count(),

                            0,

                            &start,

                            false,

                            Some(&detail),

                        );
                        format!("Image generation failed: {detail}")
                    }
                }
            };
            tokio::pin!(image_job);
            let mut heartbeat = tokio::time::interval_at(
                tokio::time::Instant::now() + std::time::Duration::from_secs(5),
                std::time::Duration::from_secs(5),
            );
            let output_text = loop {
                tokio::select! {
                    output_text = &mut image_job => break output_text,
                    _ = heartbeat.tick() => {
                        yield Ok(Event::default().comment("keepalive"));
                    }
                }
            };
            let processed = process_output(&output_text);
            let visible = processed.visible_text.unwrap_or_default();
            if !visible.is_empty() {
                let content = StreamChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_for_stream.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: json!({"content": visible}),
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Ok(Event::default().data(serde_json::to_string(&content).unwrap()));
            }

            let done = StreamChunk {
                id,
                object: "chat.completion.chunk",
                created,
                model: model_for_stream.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: json!({}),
                    finish_reason: Some(processed.finish_reason),
                }],
                usage: Some(usage_for(&prompt_for_stream, &processed.storage_text)),
            };
            yield Ok(Event::default().data(serde_json::to_string(&done).unwrap()));
            yield Ok(Event::default().data("[DONE]"));
        };
        return Ok(Sse::new(s).into_response());
    }
    if request.stream.unwrap_or(false) {
        let state_for_stream = state.clone();
        let headers_for_stream = headers.clone();
        let prompt_for_stream = prompt.clone();
        let attachments_for_stream = input.attachments.clone();
        let model_for_stream = model_name.clone();
        let trace_for_stream = trace_id.clone();
        let id = completion_id.clone();
        let s = stream! {
            let role = StreamChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_for_stream.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: json!({"role": "assistant"}),
                    finish_reason: None,
                }],
                usage: None,
            };
            yield Ok::<_, std::convert::Infallible>(Event::default().data(serde_json::to_string(&role).unwrap()));

            let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            let generate_job = state_for_stream.gemini.generate_output_with_progress(
                &model_for_stream,
                &prompt_for_stream,
                &attachments_for_stream,
                Some(progress_tx),
            );
            tokio::pin!(generate_job);
            let mut heartbeat = tokio::time::interval_at(
                tokio::time::Instant::now() + std::time::Duration::from_secs(3),
                std::time::Duration::from_secs(3),
            );
            let mut streamed_visible = String::new();
            let mut first_delta_ms: Option<u128> = None;
            let mut progress_open = true;
            let generated = loop {
                tokio::select! {
                    maybe_delta = progress_rx.recv(), if progress_open => {
                        match maybe_delta {
                            Some(delta) if !delta.is_empty() => {
                                if first_delta_ms.is_none() {
                                    let elapsed = start.elapsed().as_millis();
                                    first_delta_ms = Some(elapsed);
                                    tracing::info!(
                                        trace_id = %trace_for_stream,
                                        model = %model_for_stream,
                                        first_text_delta_ms = elapsed,
                                        "Gemini FastAPI first text delta"
                                    );
                                }
                                streamed_visible.push_str(&delta);
                                let content = StreamChunk {
                                    id: id.clone(),
                                    object: "chat.completion.chunk",
                                    created,
                                    model: model_for_stream.clone(),
                                    choices: vec![StreamChoice {
                                        index: 0,
                                        delta: json!({"content": delta}),
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };
                                yield Ok(Event::default().data(serde_json::to_string(&content).unwrap()));
                            }
                            Some(_) => {}
                            None => progress_open = false,
                        }
                    }
                    result = &mut generate_job => break result,
                    _ = heartbeat.tick() => {
                        yield Ok(Event::default().comment("keepalive"));
                    }
                }
            };

            let (output_text, force_final_visible) = match generated {
                Ok(output) => match append_image_markdown(
                    &state_for_stream,
                    &headers_for_stream,
                    output.text,
                    &output.images,
                )
                .await
                {
                    Ok(output_text) => {
                        record_history(

                            &state_for_stream,

                            "chat.completions",

                            &model_for_stream,

                            prompt_for_stream.chars().count(),

                            output_text.chars().count(),

                            &start,

                            true,

                            None,

                        );
                        (output_text, false)
                    }
                    Err(error) => {
                        let detail = error.detail;
                        record_history(

                            &state_for_stream,

                            "chat.completions",

                            &model_for_stream,

                            prompt_for_stream.chars().count(),

                            0,

                            &start,

                            false,

                            Some(&detail),

                        );
                        (format!("Gemini response postprocess failed: {detail}"), true)
                    }
                },
                Err(error) => {
                    let detail = error.to_string();
                    record_history(

                        &state_for_stream,

                        "chat.completions",

                        &model_for_stream,

                        prompt_for_stream.chars().count(),

                        0,

                        &start,

                        false,

                        Some(&detail),

                    );
                    (format!("Gemini request failed: {detail}"), true)
                }
            };
            let processed = process_output(&output_text);
            let final_visible = processed.visible_text.clone().unwrap_or_default();
            let visible_tail = if force_final_visible {
                final_visible.as_str()
            } else {
                stream_tail(&streamed_visible, &final_visible).unwrap_or("")
            };
            if !visible_tail.is_empty() {
                let content = StreamChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_for_stream.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: json!({"content": visible_tail}),
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Ok(Event::default().data(serde_json::to_string(&content).unwrap()));
            }

            if !processed.tool_calls.is_empty() {
                let tool_delta = processed.tool_calls.iter().enumerate().map(|(index, call)| {
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
                    model: model_for_stream.clone(),
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
                model: model_for_stream.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: json!({}),
                    finish_reason: Some(processed.finish_reason),
                }],
                usage: Some(usage_for(&prompt_for_stream, &processed.storage_text)),
            };
            yield Ok(Event::default().data(serde_json::to_string(&done).unwrap()));
            yield Ok(Event::default().data("[DONE]"));
        };
        return Ok(Sse::new(s).into_response());
    }

    let output = if image_tool {
        match generate_image_tool_output(&state, &headers, &latest_user_text).await {
            Ok(output_text) => {
                record_history(
                    &state,
                    "chat.completions.image_tool",
                    &model_name,
                    latest_user_text.chars().count(),
                    output_text.chars().count(),
                    &start,
                    true,
                    None,
                );
                output_text
            }
            Err(error) => {
                let detail = error.detail.clone();
                record_history(
                    &state,
                    "chat.completions.image_tool",
                    &model_name,
                    latest_user_text.chars().count(),
                    0,
                    &start,
                    false,
                    Some(&detail),
                );
                image_tool_failure_message(&detail)
            }
        }
    } else {
        let generated = if use_worker_vision {
            generate_vision_worker_output(&state, &prompt, &input.attachments).await
        } else {
            state
                .gemini
                .generate_output(&model_name, &prompt, &input.attachments)
                .await
        };
        match generated {
            Ok(output) => {
                let output_text =
                    append_image_markdown(&state, &headers, output.text, &output.images).await?;
                record_history(
                    &state,
                    if use_worker_vision {
                        "chat.completions.vision_worker"
                    } else {
                        "chat.completions"
                    },
                    &model_name,
                    prompt.chars().count(),
                    output_text.chars().count(),
                    &start,
                    true,
                    None,
                );
                output_text
            }
            Err(error) => {
                let detail = error.to_string();
                record_history(
                    &state,
                    if use_worker_vision {
                        "chat.completions.vision_worker"
                    } else {
                        "chat.completions"
                    },
                    &model_name,
                    prompt.chars().count(),
                    0,
                    &start,
                    false,
                    Some(&detail),
                );
                return Err(ApiError::from(error));
            }
        }
    };

    let processed = process_output(&output);
    let usage = usage_for(&prompt, &processed.storage_text);

    if request.stream.unwrap_or(false) {
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
                usage: Some(usage_for(&prompt, &processed.storage_text)),
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
    append_text_only_image_guard(
        state.config.image_generation.is_enabled(),
        !input.attachments.is_empty(),
        &mut input.prompt,
    );
    let prompt = input.prompt;
    if prompt.trim().is_empty() {
        return Err(ApiError::bad_request("input is required"));
    }
    let response_id = format!("resp_{}", Uuid::new_v4());
    let created = Utc::now().timestamp();
    let start = started();
    let route = choose_generation_route(RouteRequest {
        image_generation_enabled: state.config.image_generation.is_enabled(),
        vision_worker_enabled: vision_worker_enabled(&state),
        has_attachments: !input.attachments.is_empty(),
        model: &request.model,
        latest_user_text: &prompt,
    });
    let image_tool = route == GenerationRoute::ImageTool;
    let use_worker_vision = route == GenerationRoute::VisionWorker;
    let output = if image_tool {
        match generate_image_tool_output(&state, &headers, &prompt).await {
            Ok(output_text) => {
                record_history(
                    &state,
                    "responses.image_tool",
                    &request.model,
                    prompt.chars().count(),
                    output_text.chars().count(),
                    &start,
                    true,
                    None,
                );
                output_text
            }
            Err(error) => {
                let detail = error.detail.clone();
                record_history(
                    &state,
                    "responses.image_tool",
                    &request.model,
                    prompt.chars().count(),
                    0,
                    &start,
                    false,
                    Some(&detail),
                );
                return Err(error);
            }
        }
    } else {
        let generated = if use_worker_vision {
            generate_vision_worker_output(&state, &prompt, &input.attachments).await
        } else {
            state
                .gemini
                .generate_output(&request.model, &prompt, &input.attachments)
                .await
        };
        match generated {
            Ok(output) => {
                let output_text =
                    append_image_markdown(&state, &headers, output.text, &output.images).await?;
                record_history(
                    &state,
                    if use_worker_vision {
                        "responses.vision_worker"
                    } else {
                        "responses"
                    },
                    &request.model,
                    prompt.chars().count(),
                    output_text.chars().count(),
                    &start,
                    true,
                    None,
                );
                output_text
            }
            Err(error) => {
                let detail = error.to_string();
                record_history(
                    &state,
                    if use_worker_vision {
                        "responses.vision_worker"
                    } else {
                        "responses"
                    },
                    &request.model,
                    prompt.chars().count(),
                    0,
                    &start,
                    false,
                    Some(&detail),
                );
                return Err(ApiError::from(error));
            }
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

#[cfg(test)]
mod tests {
    use super::{append_text_only_image_guard, warm_generate_active_at};

    fn periods(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn text_only_image_guard_blocks_gemini_image_tool_language() {
        let mut prompt = "[user]\nplease draw a cat image".to_string();
        append_text_only_image_guard(false, false, &mut prompt);
        assert!(prompt.contains("text-only"));
        assert!(prompt.contains("Never call or mention image creation availability"));
    }

    #[test]
    fn text_only_image_guard_is_skipped_when_image_backend_enabled() {
        let mut prompt = "[user]\nplease draw a cat image".to_string();
        append_text_only_image_guard(true, false, &mut prompt);
        assert!(!prompt.contains("text-only"));
    }

    #[test]
    fn text_only_image_guard_is_skipped_for_input_attachments() {
        let mut prompt = "[user]\nplease describe this attached image".to_string();
        append_text_only_image_guard(false, true, &mut prompt);
        assert!(!prompt.contains("text-only"));
    }

    #[test]
    fn warm_period_supports_same_day_window() {
        let config = periods(&["09:00-18:00"]);
        assert!(warm_generate_active_at(&config, 9 * 60));
        assert!(warm_generate_active_at(&config, 17 * 60 + 59));
        assert!(!warm_generate_active_at(&config, 18 * 60));
    }

    #[test]
    fn warm_period_supports_midnight_crossing_window() {
        let config = periods(&["07:00-01:30"]);
        assert!(warm_generate_active_at(&config, 23 * 60));
        assert!(warm_generate_active_at(&config, 60));
        assert!(!warm_generate_active_at(&config, 2 * 60));
    }

    #[test]
    fn warm_period_ignores_invalid_windows() {
        let config = periods(&["bad", "25:00-26:00"]);
        assert!(!warm_generate_active_at(&config, 12 * 60));
    }
}
