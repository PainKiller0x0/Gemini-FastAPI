use std::{
    collections::HashMap,
    env, fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, anyhow};
use base64::{Engine as _, engine::general_purpose};
use chrono::Utc;
use futures_util::StreamExt;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::{
    Client,
    header::{CONTENT_TYPE, HeaderMap, HeaderValue, ORIGIN, REFERER, SET_COOKIE, USER_AGENT},
    multipart,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, mpsc};
use uuid::Uuid;

use crate::config::{GeminiClientConfig, GeminiModelConfig};
use crate::openai::{AttachmentSource, InputAttachment};

const ENDPOINT_GOOGLE: &str = "https://www.google.com";
const ENDPOINT_INIT: &str = "https://gemini.google.com/app";
const ENDPOINT_GENERATE: &str = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate";
const ENDPOINT_BATCH_EXEC: &str = "https://gemini.google.com/_/BardChatUi/data/batchexecute";
const ENDPOINT_UPLOAD: &str = "https://content-push.googleapis.com/upload";

static TOKEN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"\"SNlM0e\":\s*\"(.*?)\""#).unwrap());
static BUILD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"\"cfb2h\":\s*\"(.*?)\""#).unwrap());
static SID_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"\"FdrFJe\":\s*\"(.*?)\""#).unwrap());
static LANG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"\"TuX5cc\":\s*\"(.*?)\""#).unwrap());
static PUSH_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"\"qKIAYe\":\s*\"(.*?)\""#).unwrap());
static BARD_ERROR_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"BardErrorInfo\\",\[(\d+)\]"#).unwrap());
static CURL_COOKIE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?is)(?:^|\s)(?:-b|--cookie)\s+["']([^"']+)["']"#).unwrap());
static COOKIE_HEADER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?im)^\s*cookie\s*:\s*(.+?)\s*$"#).unwrap());
static WEB_TOOL_NONCE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"![-_A-Za-z0-9]{1000,}"#).unwrap());

fn contains_cjk(text: &str) -> bool {
    text.chars().any(|ch| {
        let code = ch as u32;
        (0x3400..=0x4DBF).contains(&code)
            || (0x4E00..=0x9FFF).contains(&code)
            || (0xF900..=0xFAFF).contains(&code)
    })
}

fn request_language<'a>(session_language: &'a str, prompt: &str, image_mode: bool) -> &'a str {
    if image_mode || contains_cjk(prompt) {
        "zh-CN"
    } else if session_language.is_empty() {
        "en"
    } else {
        session_language
    }
}

#[derive(Debug, Clone)]
pub struct GeminiModel {
    pub model_name: String,
    pub model_header: HashMap<String, String>,
    pub owned_by: String,
}

#[derive(Debug, Clone)]
struct SessionState {
    access_token: String,
    build_label: Option<String>,
    session_id: Option<String>,
    push_id: Option<String>,
    language: String,
    cookie_header: String,
    created_at: Instant,
}

#[derive(Debug, Clone, Default)]
pub struct GeminiOutput {
    pub text: String,
    pub images: Vec<GeminiImage>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GeminiImage {
    pub url: String,
    pub title: String,
    pub alt: String,
    pub generated: bool,
    pub cid: Option<String>,
    pub rid: Option<String>,
    pub rcid: Option<String>,
    pub image_id: Option<String>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SavedImage {
    pub filename: String,
    pub sha256: String,
}

pub struct GeminiClient {
    http: Client,
    client_config: GeminiClientConfig,
    models: Mutex<Vec<GeminiModel>>,
    reqid: AtomicU64,
    session: Mutex<Option<SessionState>>,
    refresh_interval: Duration,
    temporary_chat: bool,
}

pub struct GeminiPool {
    clients: Vec<Arc<GeminiClient>>,
    cursor: AtomicUsize,
}

impl GeminiPool {
    pub fn new(
        client_configs: Vec<GeminiClientConfig>,
        configured_models: Vec<GeminiModelConfig>,
        timeout_seconds: u64,
        refresh_interval_seconds: u64,
        append_builtin: bool,
        temporary_chat: bool,
    ) -> anyhow::Result<Self> {
        if client_configs.is_empty() {
            return Err(anyhow!("at least one Gemini client is required"));
        }
        let mut clients = Vec::new();
        for config in client_configs {
            clients.push(Arc::new(GeminiClient::new(
                config,
                configured_models.clone(),
                timeout_seconds,
                refresh_interval_seconds,
                append_builtin,
                temporary_chat,
            )?));
        }
        Ok(Self {
            clients,
            cursor: AtomicUsize::new(0),
        })
    }

    pub fn client_ids(&self) -> Vec<&str> {
        self.clients.iter().map(|client| client.id()).collect()
    }

    pub async fn refresh_runtime_models(&self) -> anyhow::Result<usize> {
        let mut added = 0usize;
        let mut ok = false;
        let mut last_error = None;
        for client in &self.clients {
            match client.refresh_runtime_models().await {
                Ok(count) => {
                    added += count;
                    ok = true;
                }
                Err(error) => {
                    last_error = Some(error);
                }
            }
        }
        if ok {
            Ok(added)
        } else {
            Err(last_error.unwrap_or_else(|| anyhow!("Gemini model discovery failed")))
        }
    }

    pub async fn models(&self) -> Vec<GeminiModel> {
        let mut out = Vec::new();
        for client in &self.clients {
            for model in client.models().await {
                if !out
                    .iter()
                    .any(|existing: &GeminiModel| existing.model_name == model.model_name)
                {
                    out.push(model);
                }
            }
        }
        out
    }

    pub async fn generate_output(
        &self,
        model_name: &str,
        prompt: &str,
        attachments: &[InputAttachment],
    ) -> anyhow::Result<GeminiOutput> {
        self.generate_output_with_progress(model_name, prompt, attachments, None)
            .await
    }

    pub async fn generate_output_with_progress(
        &self,
        model_name: &str,
        prompt: &str,
        attachments: &[InputAttachment],
        progress: Option<mpsc::UnboundedSender<String>>,
    ) -> anyhow::Result<GeminiOutput> {
        let len = self.clients.len();
        let start = self.cursor.fetch_add(1, Ordering::Relaxed) % len;
        let mut last_error = None;
        for offset in 0..len {
            let index = (start + offset) % len;
            let client = &self.clients[index];
            match client
                .generate_output_with_progress(model_name, prompt, attachments, progress.clone())
                .await
            {
                Ok(output) => return Ok(output),
                Err(error) => {
                    tracing::warn!(
                        client = client.id(),
                        ?error,
                        "Gemini client failed; trying next client if available"
                    );
                    last_error = Some(error);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow!("all Gemini clients failed")))
    }

    pub async fn generate_web_image_output(
        &self,
        model_name: &str,
        prompt: &str,
    ) -> anyhow::Result<GeminiOutput> {
        let len = self.clients.len();
        let start = self.cursor.fetch_add(1, Ordering::Relaxed) % len;
        let mut last_error = None;
        for offset in 0..len {
            let index = (start + offset) % len;
            let client = &self.clients[index];
            match client.generate_web_image_output(model_name, prompt).await {
                Ok(output) => return Ok(output),
                Err(error) => {
                    tracing::warn!(
                        client = client.id(),
                        ?error,
                        "Gemini image client failed; trying next client if available"
                    );
                    last_error = Some(error);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow!("all Gemini clients failed")))
    }

    pub async fn save_images(
        &self,
        images: &[GeminiImage],
        dir: &str,
    ) -> anyhow::Result<Vec<SavedImage>> {
        let client = self
            .clients
            .first()
            .context("at least one Gemini client is required")?;
        client.save_images(images, dir).await
    }

    pub async fn clear_sessions(&self) {
        for client in &self.clients {
            client.clear_session().await;
        }
    }

    pub async fn refresh_sessions(&self) -> anyhow::Result<()> {
        let mut ok_count = 0usize;
        let mut last_error = None;
        for client in &self.clients {
            match client.reset_and_refresh_session().await {
                Ok(()) => ok_count += 1,
                Err(error) => last_error = Some(error),
            }
        }
        if ok_count > 0 {
            Ok(())
        } else {
            Err(last_error.unwrap_or_else(|| anyhow!("all Gemini sessions failed to refresh")))
        }
    }

    pub async fn session_status(&self) -> Vec<Value> {
        let mut status = Vec::new();
        for client in &self.clients {
            status.push(client.session_status().await);
        }
        status
    }
}

impl GeminiClient {
    pub fn new(
        client_config: GeminiClientConfig,
        configured_models: Vec<GeminiModelConfig>,
        timeout_seconds: u64,
        refresh_interval_seconds: u64,
        append_builtin: bool,
        temporary_chat: bool,
    ) -> anyhow::Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"));
        headers.insert(
            ORIGIN,
            HeaderValue::from_static("https://gemini.google.com"),
        );
        headers.insert(
            REFERER,
            HeaderValue::from_static("https://gemini.google.com/"),
        );

        let mut builder = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(timeout_seconds));
        if let Some(proxy) = client_config.proxy.as_deref().filter(|s| !s.is_empty()) {
            builder = builder.proxy(reqwest::Proxy::all(proxy)?);
        }
        let http = builder.build()?;

        let mut models = Vec::new();
        for model in configured_models {
            models.push(GeminiModel {
                model_name: model.model_name,
                model_header: model.model_header,
                owned_by: "custom".to_string(),
            });
        }
        if append_builtin {
            for model in builtin_models() {
                if !models.iter().any(|m| m.model_name == model.model_name) {
                    models.push(model);
                }
            }
        }

        Ok(Self {
            http,
            client_config,
            models: Mutex::new(models),
            reqid: AtomicU64::new(10_000),
            session: Mutex::new(None),
            refresh_interval: Duration::from_secs(refresh_interval_seconds),
            temporary_chat,
        })
    }

    pub async fn models(&self) -> Vec<GeminiModel> {
        self.models.lock().await.clone()
    }

    pub fn id(&self) -> &str {
        &self.client_config.id
    }

    pub async fn refresh_runtime_models(&self) -> anyhow::Result<usize> {
        self.ensure_session().await?;
        let session = self
            .session
            .lock()
            .await
            .clone()
            .context("Gemini session is not initialized")?;
        let reqid = self.reqid.fetch_add(100_000, Ordering::Relaxed);
        let language = if session.language.is_empty() {
            "en"
        } else {
            &session.language
        };
        let mut params = vec![
            ("rpcids".to_string(), "otAQ7b".to_string()),
            ("hl".to_string(), language.to_string()),
            ("_reqid".to_string(), reqid.to_string()),
            ("rt".to_string(), "c".to_string()),
            ("source-path".to_string(), "/app".to_string()),
        ];
        if let Some(build_label) = session.build_label.as_ref() {
            params.push(("bl".to_string(), build_label.clone()));
        }
        if let Some(session_id) = session.session_id.as_ref() {
            params.push(("f.sid".to_string(), session_id.clone()));
        }
        let payload = json!([[["otAQ7b", "[]", null, "generic"]]]).to_string();
        let form = vec![
            ("at", session.access_token.as_str()),
            ("f.req", payload.as_str()),
        ];
        let response = self
            .http
            .post(ENDPOINT_BATCH_EXEC)
            .query(&params)
            .header(
                CONTENT_TYPE,
                "application/x-www-form-urlencoded;charset=utf-8",
            )
            .header("X-Same-Domain", "1")
            .header(
                "x-goog-ext-525001261-jspb",
                "[1,null,null,null,null,null,null,null,[4]]",
            )
            .header("x-goog-ext-73010989-jspb", "[0]")
            .header("Cookie", session.cookie_header)
            .form(&form)
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "Gemini model discovery failed with status {}",
                response.status()
            ));
        }
        let raw = response.text().await?;
        let discovered = extract_runtime_models(&raw)?;
        let mut models = self.models.lock().await;
        let before = models.len();
        for model in discovered {
            if !models
                .iter()
                .any(|existing| existing.model_name == model.model_name)
            {
                models.push(model);
            }
        }
        Ok(models.len().saturating_sub(before))
    }

    #[allow(dead_code)]
    pub async fn generate(&self, model_name: &str, prompt: &str) -> anyhow::Result<String> {
        Ok(self.generate_output(model_name, prompt, &[]).await?.text)
    }

    pub async fn generate_output(
        &self,
        model_name: &str,
        prompt: &str,
        attachments: &[InputAttachment],
    ) -> anyhow::Result<GeminiOutput> {
        self.generate_output_with_progress(model_name, prompt, attachments, None)
            .await
    }

    pub async fn generate_output_with_progress(
        &self,
        model_name: &str,
        prompt: &str,
        attachments: &[InputAttachment],
        progress: Option<mpsc::UnboundedSender<String>>,
    ) -> anyhow::Result<GeminiOutput> {
        self.generate_output_inner(model_name, prompt, attachments, false, progress)
            .await
    }

    pub async fn generate_web_image_output(
        &self,
        model_name: &str,
        prompt: &str,
    ) -> anyhow::Result<GeminiOutput> {
        self.generate_output_inner(model_name, prompt, &[], true, None)
            .await
    }

    async fn generate_output_inner(
        &self,
        model_name: &str,
        prompt: &str,
        attachments: &[InputAttachment],
        image_mode: bool,
        progress: Option<mpsc::UnboundedSender<String>>,
    ) -> anyhow::Result<GeminiOutput> {
        if prompt.trim().is_empty() {
            return Err(anyhow!("prompt cannot be empty"));
        }
        let model = self
            .models
            .lock()
            .await
            .iter()
            .find(|m| m.model_name == model_name)
            .cloned()
            .ok_or_else(|| anyhow!("unknown model: {model_name}"))?;

        self.ensure_session().await?;
        let session = self
            .session
            .lock()
            .await
            .clone()
            .context("Gemini session is not initialized")?;
        if !attachments.is_empty() {
            self.send_bard_activity(&session).await?;
        }
        let file_data = self.upload_attachments(&session, attachments).await?;
        if !attachments.is_empty() {
            self.send_bard_activity(&session).await?;
        }

        let reqid = self.reqid.fetch_add(100_000, Ordering::Relaxed);
        let uuid = Uuid::new_v4().to_string().to_uppercase();
        let language = request_language(&session.language, prompt, image_mode);

        let inner = build_stream_generate_inner(
            prompt,
            language,
            file_data,
            &uuid,
            image_mode,
            self.temporary_chat,
        );

        let mut request_headers = HeaderMap::new();
        request_headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/x-www-form-urlencoded;charset=utf-8"),
        );
        request_headers.insert("X-Same-Domain", HeaderValue::from_static("1"));
        request_headers.insert(
            "x-goog-ext-525005358-jspb",
            HeaderValue::from_str(&format!(r#"["{}",1]"#, uuid))?,
        );
        for (name, value) in &model.model_header {
            request_headers.insert(
                name.parse::<reqwest::header::HeaderName>()?,
                HeaderValue::from_str(value)?,
            );
        }
        add_web_required_headers(&mut request_headers);
        if image_mode {
            add_image_tool_browser_headers(&mut request_headers);
            request_headers.insert(
                "x-goog-ext-73010990-jspb",
                HeaderValue::from_static("[0,0,0]"),
            );
            if let Some(header) =
                image_mode_model_header(&model, "F037BA73-BD98-4D15-8073-AE6F4E0BB60E")
            {
                request_headers
                    .insert("x-goog-ext-525001261-jspb", HeaderValue::from_str(&header)?);
            }
        }

        let inner_string = serde_json::to_string(&inner)?;
        let f_req = serde_json::to_string(&json!([null, inner_string]))?;
        let mut params = vec![
            ("hl".to_string(), language.to_string()),
            ("_reqid".to_string(), reqid.to_string()),
            ("rt".to_string(), "c".to_string()),
        ];
        if let Some(build_label) = session.build_label.as_ref() {
            params.push(("bl".to_string(), build_label.clone()));
        }
        if let Some(session_id) = session.session_id.as_ref() {
            params.push(("f.sid".to_string(), session_id.clone()));
        }
        let form = vec![
            ("at", session.access_token.as_str()),
            ("f.req", f_req.as_str()),
        ];

        let request_started = Instant::now();
        let response = self
            .http
            .post(ENDPOINT_GENERATE)
            .query(&params)
            .headers(request_headers)
            .header("Cookie", session.cookie_header.clone())
            .form(&form)
            .send()
            .await?;
        tracing::info!(
            client = self.id(),
            model = model_name,
            prompt_chars = prompt.chars().count(),
            headers_ms = request_started.elapsed().as_millis(),
            "Gemini generate response headers received"
        );

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            *self.session.lock().await = None;
            return Err(anyhow!(
                "Gemini generate failed with status {status}: {}",
                body.chars().take(300).collect::<String>()
            ));
        }

        let raw = read_response_text_with_progress(
            response,
            progress,
            self.id(),
            model_name,
            request_started,
        )
        .await?;
        extract_output_from_response(&raw)
    }

    async fn upload_attachments(
        &self,
        session: &SessionState,
        attachments: &[InputAttachment],
    ) -> anyhow::Result<Value> {
        if attachments.is_empty() {
            return Ok(Value::Null);
        }
        let push_id = session
            .push_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .context("Gemini upload push id not found")?;
        let mut uploaded = Vec::new();
        for attachment in attachments {
            let (filename, content_type, data) = self.load_attachment(attachment).await?;
            let part = multipart::Part::bytes(data)
                .file_name(filename.clone())
                .mime_str(&content_type)?;
            let form = multipart::Form::new().part("file", part);
            let response = self
                .http
                .post(ENDPOINT_UPLOAD)
                .header("X-Tenant-Id", "bard-storage")
                .header("Push-ID", push_id)
                .header("Cookie", session.cookie_header.clone())
                .multipart(form)
                .send()
                .await?;
            if !response.status().is_success() {
                return Err(anyhow!(
                    "Gemini upload failed with status {}",
                    response.status()
                ));
            }
            let file_id = response.text().await?;
            uploaded.push(json!([[file_id], filename]));
        }
        Ok(Value::Array(uploaded))
    }

    async fn send_bard_activity(&self, session: &SessionState) -> anyhow::Result<()> {
        let reqid = self.reqid.fetch_add(100_000, Ordering::Relaxed);
        let language = if session.language.is_empty() {
            "en"
        } else {
            &session.language
        };
        let mut params = vec![
            ("rpcids".to_string(), "ESY5D".to_string()),
            ("hl".to_string(), language.to_string()),
            ("_reqid".to_string(), reqid.to_string()),
            ("rt".to_string(), "c".to_string()),
            ("source-path".to_string(), "/app".to_string()),
        ];
        if let Some(build_label) = session.build_label.as_ref() {
            params.push(("bl".to_string(), build_label.clone()));
        }
        if let Some(session_id) = session.session_id.as_ref() {
            params.push(("f.sid".to_string(), session_id.clone()));
        }
        let payload =
            json!([[["ESY5D", "[[[\"bard_activity_enabled\"]]]", null, "generic"]]]).to_string();
        let form = vec![
            ("at", session.access_token.as_str()),
            ("f.req", payload.as_str()),
        ];
        let response = self
            .http
            .post(ENDPOINT_BATCH_EXEC)
            .query(&params)
            .header(
                CONTENT_TYPE,
                "application/x-www-form-urlencoded;charset=utf-8",
            )
            .header("X-Same-Domain", "1")
            .header(
                "x-goog-ext-525001261-jspb",
                "[1,null,null,null,null,null,null,null,[4]]",
            )
            .header("x-goog-ext-73010989-jspb", "[0]")
            .header("x-goog-ext-73010990-jspb", "[0]")
            .header("Cookie", session.cookie_header.clone())
            .form(&form)
            .send()
            .await?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!(
                "Gemini activity RPC failed with status {}",
                response.status()
            ))
        }
    }

    async fn send_bard_settings(&self, session: &SessionState) -> anyhow::Result<()> {
        let keys = [
            "bard_activity_enabled",
            "disable_generated_image_download_dialog",
            "disable_image_upload_tooltip",
            "gempix_discovery_banner_dismissal_count",
            "gempix_discovery_banner_last_dismissed",
            "has_seen_image_grams_discovery_banner",
            "has_seen_image_preview_in_input_area_tooltip",
            "has_seen_kallo_discovery_banner",
            "has_seen_kallo_tooltip",
            "has_seen_model_tooltip_in_input_area_for_gempix",
            "upload_disclaimer_last_consent_time_sec",
            "web_and_app_activity_enabled",
        ];
        let reqid = self.reqid.fetch_add(100_000, Ordering::Relaxed);
        let language = if session.language.is_empty() {
            "en"
        } else {
            &session.language
        };
        let mut params = vec![
            ("rpcids".to_string(), "ESY5D".to_string()),
            ("hl".to_string(), language.to_string()),
            ("_reqid".to_string(), reqid.to_string()),
            ("rt".to_string(), "c".to_string()),
            ("source-path".to_string(), "/app".to_string()),
        ];
        if let Some(build_label) = session.build_label.as_ref() {
            params.push(("bl".to_string(), build_label.clone()));
        }
        if let Some(session_id) = session.session_id.as_ref() {
            params.push(("f.sid".to_string(), session_id.clone()));
        }
        let payload_body = serde_json::to_string(&json!([[keys]]))?;
        let payload = json!([[["ESY5D", payload_body, null, "generic"]]]).to_string();
        let form = vec![
            ("at", session.access_token.as_str()),
            ("f.req", payload.as_str()),
        ];
        let response = self
            .http
            .post(ENDPOINT_BATCH_EXEC)
            .query(&params)
            .header(
                CONTENT_TYPE,
                "application/x-www-form-urlencoded;charset=utf-8",
            )
            .header("X-Same-Domain", "1")
            .header(
                "x-goog-ext-525001261-jspb",
                "[1,null,null,null,null,null,null,null,[4]]",
            )
            .header("x-goog-ext-73010989-jspb", "[0]")
            .header("x-goog-ext-73010990-jspb", "[0]")
            .header("Cookie", session.cookie_header.clone())
            .form(&form)
            .send()
            .await?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!(
                "Gemini settings RPC failed with status {}",
                response.status()
            ))
        }
    }

    async fn load_attachment(
        &self,
        attachment: &InputAttachment,
    ) -> anyhow::Result<(String, String, Vec<u8>)> {
        match &attachment.source {
            AttachmentSource::Data(data) => Ok((
                attachment.filename.clone(),
                attachment
                    .content_type
                    .clone()
                    .unwrap_or_else(|| guess_content_type(&attachment.filename).to_string()),
                data.clone(),
            )),
            AttachmentSource::Url(url) => {
                let response = self.http.get(url).send().await?;
                if !response.status().is_success() {
                    return Err(anyhow!(
                        "attachment download failed with status {}",
                        response.status()
                    ));
                }
                let content_type = response
                    .headers()
                    .get(CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.split(';').next().unwrap_or(s).trim().to_string())
                    .unwrap_or_else(|| guess_content_type(&attachment.filename).to_string());
                let data = response.bytes().await?.to_vec();
                Ok((attachment.filename.clone(), content_type, data))
            }
        }
    }

    pub async fn save_images(
        &self,
        images: &[GeminiImage],
        dir: &str,
    ) -> anyhow::Result<Vec<SavedImage>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        self.ensure_session().await?;
        let session = self
            .session
            .lock()
            .await
            .clone()
            .context("Gemini session is not initialized")?;
        tokio::fs::create_dir_all(dir).await?;
        let mut saved = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for image in images {
            let Some((data, content_type)) = self.download_image(&session, image).await? else {
                continue;
            };
            let mut hasher = Sha256::new();
            hasher.update(&data);
            let sha256 = format!("{:x}", hasher.finalize());
            if !seen.insert(sha256.clone()) {
                continue;
            }
            let ext = ext_from_content_type(&content_type).unwrap_or("png");
            let filename = format!("img_{}.{}", Uuid::new_v4().simple(), ext);
            let path = Path::new(dir).join(&filename);
            tokio::fs::write(path, data).await?;
            saved.push(SavedImage { filename, sha256 });
        }
        Ok(saved)
    }

    async fn download_image(
        &self,
        session: &SessionState,
        image: &GeminiImage,
    ) -> anyhow::Result<Option<(Vec<u8>, String)>> {
        let mut urls = Vec::new();
        if image.generated {
            if let (Some(cid), Some(rid), Some(rcid), Some(image_id)) = (
                image.cid.as_deref(),
                image.rid.as_deref(),
                image.rcid.as_deref(),
                image.image_id.as_deref(),
            ) {
                if let Ok(Some(full_size)) = self
                    .get_full_size_image(session, cid, rid, rcid, image_id)
                    .await
                {
                    if let Ok(Some(url)) =
                        self.resolve_generated_image_url(session, &full_size).await
                    {
                        urls.push(url);
                    }
                    urls.push(full_size);
                }
            }
        }
        urls.push(image_url_with_size(&image.url, image.generated, "s2048-rj"));
        urls.push(image_url_with_size(&image.url, image.generated, "s1024-rj"));
        urls.push(image.url.clone());

        for url in dedupe_strings(urls) {
            let response = self
                .http
                .get(&url)
                .header("Cookie", session.cookie_header.clone())
                .header(REFERER, "https://gemini.google.com/")
                .send()
                .await?;
            if !response.status().is_success() {
                tracing::warn!(status = %response.status(), "Gemini image download failed");
                continue;
            }
            let content_type = response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.split(';').next().unwrap_or(s).trim().to_string())
                .unwrap_or_else(|| "image/png".to_string());
            let data = response.bytes().await?.to_vec();
            if data.is_empty() {
                continue;
            }
            return Ok(Some((data, content_type)));
        }
        Ok(None)
    }

    async fn resolve_generated_image_url(
        &self,
        session: &SessionState,
        full_size_url: &str,
    ) -> anyhow::Result<Option<String>> {
        let req_url = format!(
            "{}=d-I?alr=yes",
            full_size_url.trim_end_matches("=d-I?alr=yes")
        );
        let response = self
            .http
            .get(&req_url)
            .header("Cookie", session.cookie_header.clone())
            .header(REFERER, "https://gemini.google.com/")
            .send()
            .await?;
        if !response.status().is_success() {
            return Ok(None);
        }
        let next_url = response.text().await?.trim().to_string();
        if next_url.is_empty() {
            return Ok(None);
        }
        let response = self
            .http
            .get(&next_url)
            .header("Cookie", session.cookie_header.clone())
            .header(REFERER, "https://gemini.google.com/")
            .send()
            .await?;
        if !response.status().is_success() {
            return Ok(Some(next_url));
        }
        let final_url = response.text().await?.trim().to_string();
        if final_url.is_empty() {
            Ok(Some(next_url))
        } else {
            Ok(Some(final_url))
        }
    }

    async fn get_full_size_image(
        &self,
        session: &SessionState,
        cid: &str,
        rid: &str,
        rcid: &str,
        image_id: &str,
    ) -> anyhow::Result<Option<String>> {
        let payload = json!([
            [null, null, null, [null, null, null, null, null, ""]],
            [image_id, 0],
            null,
            [19, ""],
            null,
            null,
            null,
            null,
            null,
            "",
        ]);
        let payload = json!([payload, [rid, rcid, cid, null, ""], 1, 0, 1]);
        let raw = self
            .batch_execute(session, "c8o8Fe", &serde_json::to_string(&payload)?)
            .await?;
        for frame in parse_frames(&raw) {
            let Some(body_str) = get_path(&frame, &[2]).and_then(Value::as_str) else {
                continue;
            };
            let Ok(body) = serde_json::from_str::<Value>(body_str) else {
                continue;
            };
            if let Some(url) = get_path(&body, &[0]).and_then(Value::as_str) {
                if !url.trim().is_empty() {
                    return Ok(Some(url.to_string()));
                }
            }
        }
        Ok(None)
    }

    async fn batch_execute(
        &self,
        session: &SessionState,
        rpcid: &str,
        payload: &str,
    ) -> anyhow::Result<String> {
        let reqid = self.reqid.fetch_add(100_000, Ordering::Relaxed);
        let language = if session.language.is_empty() {
            "en"
        } else {
            &session.language
        };
        let mut params = vec![
            ("rpcids".to_string(), rpcid.to_string()),
            ("hl".to_string(), language.to_string()),
            ("_reqid".to_string(), reqid.to_string()),
            ("rt".to_string(), "c".to_string()),
            ("source-path".to_string(), "/app".to_string()),
        ];
        if let Some(build_label) = session.build_label.as_ref() {
            params.push(("bl".to_string(), build_label.clone()));
        }
        if let Some(session_id) = session.session_id.as_ref() {
            params.push(("f.sid".to_string(), session_id.clone()));
        }
        let rpc_payload = json!([[[rpcid, payload, null, "generic"]]]).to_string();
        let form = vec![
            ("at", session.access_token.as_str()),
            ("f.req", rpc_payload.as_str()),
        ];
        let response = self
            .http
            .post(ENDPOINT_BATCH_EXEC)
            .query(&params)
            .header(
                CONTENT_TYPE,
                "application/x-www-form-urlencoded;charset=utf-8",
            )
            .header("X-Same-Domain", "1")
            .header(
                "x-goog-ext-525001261-jspb",
                "[1,null,null,null,null,null,null,null,[4]]",
            )
            .header("x-goog-ext-73010989-jspb", "[0]")
            .header("x-goog-ext-73010990-jspb", "[0]")
            .header("Cookie", session.cookie_header.clone())
            .form(&form)
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "Gemini batch RPC {rpcid} failed with status {}",
                response.status()
            ));
        }
        Ok(response.text().await?)
    }

    async fn ensure_session(&self) -> anyhow::Result<()> {
        {
            let guard = self.session.lock().await;
            if let Some(session) = guard.as_ref() {
                if session.created_at.elapsed() < self.refresh_interval {
                    return Ok(());
                }
            }
        }

        let mut cookie_header = self.cookie_header();
        if let Ok(response) = self
            .http
            .get(ENDPOINT_GOOGLE)
            .header("Cookie", &cookie_header)
            .send()
            .await
        {
            merge_set_cookie(&mut cookie_header, response.headers());
        }

        let response = self
            .http
            .get(ENDPOINT_INIT)
            .header("Cookie", cookie_header.clone())
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(anyhow!(
                "Gemini init failed with status {}",
                response.status()
            ));
        }
        merge_set_cookie(&mut cookie_header, response.headers());
        let body = response.text().await?;
        let build_label = capture(&BUILD_RE, &body);
        let session_id = capture(&SID_RE, &body);
        let push_id = capture(&PUSH_RE, &body);
        let language = capture(&LANG_RE, &body).unwrap_or_else(|| "en".to_string());
        let access_token = capture(&TOKEN_RE, &body).unwrap_or_default();
        if access_token.is_empty() && build_label.is_none() && session_id.is_none() {
            return Err(anyhow!("Gemini init markers not found"));
        }

        let session = SessionState {
            access_token,
            build_label,
            session_id,
            push_id,
            language,
            cookie_header,
            created_at: Instant::now(),
        };
        *self.session.lock().await = Some(session.clone());
        if let Err(error) = self.send_bard_settings(&session).await {
            tracing::debug!(?error, "Gemini settings warmup failed");
        }
        if let Err(error) = self.send_bard_activity(&session).await {
            tracing::debug!(?error, "Gemini activity warmup failed");
        }
        Ok(())
    }

    async fn clear_session(&self) {
        *self.session.lock().await = None;
    }

    async fn reset_and_refresh_session(&self) -> anyhow::Result<()> {
        self.clear_session().await;
        self.ensure_session().await
    }

    async fn session_status(&self) -> Value {
        let session = self.session.lock().await;
        let (cookie_header, cookie_source) = self.cookie_header_with_source();
        json!({
            "id": self.id(),
            "session_cached": session.is_some(),
            "session_age_seconds": session.as_ref().map(|s| s.created_at.elapsed().as_secs()),
            "cookie_source": cookie_source,
            "cookie_len": cookie_header.len(),
            "cookie_sha256": short_fingerprint(&cookie_header),
            "cookie_has_full_header": cookie_header.contains("SAPISID=") || cookie_header.contains("__Secure-3PSID="),
            "cookie_has_1psid": cookie_header.contains("__Secure-1PSID="),
            "cookie_has_1psidts": cookie_header.contains("__Secure-1PSIDTS="),
            "cookie_has_1psidcc": cookie_header.contains("__Secure-1PSIDCC="),
        })
    }

    fn cookie_header(&self) -> String {
        self.cookie_header_with_source().0
    }

    fn cookie_header_with_source(&self) -> (String, String) {
        if let Some(value) = env::var("GEMINI_COOKIE_HEADER")
            .ok()
            .and_then(|s| normalise_cookie_value(&s))
        {
            return (value, "env:GEMINI_COOKIE_HEADER".to_string());
        }
        if let Some((value, source)) = env::var("GEMINI_COOKIE_FILE").ok().and_then(|path| {
            read_cookie_file(&path).map(|value| (value, format!("env-file:{path}")))
        }) {
            return (value, source);
        }
        if let Some((value, source)) = self.client_config.cookie_file.as_deref().and_then(|path| {
            read_cookie_file(path).map(|value| (value, format!("client-file:{path}")))
        }) {
            return (value, source);
        }
        if let Some(value) = self
            .client_config
            .cookie_header
            .as_deref()
            .and_then(normalise_cookie_value)
        {
            return (value, "config:cookie_header".to_string());
        }
        let mut parts = vec![
            format!("__Secure-1PSID={}", self.client_config.secure_1psid),
            format!("__Secure-1PSIDTS={}", self.client_config.secure_1psidts),
        ];
        if let Some(value) = self
            .client_config
            .secure_1psidcc
            .as_deref()
            .filter(|s| !s.is_empty())
        {
            parts.push(format!("__Secure-1PSIDCC={value}"));
        }
        (parts.join("; "), "config:minimal_cookies".to_string())
    }
}

pub fn extract_cookie_header_from_text(text: &str) -> Option<String> {
    let normalised = text.replace("^\"", "\"").replace('^', "");
    if let Some(capture) = COOKIE_HEADER_RE.captures(&normalised) {
        if let Some(value) = normalise_cookie_value(capture.get(1)?.as_str()) {
            return Some(value);
        }
    }
    if let Some(capture) = CURL_COOKIE_RE.captures(&normalised) {
        if let Some(value) = normalise_cookie_value(capture.get(1)?.as_str()) {
            return Some(value);
        }
    }
    fallback_cookie_header(&normalised)
}

pub fn extract_web_tool_nonce_from_text(text: &str) -> Option<String> {
    let normalised = text.replace('^', "");
    if let Some(value) = WEB_TOOL_NONCE_RE.find(&normalised) {
        return Some(value.as_str().to_string());
    }
    let decoded = percent_decode_lossy(&normalised);
    WEB_TOOL_NONCE_RE
        .find(&decoded)
        .map(|m| m.as_str().to_string())
}

pub fn short_fingerprint(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string()
}

fn read_cookie_file(path: &str) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    extract_cookie_header_from_text(&raw).or_else(|| normalise_cookie_value(&raw))
}

fn normalise_cookie_value(value: &str) -> Option<String> {
    let mut value = value
        .replace("\\\"", "\"")
        .replace("^\"", "\"")
        .replace('^', "")
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string();
    if value.to_ascii_lowercase().starts_with("cookie:") {
        value = value[7..].trim().to_string();
    }
    if value.contains("__Secure-1PSID=") && value.contains("__Secure-1PSIDTS=") {
        Some(value)
    } else {
        None
    }
}

fn fallback_cookie_header(text: &str) -> Option<String> {
    let idx = text.find("__Secure-1PSID=")?;
    let start = text[..idx]
        .rfind(|c| matches!(c, '"' | '\n' | '\r'))
        .map(|i| i + 1)
        .unwrap_or(0);
    let end = text[idx..]
        .find(|c| matches!(c, '"' | '\n' | '\r'))
        .map(|i| idx + i)
        .unwrap_or(text.len());
    normalise_cookie_value(&text[start..end])
}

fn percent_decode_lossy(input: &str) -> String {
    let mut out = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn build_stream_generate_inner(
    prompt: &str,
    language: &str,
    file_data: Value,
    uuid: &str,
    image_mode: bool,
    temporary_chat: bool,
) -> Vec<Value> {
    let mut inner = vec![Value::Null; if image_mode { 81 } else { 69 }];
    inner[0] = json!([prompt, 0, null, file_data, null, null, 0]);
    inner[1] = json!([language]);
    inner[2] = json!(["", "", "", null, null, null, null, null, null, ""]);
    inner[6] = json!([1]);
    inner[7] = json!(1);
    inner[10] = json!(1);
    inner[11] = json!(0);
    inner[17] = json!([[0]]);
    inner[18] = json!(0);
    inner[27] = json!(1);
    inner[30] = json!([4]);
    inner[41] = json!([1]);
    if temporary_chat {
        inner[45] = json!(1);
    }
    if image_mode {
        inner[3] = json!(web_tool_nonce());
        inner[4] = json!(Uuid::new_v4().simple().to_string());
        inner[49] = json!(14);
        inner[67] = json!(0);
        inner[79] = json!(1);
        inner[80] = json!(2);
    }
    inner[53] = json!(0);
    inner[59] = json!(uuid);
    inner[61] = json!([]);
    inner[68] = json!(2);
    inner
}

fn capture(regex: &Regex, text: &str) -> Option<String> {
    regex
        .captures(text)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

fn builtin_models() -> Vec<GeminiModel> {
    vec![
        model("gemini-3.5-flash", "56fdd199312815e2", "gemini-web"),
        model("gemini-3.1-pro", "e6fa609c3fa255c0", "gemini-web"),
        model("gemini-3.1-flash-lite", "8c46e95b1a07cecc", "gemini-web"),
        model("gemini-3-flash", "56fdd199312815e2", "gemini-web"),
        model("gemini-3-pro", "e6fa609c3fa255c0", "gemini-web"),
    ]
}

fn model(name: &str, model_id: &str, owned_by: &str) -> GeminiModel {
    let mut header = HashMap::new();
    header.insert(
        "x-goog-ext-525001261-jspb".to_string(),
        format!(r#"[1,null,null,null,"{model_id}",null,null,0,[4],null,null,1]"#),
    );
    add_web_required_header_map(&mut header);
    GeminiModel {
        model_name: name.to_string(),
        model_header: header,
        owned_by: owned_by.to_string(),
    }
}

fn add_web_required_header_map(header: &mut HashMap<String, String>) {
    header
        .entry("x-goog-ext-73010989-jspb".to_string())
        .or_insert_with(|| "[0]".to_string());
    header
        .entry("x-goog-ext-73010990-jspb".to_string())
        .or_insert_with(|| "[0]".to_string());
}

fn add_web_required_headers(headers: &mut HeaderMap) {
    headers
        .entry("x-goog-ext-73010989-jspb")
        .or_insert(HeaderValue::from_static("[0]"));
    headers
        .entry("x-goog-ext-73010990-jspb")
        .or_insert(HeaderValue::from_static("[0]"));
}

fn add_image_tool_browser_headers(headers: &mut HeaderMap) {
    // Gemini Web image tools are more sensitive to browser fingerprint headers
    // than normal chat. These mirror the captured StreamGenerate request shape.
    let values = [
        ("accept", "*/*"),
        ("accept-language", "zh-CN,zh;q=0.9"),
        (
            "user-agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
        ),
        ("x-browser-channel", "stable"),
        (
            "x-browser-copyright",
            "Copyright 2026 Google LLC. All Rights Reserved.",
        ),
        ("x-browser-validation", "puPtlXuojC+VILE1bgaJ40YGt+E="),
        ("x-browser-year", "2026"),
        ("sec-fetch-dest", "empty"),
        ("sec-fetch-mode", "cors"),
        ("sec-fetch-site", "same-origin"),
        ("dnt", "1"),
        ("priority", "u=1, i"),
        (
            "sec-ch-ua",
            "\"Chromium\";v=\"148\", \"Google Chrome\";v=\"148\", \"Not/A)Brand\";v=\"99\"",
        ),
        ("sec-ch-ua-arch", "\"x86\""),
        ("sec-ch-ua-bitness", "\"64\""),
        ("sec-ch-ua-form-factors", "\"Desktop\""),
        ("sec-ch-ua-full-version", "\"148.0.7778.168\""),
        (
            "sec-ch-ua-full-version-list",
            "\"Chromium\";v=\"148.0.7778.168\", \"Google Chrome\";v=\"148.0.7778.168\", \"Not/A)Brand\";v=\"99.0.0.0\"",
        ),
        ("sec-ch-ua-mobile", "?0"),
        ("sec-ch-ua-model", "\"\""),
        ("sec-ch-ua-platform", "\"Windows\""),
        ("sec-ch-ua-platform-version", "\"10.0.0\""),
        ("sec-ch-ua-wow64", "?0"),
    ];
    for (name, value) in values {
        if let Ok(header_value) = HeaderValue::from_str(value) {
            headers.insert(name, header_value);
        }
    }
}

fn image_mode_model_header(model: &GeminiModel, request_uuid: &str) -> Option<String> {
    let raw = model.model_header.get("x-goog-ext-525001261-jspb")?;
    let parsed = serde_json::from_str::<Value>(raw).ok()?;
    let model_id = get_path(&parsed, &[4]).and_then(Value::as_str)?;
    Some(format!(
        r#"[1,null,null,null,"{model_id}",null,null,0,[4,5,6,8],null,null,2,null,null,1,2,"{request_uuid}"]"#
    ))
}

fn web_tool_nonce() -> String {
    if let Some(value) = configured_web_tool_nonce() {
        return value;
    }
    let mut nonce = String::from("!");
    while nonce.len() < 2584 {
        nonce.push_str(&general_purpose::URL_SAFE_NO_PAD.encode(Uuid::new_v4().as_bytes()));
    }
    nonce.truncate(2584);
    nonce
}

fn configured_web_tool_nonce() -> Option<String> {
    std::env::var("GEMINI_WEB_TOOL_NONCE")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() > 1000)
        .or_else(|| {
            std::env::var("GEMINI_WEB_TOOL_NONCE_FILE")
                .ok()
                .and_then(|path| std::fs::read_to_string(path).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| s.len() > 1000)
        })
}

fn image_url_with_size(url: &str, generated: bool, size: &str) -> String {
    if !generated {
        return url.to_string();
    }
    if url.contains("=s1024-rj") {
        url.replace("=s1024-rj", &format!("={size}"))
    } else if url.contains("=s2048-rj") {
        url.replace("=s2048-rj", &format!("={size}"))
    } else {
        format!("{url}={size}")
    }
}

fn dedupe_strings(values: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for value in values {
        if value.trim().is_empty() {
            continue;
        }
        if seen.insert(value.clone()) {
            out.push(value);
        }
    }
    out
}

async fn read_response_text_with_progress(
    response: reqwest::Response,
    progress: Option<mpsc::UnboundedSender<String>>,
    client_id: &str,
    model_name: &str,
    request_started: Instant,
) -> anyhow::Result<String> {
    let body_started = Instant::now();
    let Some(progress) = progress else {
        let raw = response.text().await?;
        tracing::info!(
            client = client_id,
            model = model_name,
            body_ms = body_started.elapsed().as_millis(),
            total_ms = request_started.elapsed().as_millis(),
            response_bytes = raw.len(),
            "Gemini generate response body read"
        );
        return Ok(raw);
    };

    let mut raw_bytes = Vec::new();
    let mut emitted = String::new();
    let mut upstream = response.bytes_stream();
    let mut parse_pos = 0usize;
    let mut frames_seen = 0usize;
    let mut first_complete_frame_ms: Option<u128> = None;
    let mut first_complete_frame_kind: Option<&'static str> = None;
    let mut first_upstream_chunk_ms: Option<u128> = None;
    let mut first_text_delta_ms: Option<u128> = None;
    while let Some(chunk) = upstream.next().await {
        let chunk = chunk?;
        if first_upstream_chunk_ms.is_none() {
            let elapsed = request_started.elapsed().as_millis();
            first_upstream_chunk_ms = Some(elapsed);
            tracing::info!(
                client = client_id,
                model = model_name,
                first_upstream_chunk_ms = elapsed,
                "Gemini generate first upstream chunk"
            );
        }
        raw_bytes.extend_from_slice(&chunk);
        let raw = String::from_utf8_lossy(&raw_bytes);
        for frame in parse_frames_from(&raw, &mut parse_pos) {
            frames_seen += 1;
            let frame_kind = classify_response_frame(&frame);
            if first_complete_frame_ms.is_none() {
                let elapsed = request_started.elapsed().as_millis();
                first_complete_frame_ms = Some(elapsed);
                first_complete_frame_kind = Some(frame_kind);
                tracing::info!(
                    client = client_id,
                    model = model_name,
                    first_complete_frame_ms = elapsed,
                    frame_kind,
                    "Gemini generate first complete frame"
                );
            }
            if let Some(text) = extract_partial_text_from_frame(&frame) {
                if send_stream_delta(&progress, &mut emitted, &text)
                    && first_text_delta_ms.is_none()
                {
                    let elapsed = request_started.elapsed().as_millis();
                    first_text_delta_ms = Some(elapsed);
                    let first_text_gap_ms = first_upstream_chunk_ms
                        .map(|first_chunk| elapsed.saturating_sub(first_chunk))
                        .unwrap_or(elapsed);
                    let first_frame_gap_ms = first_complete_frame_ms
                        .map(|first_frame| elapsed.saturating_sub(first_frame));
                    tracing::info!(
                        client = client_id,
                        model = model_name,
                        first_text_delta_ms = elapsed,
                        first_upstream_chunk_ms = first_upstream_chunk_ms,
                        first_complete_frame_ms = first_complete_frame_ms,
                        first_text_gap_ms,
                        first_frame_gap_ms = first_frame_gap_ms,
                        frames_before_text = frames_seen,
                        streamed_chars = emitted.chars().count(),
                        "Gemini generate first text delta"
                    );
                }
            }
        }
    }
    let raw = String::from_utf8_lossy(&raw_bytes).to_string();
    let first_text_gap_ms = match (first_upstream_chunk_ms, first_text_delta_ms) {
        (Some(first_chunk), Some(first_text)) => Some(first_text.saturating_sub(first_chunk)),
        _ => None,
    };
    tracing::info!(
        client = client_id,
        model = model_name,
        body_ms = body_started.elapsed().as_millis(),
        total_ms = request_started.elapsed().as_millis(),
        first_upstream_chunk_ms = first_upstream_chunk_ms,
        first_complete_frame_ms = first_complete_frame_ms,
        first_complete_frame_kind = first_complete_frame_kind.unwrap_or("none"),
        first_text_delta_ms = first_text_delta_ms,
        first_text_gap_ms = first_text_gap_ms,
        frames_seen,
        response_bytes = raw.len(),
        streamed_chars = emitted.chars().count(),
        "Gemini generate response body streamed"
    );
    Ok(raw)
}

fn send_stream_delta(
    progress: &mpsc::UnboundedSender<String>,
    emitted: &mut String,
    text: &str,
) -> bool {
    if is_effectively_empty_delta(text) || starts_like_tool_block(text) {
        return false;
    }
    if text.starts_with(emitted.as_str()) {
        let delta = &text[emitted.len()..];
        if !delta.is_empty() {
            let _ = progress.send(delta.to_string());
            *emitted = text.to_string();
            return true;
        }
    } else if emitted.is_empty() {
        let _ = progress.send(text.to_string());
        *emitted = text.to_string();
        return true;
    }
    false
}

fn is_effectively_empty_delta(text: &str) -> bool {
    text.trim_matches(|ch: char| {
        ch.is_whitespace()
            || matches!(
                ch,
                '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{2060}' | '\u{feff}'
            )
    })
    .is_empty()
}

fn starts_like_tool_block(text: &str) -> bool {
    let compact: String = text
        .trim_start()
        .trim_start_matches('\\')
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .take(24)
        .collect::<String>()
        .to_ascii_lowercase();
    !compact.is_empty()
        && ("[toolcalls]".starts_with(&compact) || compact.starts_with("[toolcalls]"))
}

fn extract_partial_text_from_frame(frame: &Value) -> Option<String> {
    let inner = get_path(frame, &[2]).and_then(Value::as_str)?;
    let parsed_inner = serde_json::from_str::<Value>(inner).ok()?;
    let candidates = get_path(&parsed_inner, &[4]).and_then(Value::as_array)?;
    candidates.iter().filter_map(candidate_text).last()
}

fn candidate_text(candidate: &Value) -> Option<String> {
    get_path(candidate, &[1, 0])
        .or_else(|| get_path(candidate, &[22, 0]))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToString::to_string)
}

fn classify_response_frame(frame: &Value) -> &'static str {
    let Some(inner) = get_path(frame, &[2]).and_then(Value::as_str) else {
        return "no_inner";
    };
    let Ok(parsed_inner) = serde_json::from_str::<Value>(inner) else {
        return "invalid_inner";
    };
    let Some(candidates) = get_path(&parsed_inner, &[4]).and_then(Value::as_array) else {
        return "no_candidates";
    };
    if candidates
        .iter()
        .any(|candidate| candidate_text(candidate).is_some())
    {
        "text"
    } else {
        "candidate_no_text"
    }
}

fn extract_output_from_response(raw: &str) -> anyhow::Result<GeminiOutput> {
    if let Some(code) = capture(&BARD_ERROR_RE, raw) {
        return Err(anyhow!("Gemini API error code: {code}"));
    }
    let frames = parse_frames(raw);
    for frame in &frames {
        if let Some(code) = get_path(frame, &[5, 2, 0, 1, 0]).and_then(Value::as_i64) {
            return Err(anyhow!("Gemini API error code: {code}"));
        }
    }
    let mut output = GeminiOutput::default();
    for frame in frames {
        let Some(inner) = get_path(&frame, &[2]).and_then(Value::as_str) else {
            continue;
        };
        let Ok(parsed_inner) = serde_json::from_str::<Value>(inner) else {
            continue;
        };
        let cid = get_path(&parsed_inner, &[1, 0])
            .and_then(Value::as_str)
            .map(ToString::to_string);
        let rid = get_path(&parsed_inner, &[1, 1])
            .and_then(Value::as_str)
            .map(ToString::to_string);
        let Some(candidates) = get_path(&parsed_inner, &[4]).and_then(Value::as_array) else {
            continue;
        };
        for candidate in candidates {
            let rcid = get_path(candidate, &[0])
                .and_then(Value::as_str)
                .map(ToString::to_string);
            if let Some(text) = get_path(candidate, &[1, 0])
                .or_else(|| get_path(candidate, &[22, 0]))
                .and_then(Value::as_str)
            {
                if !text.trim().is_empty() {
                    output.text = text.to_string();
                }
            }
            for (idx, web_image) in get_path(candidate, &[12, 1])
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                if let Some(url) = get_path(web_image, &[0, 0, 0]).and_then(Value::as_str) {
                    output.images.push(GeminiImage {
                        url: url.to_string(),
                        title: format!("[Image {}]", idx + 1),
                        alt: get_path(web_image, &[0, 4])
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        generated: false,
                        cid: None,
                        rid: None,
                        rcid: None,
                        image_id: None,
                    });
                }
            }
            for (idx, gen_image) in generated_image_values(candidate).into_iter().enumerate() {
                if let Some(url) = get_path(gen_image, &[0, 3, 3]).and_then(Value::as_str) {
                    let image_id = get_path(gen_image, &[1, 0])
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                        .or_else(|| {
                            Some(format!(
                                "http://googleusercontent.com/image_generation_content/{idx}"
                            ))
                        });
                    output.images.push(GeminiImage {
                        url: url.to_string(),
                        title: format!("[Generated Image {}]", idx + 1),
                        alt: get_path(gen_image, &[0, 3, 2])
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        generated: true,
                        cid: cid.clone(),
                        rid: rid.clone(),
                        rcid: rcid.clone(),
                        image_id,
                    });
                }
            }
        }
    }
    if output.text.is_empty() && output.images.is_empty() {
        Err(anyhow!("empty Gemini response at {}", Utc::now()))
    } else {
        Ok(output)
    }
}

fn generated_image_values(candidate: &Value) -> Vec<&Value> {
    let mut out = Vec::new();
    if let Some(items) = candidate.pointer("/12/7/0").and_then(Value::as_array) {
        out.extend(items);
    }
    if let Some(items) = candidate.pointer("/12/0/8/0").and_then(Value::as_array) {
        out.extend(items);
    }
    out
}

fn parse_frames(raw: &str) -> Vec<Value> {
    let mut pos = 0usize;
    parse_frames_from(raw, &mut pos)
}

fn parse_frames_from(raw: &str, pos: &mut usize) -> Vec<Value> {
    let mut cursor = (*pos).min(raw.len());
    if cursor == 0 && raw.starts_with(")]}'") {
        cursor = 4;
    }
    let mut frames = Vec::new();

    loop {
        while cursor < raw.len() {
            let Some(ch) = raw[cursor..].chars().next() else {
                break;
            };
            if !ch.is_whitespace() {
                break;
            }
            cursor += ch.len_utf8();
            *pos = cursor;
        }
        if cursor >= raw.len() {
            break;
        }

        let frame_start = cursor;
        let digit_start = cursor;
        while cursor < raw.len() {
            let Some(ch) = raw[cursor..].chars().next() else {
                break;
            };
            if !ch.is_ascii_digit() {
                break;
            }
            cursor += ch.len_utf8();
        }
        if digit_start == cursor {
            cursor = frame_start;
            break;
        }

        let Ok(length) = raw[digit_start..cursor].parse::<usize>() else {
            cursor = frame_start;
            break;
        };
        let payload_start = cursor;
        let mut units = 0usize;
        while cursor < raw.len() && units < length {
            let Some(ch) = raw[cursor..].chars().next() else {
                break;
            };
            units += ch.len_utf16();
            cursor += ch.len_utf8();
        }
        if units < length {
            cursor = frame_start;
            break;
        }

        let chunk = raw[payload_start..cursor].trim();
        *pos = cursor;
        if chunk.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(chunk) {
            if let Value::Array(values) = value {
                frames.extend(values);
            } else {
                frames.push(value);
            }
        }
    }
    *pos = (*pos).max(cursor);
    frames
}

fn get_path<'a>(value: &'a Value, path: &[usize]) -> Option<&'a Value> {
    let mut current = value;
    for index in path {
        current = current.as_array()?.get(*index)?;
    }
    Some(current)
}

fn merge_set_cookie(cookie_header: &mut String, headers: &HeaderMap) {
    for value in headers.get_all(SET_COOKIE).iter() {
        let Ok(raw) = value.to_str() else {
            continue;
        };
        let Some(pair) = raw
            .split(';')
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            continue;
        };
        let Some((name, _)) = pair.split_once('=') else {
            continue;
        };
        let exists = cookie_header
            .split(';')
            .filter_map(|part| part.trim().split_once('='))
            .any(|(existing, _)| existing == name);
        if !exists {
            cookie_header.push_str("; ");
            cookie_header.push_str(pair);
        }
    }
}

fn extract_runtime_models(raw: &str) -> anyhow::Result<Vec<GeminiModel>> {
    let frames = parse_frames(raw);
    let mut out = Vec::new();
    for frame in frames {
        let Some(body_str) = get_path(&frame, &[2]).and_then(Value::as_str) else {
            continue;
        };
        let Ok(body) = serde_json::from_str::<Value>(body_str) else {
            continue;
        };
        let status = get_path(&body, &[14])
            .and_then(Value::as_i64)
            .unwrap_or(1000);
        let tier_flags = get_path(&body, &[16])
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let capability_flags = get_path(&body, &[17])
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let (capacity, capacity_field) = compute_capacity(&tier_flags, &capability_flags);
        let Some(models) = get_path(&body, &[15]).and_then(Value::as_array) else {
            continue;
        };
        for model_data in models {
            let Some(model_id) = get_path(model_data, &[0])
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            else {
                continue;
            };
            let display_name = get_path(model_data, &[1])
                .and_then(Value::as_str)
                .unwrap_or("");
            let model_name = runtime_model_name(model_id, display_name);
            if model_name.is_empty() {
                continue;
            }
            let mut header = HashMap::new();
            header.insert(
                "x-goog-ext-525001261-jspb".to_string(),
                model_header(model_id, capacity, capacity_field),
            );
            add_web_required_header_map(&mut header);
            let owned_by = if status == 1000 {
                "gemini-web-runtime"
            } else {
                "gemini-web-runtime-limited"
            };
            out.push(GeminiModel {
                model_name,
                model_header: header,
                owned_by: owned_by.to_string(),
            });
        }
    }
    Ok(out)
}

fn runtime_model_name(model_id: &str, display_name: &str) -> String {
    match model_id {
        "e6fa609c3fa255c0" => "gemini-3.1-pro".to_string(),
        "56fdd199312815e2" => "gemini-3.5-flash".to_string(),
        "8c46e95b1a07cecc" => "gemini-3.1-flash-lite".to_string(),
        "fbb127bbb056c959" => "gemini-3-flash".to_string(),
        "9d8ca3786ebdfbea" => "gemini-3-pro".to_string(),
        _ => {
            let slug = display_name
                .to_lowercase()
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '.' {
                        c
                    } else {
                        '-'
                    }
                })
                .collect::<String>()
                .trim_matches('-')
                .to_string();
            if slug.is_empty() {
                String::new()
            } else if slug.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                format!("gemini-{slug}")
            } else {
                format!("gemini-web-{slug}")
            }
        }
    }
}

fn compute_capacity(tier_flags: &[Value], capability_flags: &[Value]) -> (i64, i64) {
    let has_tier = |needle| tier_flags.iter().any(|v| v.as_i64() == Some(needle));
    let has_cap = |needle| capability_flags.iter().any(|v| v.as_i64() == Some(needle));
    if has_tier(21) {
        return (1, 13);
    }
    if has_tier(22) {
        return (2, 13);
    }
    if has_cap(115) {
        return (4, 12);
    }
    if has_tier(16) || has_cap(106) {
        return (3, 12);
    }
    if has_tier(8) || (!has_cap(106) && has_cap(19)) {
        return (2, 12);
    }
    (1, 12)
}

fn model_header(model_id: &str, capacity: i64, capacity_field: i64) -> String {
    if capacity_field == 13 {
        format!(r#"[1,null,null,null,"{model_id}",null,null,0,[4],null,null,null,{capacity}]"#)
    } else {
        format!(r#"[1,null,null,null,"{model_id}",null,null,0,[4],null,null,{capacity}]"#)
    }
}

fn guess_content_type(filename: &str) -> &'static str {
    match filename
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        "txt" | "md" => "text/plain",
        "json" => "application/json",
        _ => "application/octet-stream",
    }
}

fn ext_from_content_type(content_type: &str) -> Option<&'static str> {
    match content_type {
        "image/png" => Some("png"),
        "image/jpeg" => Some("jpg"),
        "image/gif" => Some("gif"),
        "image/webp" => Some("webp"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_stream_generate_inner, contains_cjk, extract_partial_text_from_frame,
        parse_frames_from, request_language,
    };
    use serde_json::{Value, json};

    fn encoded_frame(frame: Value) -> String {
        let payload = json!([frame]).to_string();
        format!("{}{}", payload.encode_utf16().count(), payload)
    }

    #[test]
    fn incremental_frame_parser_waits_for_complete_frame() {
        let inner = json!([null, ["cid", "rid"], null, null, [["rcid", ["??"]]]]).to_string();
        let raw = encoded_frame(json!([null, null, inner]));
        let split_at = raw.len() - 3;
        let mut pos = 0usize;

        assert!(parse_frames_from(&raw[..split_at], &mut pos).is_empty());
        assert_eq!(pos, 0);

        let frames = parse_frames_from(&raw, &mut pos);
        assert_eq!(frames.len(), 1);
        assert_eq!(
            extract_partial_text_from_frame(&frames[0]).as_deref(),
            Some("??")
        );
        assert_eq!(pos, raw.len());
    }

    #[test]
    fn incremental_frame_parser_only_returns_new_frames() {
        let first_inner = json!([null, ["cid", "rid"], null, null, [["rcid", ["?"]]]]).to_string();
        let second_inner =
            json!([null, ["cid", "rid"], null, null, [["rcid", ["??"]]]]).to_string();
        let raw = format!(
            "{}{}",
            encoded_frame(json!([null, null, first_inner])),
            encoded_frame(json!([null, null, second_inner]))
        );
        let mut pos = 0usize;

        let frames = parse_frames_from(&raw, &mut pos);
        assert_eq!(frames.len(), 2);
        assert_eq!(
            extract_partial_text_from_frame(&frames[0]).as_deref(),
            Some("?")
        );
        assert_eq!(
            extract_partial_text_from_frame(&frames[1]).as_deref(),
            Some("??")
        );

        assert!(parse_frames_from(&raw, &mut pos).is_empty());
    }

    #[test]
    fn stream_generate_inner_keeps_normal_chat_saved_by_default() {
        let inner = build_stream_generate_inner("hello", "en", Value::Null, "UUID", false, false);

        assert_eq!(inner.len(), 69);
        assert_eq!(inner[0], json!(["hello", 0, null, null, null, null, 0]));
        assert_eq!(inner[45], Value::Null);
    }

    #[test]
    fn stream_generate_inner_sets_temporary_chat_flag() {
        let inner = build_stream_generate_inner("hello", "en", Value::Null, "UUID", false, true);

        assert_eq!(inner.len(), 69);
        assert_eq!(inner[45], json!(1));
    }

    #[test]
    fn stream_generate_inner_sets_temporary_chat_flag_in_image_mode() {
        let inner =
            build_stream_generate_inner("draw a cat", "zh-CN", Value::Null, "UUID", true, true);

        assert_eq!(inner.len(), 81);
        assert_eq!(inner[17], json!([[0]]));
        assert_eq!(inner[45], json!(1));
    }

    #[test]
    fn detects_cjk_prompt_text() {
        assert!(contains_cjk(
            "\u{7528}\u{4e00}\u{53e5}\u{4e2d}\u{6587}\u{56de}\u{590d}"
        ));
        assert!(contains_cjk("\u{4e2d}\u{6587} mixed English"));
        assert!(!contains_cjk("plain ascii prompt"));
    }

    #[test]
    fn cjk_prompts_force_chinese_locale() {
        assert_eq!(
            request_language("en", "\u{4eca}\u{5929}\u{5468}\u{51e0}\u{ff1f}", false),
            "zh-CN"
        );
        assert_eq!(
            request_language("", "\u{4eca}\u{5929}\u{5468}\u{51e0}\u{ff1f}", false),
            "zh-CN"
        );
        assert_eq!(request_language("en", "draw image", true), "zh-CN");
        assert_eq!(request_language("", "plain ascii", false), "en");
        assert_eq!(request_language("fr", "plain ascii", false), "fr");
    }
}
