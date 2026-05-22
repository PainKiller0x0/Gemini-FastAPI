use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, anyhow};
use chrono::Utc;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::{
    Client,
    header::{CONTENT_TYPE, HeaderMap, HeaderValue, ORIGIN, REFERER, SET_COOKIE, USER_AGENT},
    multipart,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
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
        let len = self.clients.len();
        let start = self.cursor.fetch_add(1, Ordering::Relaxed) % len;
        let mut last_error = None;
        for offset in 0..len {
            let index = (start + offset) % len;
            let client = &self.clients[index];
            match client
                .generate_output(model_name, prompt, attachments)
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
}

impl GeminiClient {
    pub fn new(
        client_config: GeminiClientConfig,
        configured_models: Vec<GeminiModelConfig>,
        timeout_seconds: u64,
        refresh_interval_seconds: u64,
        append_builtin: bool,
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
        let language = if session.language.is_empty() {
            "en"
        } else {
            &session.language
        };

        let mut inner = vec![Value::Null; 69];
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
        inner[53] = json!(0);
        inner[59] = json!(uuid);
        inner[61] = json!([]);
        inner[68] = json!(2);

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

        let response = self
            .http
            .post(ENDPOINT_GENERATE)
            .query(&params)
            .headers(request_headers)
            .header("Cookie", session.cookie_header.clone())
            .form(&form)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            *self.session.lock().await = None;
            return Err(anyhow!(
                "Gemini generate failed with status {status}: {}",
                body.chars().take(300).collect::<String>()
            ));
        }

        let raw = response.text().await?;
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
            let mut url = image.url.clone();
            if image.generated {
                if url.contains("=s1024-rj") {
                    url = url.replace("=s1024-rj", "=s2048-rj");
                } else if !url.contains("=s2048-rj") {
                    url.push_str("=s2048-rj");
                }
            }
            let response = self
                .http
                .get(&url)
                .header("Cookie", session.cookie_header.clone())
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

    fn cookie_header(&self) -> String {
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
        parts.join("; ")
    }
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
        let Some(candidates) = get_path(&parsed_inner, &[4]).and_then(Value::as_array) else {
            continue;
        };
        for candidate in candidates {
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
                    });
                }
            }
            for (idx, gen_image) in generated_image_values(candidate).into_iter().enumerate() {
                if let Some(url) = get_path(gen_image, &[0, 3, 3]).and_then(Value::as_str) {
                    output.images.push(GeminiImage {
                        url: url.to_string(),
                        title: format!("[Generated Image {}]", idx + 1),
                        alt: get_path(gen_image, &[0, 3, 2])
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        generated: true,
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
    let mut content = raw.to_string();
    if content.starts_with(")]}'") {
        content = content[4..].trim_start().to_string();
    }
    let chars: Vec<char> = content.chars().collect();
    let mut pos = 0usize;
    let mut frames = Vec::new();

    while pos < chars.len() {
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }
        let digit_start = pos;
        while pos < chars.len() && chars[pos].is_ascii_digit() {
            pos += 1;
        }
        if digit_start == pos {
            break;
        }
        let length: usize = chars[digit_start..pos]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(0);
        let start = pos;
        let mut units = 0usize;
        while pos < chars.len() && units < length {
            units += chars[pos].len_utf16();
            pos += 1;
        }
        if units < length {
            break;
        }
        let chunk = chars[start..pos].iter().collect::<String>();
        let chunk = chunk.trim();
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
