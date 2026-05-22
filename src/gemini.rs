use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, anyhow};
use chrono::Utc;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::{
    Client,
    header::{CONTENT_TYPE, HeaderMap, HeaderValue, ORIGIN, REFERER, SET_COOKIE, USER_AGENT},
};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config::{GeminiClientConfig, GeminiModelConfig};

const ENDPOINT_GOOGLE: &str = "https://www.google.com";
const ENDPOINT_INIT: &str = "https://gemini.google.com/app";
const ENDPOINT_GENERATE: &str = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate";
const ENDPOINT_BATCH_EXEC: &str = "https://gemini.google.com/_/BardChatUi/data/batchexecute";

static TOKEN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"\"SNlM0e\":\s*\"(.*?)\""#).unwrap());
static BUILD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"\"cfb2h\":\s*\"(.*?)\""#).unwrap());
static SID_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"\"FdrFJe\":\s*\"(.*?)\""#).unwrap());
static LANG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"\"TuX5cc\":\s*\"(.*?)\""#).unwrap());

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
    language: String,
    cookie_header: String,
    created_at: Instant,
}

pub struct GeminiClient {
    http: Client,
    client_config: GeminiClientConfig,
    models: Mutex<Vec<GeminiModel>>,
    reqid: AtomicU64,
    session: Mutex<Option<SessionState>>,
    refresh_interval: Duration,
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

    pub async fn generate(&self, model_name: &str, prompt: &str) -> anyhow::Result<String> {
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
        let session_guard = self.session.lock().await;
        let session = session_guard
            .as_ref()
            .context("Gemini session is not initialized")?;

        let reqid = self.reqid.fetch_add(100_000, Ordering::Relaxed);
        let uuid = Uuid::new_v4().to_string().to_uppercase();
        let language = if session.language.is_empty() {
            "en"
        } else {
            &session.language
        };

        let mut inner = vec![Value::Null; 69];
        inner[0] = json!([prompt, 0, null, null, null, null, 0]);
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
        extract_text_from_response(&raw).context("no Gemini text candidate found")
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
        let access_token = capture(&TOKEN_RE, &body).context("SNlM0e access token not found")?;
        let build_label = capture(&BUILD_RE, &body);
        let session_id = capture(&SID_RE, &body);
        let language = capture(&LANG_RE, &body).unwrap_or_else(|| "en".to_string());

        *self.session.lock().await = Some(SessionState {
            access_token,
            build_label,
            session_id,
            language,
            cookie_header,
            created_at: Instant::now(),
        });
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
    GeminiModel {
        model_name: name.to_string(),
        model_header: header,
        owned_by: owned_by.to_string(),
    }
}

fn extract_text_from_response(raw: &str) -> anyhow::Result<String> {
    let frames = parse_frames(raw);
    let mut final_text = String::new();
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
            if let Some(text) = get_path(candidate, &[1, 0]).and_then(Value::as_str) {
                if !text.trim().is_empty() {
                    final_text = text.to_string();
                }
            }
        }
    }
    if final_text.is_empty() {
        Err(anyhow!("empty Gemini response at {}", Utc::now()))
    } else {
        Ok(final_text)
    }
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
