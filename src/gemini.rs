use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
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

#[derive(Debug)]
struct SessionState {
    access_token: String,
    build_label: Option<String>,
    session_id: Option<String>,
    language: String,
    cookie_header: String,
}

pub struct GeminiClient {
    http: Client,
    client_config: GeminiClientConfig,
    models: Vec<GeminiModel>,
    reqid: AtomicU64,
    session: Mutex<Option<SessionState>>,
}

impl GeminiClient {
    pub fn new(
        client_config: GeminiClientConfig,
        configured_models: Vec<GeminiModelConfig>,
        timeout_seconds: u64,
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
            models,
            reqid: AtomicU64::new(10_000),
            session: Mutex::new(None),
        })
    }

    pub fn models(&self) -> Vec<GeminiModel> {
        self.models.clone()
    }

    pub async fn generate(&self, model_name: &str, prompt: &str) -> anyhow::Result<String> {
        if prompt.trim().is_empty() {
            return Err(anyhow!("prompt cannot be empty"));
        }
        let model = self
            .models
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
        if self.session.lock().await.is_some() {
            return Ok(());
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
