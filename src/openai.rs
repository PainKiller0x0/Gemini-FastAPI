use base64::{Engine as _, engine::general_purpose};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const TOOL_WRAP_HINT: &str = r#"### SYSTEM: TOOL CALLING PROTOCOL (MANDATORY) ###
If tool execution is required, output ONLY this protocol block. Do not add prose.

[ToolCalls]
[Call:tool_name]
[CallParameter:parameter_name]
```
value
```
[/CallParameter]
[/Call]
[/ToolCalls]

Every argument value must be wrapped in a markdown code fence. Close all tags exactly."#;

static TOOL_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)\\?\[\s*ToolCalls\s*\\?\]\s*(.*?)\s*\\?\[\s*\\?/\s*ToolCalls\s*\\?\]")
        .unwrap()
});
static TOOL_CALL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)\\?\[\s*Call\s*\\?:\s*(?P<name>[^\]]+?)\s*\\?\]\s*(?P<body>.*?)\s*\\?\[\s*\\?/\s*Call\s*\\?\]").unwrap()
});
static TAGGED_ARG_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)\\?\[\s*CallParameter\s*\\?:\s*(?P<name>[^\]]+?)\s*\\?\]\s*(?P<body>.*?)\s*\\?\[\s*\\?/\s*CallParameter\s*\\?\]").unwrap()
});
static CHATML_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)\\?\s*<\s*\\?\|\s*im\s*\\?_(?:start|end)\s*\\?\|\s*>\s*(?:system|user|assistant|tool)?\s*").unwrap()
});

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelData>,
}

#[derive(Debug, Serialize)]
pub struct ModelData {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: Value,
    pub finish_reason: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ResponseCreateRequest {
    pub model: String,
    pub input: Value,
    #[serde(default)]
    pub instructions: Option<Value>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ProcessedOutput {
    pub visible_text: Option<String>,
    pub storage_text: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
}

#[derive(Debug, Clone)]
pub enum AttachmentSource {
    Url(String),
    Data(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct InputAttachment {
    pub filename: String,
    pub content_type: Option<String>,
    pub source: AttachmentSource,
}

#[derive(Debug, Clone)]
pub struct GeminiInput {
    pub prompt: String,
    pub attachments: Vec<InputAttachment>,
}

pub fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    let mut tool_id_to_name = std::collections::HashMap::new();
    for message in messages {
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                tool_id_to_name.insert(call.id.clone(), call.function.name.clone());
            }
        }
    }

    let mut parts = Vec::new();
    for message in messages {
        let role = normalize_role(&message.role);
        let mut text = content_to_text(message.content.as_ref());

        if role == "tool" {
            let name = message
                .name
                .clone()
                .or_else(|| {
                    message
                        .tool_call_id
                        .as_ref()
                        .and_then(|id| tool_id_to_name.get(id).cloned())
                })
                .unwrap_or_else(|| "unknown".to_string());
            let body = text.unwrap_or_default();
            text = Some(format!(
                "[ToolResults]\n[Result:{name}]\n[ToolResult]\n{body}\n[/ToolResult]\n[/Result]\n[/ToolResults]"
            ));
        }

        if let Some(reasoning) = message
            .reasoning_content
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            let current = text.unwrap_or_default();
            text = Some(
                format!("{current}\n<reasoning>{}</reasoning>", reasoning.trim())
                    .trim()
                    .to_string(),
            );
        }

        if let Some(calls) = &message.tool_calls {
            let mut block = String::from("[ToolCalls]\n");
            for call in calls {
                block.push_str(&format!("[Call:{}]\n", call.function.name));
                let args = serde_json::from_str::<Value>(&call.function.arguments)
                    .unwrap_or_else(|_| json!({}));
                if let Some(obj) = args.as_object() {
                    for (name, value) in obj {
                        let rendered = value
                            .as_str()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| value.to_string());
                        block.push_str(&format!(
                            "[CallParameter:{name}]\n```\n{rendered}\n```\n[/CallParameter]\n"
                        ));
                    }
                }
                block.push_str("[/Call]\n");
            }
            block.push_str("[/ToolCalls]");
            let current = text.unwrap_or_default();
            text = Some(format!("{current}\n{block}").trim().to_string());
        }

        if let Some(text) = text.filter(|s| !s.trim().is_empty() || role == "tool") {
            parts.push(add_tag(role, text.trim(), false));
        }
    }
    parts.push(add_tag("assistant", "", true));
    parts.join("\n")
}

pub fn messages_to_gemini_input(messages: &[ChatMessage]) -> GeminiInput {
    let mut attachments = Vec::new();
    for message in messages {
        collect_attachments(message.content.as_ref(), &mut attachments);
    }
    let prompt = if attachments.is_empty() {
        messages_to_prompt(messages)
    } else {
        messages_to_plain_prompt(messages)
    };
    GeminiInput {
        prompt,
        attachments,
    }
}

pub fn latest_user_plain_text(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .rev()
        .find(|message| normalize_role(&message.role) == "user")
        .and_then(|message| content_to_plain_text(message.content.as_ref()))
        .unwrap_or_default()
}

fn messages_to_plain_prompt(messages: &[ChatMessage]) -> String {
    let mut system_parts = Vec::new();
    let mut user_parts = Vec::new();
    for message in messages {
        let Some(text) = content_to_plain_text(message.content.as_ref()) else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        match normalize_role(&message.role) {
            "system" => system_parts.push(text.to_string()),
            "user" => user_parts.push(text.to_string()),
            "assistant" => user_parts.push(format!("Assistant: {text}")),
            "tool" => user_parts.push(format!("Tool: {text}")),
            _ => user_parts.push(text.to_string()),
        }
    }
    let mut parts = Vec::new();
    if !system_parts.is_empty() {
        parts.push(system_parts.join("\n\n"));
    }
    if !user_parts.is_empty() {
        parts.push(user_parts.join("\n\n"));
    }
    parts.join("\n\n")
}

fn add_tag(role: &str, content: &str, unclosed: bool) -> String {
    if unclosed {
        format!("<|im_start|>{role}\n{content}")
    } else {
        format!("<|im_start|>{role}\n{content}\n<|im_end|>")
    }
}

fn normalize_role(role: &str) -> &str {
    match role {
        "developer" => "system",
        other => other,
    }
}

fn content_to_plain_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                match item.get("type").and_then(Value::as_str) {
                    Some("text") | Some("input_text") | Some("output_text") => {
                        if let Some(text) = item.get("text").and_then(Value::as_str) {
                            parts.push(text.to_string());
                        }
                    }
                    Some("image_url") | Some("input_image") => {}
                    Some("file") | Some("input_file") => {
                        let filename = item
                            .get("file")
                            .and_then(|f| f.get("filename"))
                            .or_else(|| item.get("filename"))
                            .and_then(Value::as_str)
                            .unwrap_or("attachment");
                        parts.push(format!("[File attachment: {filename}]"));
                    }
                    _ => {
                        if let Some(text) = item.get("text").and_then(Value::as_str) {
                            parts.push(text.to_string());
                        }
                    }
                }
            }
            Some(parts.join("\n"))
        }
        other => Some(other.to_string()),
    }
}

fn content_to_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let mut parts = Vec::new();
            for (index, item) in items.iter().enumerate() {
                match item.get("type").and_then(Value::as_str) {
                    Some("text") | Some("input_text") | Some("output_text") => {
                        if let Some(text) = item.get("text").and_then(Value::as_str) {
                            parts.push(text.to_string());
                        }
                    }
                    Some("image_url") => {
                        let has_image = item
                            .get("image_url")
                            .and_then(|v| v.get("url").or(Some(v)))
                            .and_then(Value::as_str)
                            .map(|s| !s.trim().is_empty())
                            .unwrap_or(false);
                        if has_image {
                            parts.push(format!("[Image attachment: input_image_{}]", index + 1));
                        }
                    }
                    Some("input_image") => {
                        let has_image = item
                            .get("image_url")
                            .and_then(Value::as_str)
                            .map(|s| !s.trim().is_empty())
                            .unwrap_or(false);
                        if has_image {
                            parts.push(format!("[Image attachment: input_image_{}]", index + 1));
                        }
                    }
                    Some("file") | Some("input_file") => {
                        let filename = item
                            .get("file")
                            .and_then(|f| f.get("filename"))
                            .or_else(|| item.get("filename"))
                            .and_then(Value::as_str)
                            .unwrap_or("attachment");
                        if let Some(url) = item
                            .get("file")
                            .and_then(|f| f.get("url"))
                            .or_else(|| item.get("file_url"))
                            .and_then(Value::as_str)
                        {
                            parts.push(format!("[File: {filename}; URL: {url}]"));
                        } else if item.get("file").and_then(|f| f.get("file_data")).is_some()
                            || item.get("file_data").is_some()
                        {
                            parts.push(format!("[File: {filename}; base64 data supplied]"));
                        }
                    }
                    _ => {
                        if let Some(text) = item.get("text").and_then(Value::as_str) {
                            parts.push(text.to_string());
                        }
                    }
                }
            }
            Some(parts.join("\n"))
        }
        other => Some(other.to_string()),
    }
}

fn collect_attachments(value: Option<&Value>, attachments: &mut Vec<InputAttachment>) {
    let Some(Value::Array(items)) = value else {
        return;
    };
    for (index, item) in items.iter().enumerate() {
        match item.get("type").and_then(Value::as_str) {
            Some("image_url") => {
                let url = item
                    .get("image_url")
                    .and_then(|v| v.get("url").or(Some(v)))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if let Some(att) =
                    attachment_from_url_or_data(url, format!("input_image_{}.png", index + 1))
                {
                    attachments.push(att);
                }
            }
            Some("input_image") => {
                let url = item.get("image_url").and_then(Value::as_str).unwrap_or("");
                if let Some(att) =
                    attachment_from_url_or_data(url, format!("input_image_{}.png", index + 1))
                {
                    attachments.push(att);
                }
            }
            Some("file") | Some("input_file") => {
                let file = item.get("file");
                let filename = file
                    .and_then(|f| f.get("filename"))
                    .or_else(|| item.get("filename"))
                    .and_then(Value::as_str)
                    .unwrap_or("attachment.bin")
                    .to_string();
                let data = file
                    .and_then(|f| f.get("file_data"))
                    .or_else(|| item.get("file_data"))
                    .and_then(Value::as_str);
                let url = file
                    .and_then(|f| f.get("url"))
                    .or_else(|| item.get("file_url"))
                    .and_then(Value::as_str);
                if let Some(data) = data {
                    if let Some(att) = attachment_from_url_or_data(data, filename.clone()) {
                        attachments.push(att);
                    }
                } else if let Some(url) = url {
                    if let Some(att) = attachment_from_url_or_data(url, filename.clone()) {
                        attachments.push(att);
                    }
                }
            }
            _ => {}
        }
    }
}

fn attachment_from_url_or_data(value: &str, fallback_filename: String) -> Option<InputAttachment> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    if value.starts_with("http://") || value.starts_with("https://") {
        return Some(InputAttachment {
            filename: fallback_filename,
            content_type: None,
            source: AttachmentSource::Url(value.to_string()),
        });
    }
    if let Some((content_type, encoded)) = parse_data_url(value) {
        let data = general_purpose::STANDARD.decode(encoded).ok()?;
        let filename = filename_with_ext(fallback_filename, content_type);
        return Some(InputAttachment {
            filename,
            content_type: Some(content_type.to_string()),
            source: AttachmentSource::Data(data),
        });
    }
    let data = general_purpose::STANDARD.decode(value).ok()?;
    Some(InputAttachment {
        filename: fallback_filename,
        content_type: None,
        source: AttachmentSource::Data(data),
    })
}

fn parse_data_url(value: &str) -> Option<(&str, &str)> {
    let rest = value.strip_prefix("data:")?;
    let (meta, encoded) = rest.split_once(',')?;
    if !meta.ends_with(";base64") {
        return None;
    }
    Some((meta.trim_end_matches(";base64"), encoded))
}

fn filename_with_ext(filename: String, content_type: &str) -> String {
    if filename.contains('.') {
        return filename;
    }
    let ext = match content_type {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "application/pdf" => "pdf",
        "text/plain" => "txt",
        _ => "bin",
    };
    format!("{filename}.{ext}")
}

pub fn estimate_tokens(text: &str) -> u64 {
    ((text.chars().count() as f64) / 3.0).ceil() as u64
}

pub fn chat_extra_instructions(request: &ChatCompletionRequest) -> String {
    build_extra_instructions(
        request.response_format.as_ref(),
        request.tools.as_ref(),
        request.tool_choice.as_ref(),
    )
}

pub fn response_extra_instructions(request: &ResponseCreateRequest) -> String {
    build_extra_instructions(
        request.response_format.as_ref(),
        request.tools.as_ref(),
        request.tool_choice.as_ref(),
    )
}

fn build_extra_instructions(
    response_format: Option<&Value>,
    tools: Option<&Vec<Value>>,
    tool_choice: Option<&Value>,
) -> String {
    let mut parts = Vec::new();
    if let Some(format) = response_format {
        match format.get("type").and_then(Value::as_str) {
            Some("json_object") => parts
                .push("Return one valid JSON object only. Do not wrap it in Markdown.".to_string()),
            Some("json_schema") => {
                if let Some(schema) = format.get("json_schema").and_then(|v| v.get("schema")) {
                    let name = format
                        .get("json_schema")
                        .and_then(|v| v.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or("response");
                    parts.push(format!(
                        "Return one valid JSON document only. It must conform to schema `{name}`. JSON Schema: {}",
                        serde_json::to_string(schema).unwrap_or_else(|_| "{}".to_string())
                    ));
                }
            }
            _ => {}
        }
    }

    if let Some(tools) = tools.filter(|t| !t.is_empty()) {
        let function_tools: Vec<Value> = tools
            .iter()
            .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
            .cloned()
            .collect();
        if !function_tools.is_empty() {
            parts.push(
                "SYSTEM INTERFACE: You have access to technical tools. Use them when necessary."
                    .to_string(),
            );
            parts.push(serde_json::to_string(&function_tools).unwrap_or_else(|_| "[]".to_string()));
            if let Some(choice) = tool_choice {
                if choice == "none" {
                    parts.push("For this request do not call any tool.".to_string());
                } else if choice == "required" {
                    parts
                        .push("You must call at least one tool before final response.".to_string());
                } else if let Some(name) = choice.pointer("/function/name").and_then(Value::as_str)
                {
                    parts.push(format!("You must call tool `{name}`."));
                }
            }
            parts.push(TOOL_WRAP_HINT.to_string());
        }

        let image_tools: Vec<&Value> = tools
            .iter()
            .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("image_generation"))
            .collect();
        if !image_tools.is_empty()
            || tool_choice
                .and_then(|v| v.get("type"))
                .and_then(Value::as_str)
                == Some("image_generation")
        {
            parts.push("IMAGE GENERATION ENABLED: when image generation is requested, return the real generated image directly and avoid filler text.".to_string());
        }
    }
    parts.join("\n\n")
}

pub fn process_output(raw_text: &str) -> ProcessedOutput {
    let (visible, calls) = extract_tool_calls(raw_text);
    let cleaned = strip_system_hints(raw_text).trim().to_string();
    let visible = visible.trim().to_string();
    let finish_reason = if calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    }
    .to_string();
    ProcessedOutput {
        visible_text: if visible.is_empty() {
            None
        } else {
            Some(visible)
        },
        storage_text: cleaned,
        tool_calls: calls,
        finish_reason,
    }
}

fn extract_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut calls = Vec::new();
    for call_match in TOOL_CALL_RE.captures_iter(text) {
        let name = unescape(
            call_match
                .name("name")
                .map(|m| m.as_str())
                .unwrap_or("")
                .trim(),
        );
        if name.is_empty() {
            continue;
        }
        let body = call_match.name("body").map(|m| m.as_str()).unwrap_or("");
        let mut args = serde_json::Map::new();
        for arg_match in TAGGED_ARG_RE.captures_iter(body) {
            let arg_name = unescape(
                arg_match
                    .name("name")
                    .map(|m| m.as_str())
                    .unwrap_or("")
                    .trim(),
            );
            let arg_body =
                strip_param_fence(arg_match.name("body").map(|m| m.as_str()).unwrap_or(""));
            args.insert(arg_name, Value::String(unescape(&arg_body)));
        }
        let arguments = Value::Object(args).to_string();
        let id = stable_call_id(&name, &arguments, calls.len());
        calls.push(ToolCall {
            id,
            kind: "function".to_string(),
            function: FunctionCall { name, arguments },
        });
    }

    if calls.is_empty() {
        if let Ok(value) = serde_json::from_str::<Value>(text.trim()) {
            if let Some(items) = value.get("tool_calls").and_then(Value::as_array) {
                for item in items {
                    let name = item
                        .get("name")
                        .or_else(|| item.pointer("/function/name"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    if name.is_empty() {
                        continue;
                    }
                    let arguments_value = item
                        .get("arguments")
                        .or_else(|| item.pointer("/function/arguments"))
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let arguments = if arguments_value.is_string() {
                        arguments_value.as_str().unwrap_or("{}").to_string()
                    } else {
                        arguments_value.to_string()
                    };
                    let id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                        .unwrap_or_else(|| stable_call_id(&name, &arguments, calls.len()));
                    calls.push(ToolCall {
                        id,
                        kind: "function".to_string(),
                        function: FunctionCall { name, arguments },
                    });
                }
                if !calls.is_empty() {
                    return (String::new(), calls);
                }
            }
        }
    }

    (strip_system_hints(text), calls)
}

fn strip_system_hints(text: &str) -> String {
    let no_tools = TOOL_BLOCK_RE.replace_all(text, "");
    let no_tags = CHATML_RE.replace_all(&no_tools, "");
    no_tags.trim().to_string()
}

fn strip_param_fence(input: &str) -> String {
    let s = input.trim();
    if !s.starts_with("```") {
        return s.to_string();
    }
    let mut lines: Vec<&str> = s.lines().collect();
    if lines.len() >= 2
        && lines
            .first()
            .is_some_and(|l| l.trim_start().starts_with("```"))
        && lines.last().is_some_and(|l| l.trim() == "```")
    {
        lines.remove(0);
        lines.pop();
        return lines.join("\n").trim().to_string();
    }
    s.trim_matches('`').trim().to_string()
}

fn unescape(input: &str) -> String {
    input
        .replace("\\[", "[")
        .replace("\\]", "]")
        .replace("\\`", "`")
}

fn stable_call_id(name: &str, arguments: &str, index: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{name}:{arguments}:{index}"));
    let digest = hasher.finalize();
    format!("call_{:x}", digest)[..29].to_string()
}

#[allow(dead_code)]
pub fn response_input_to_prompt(input: &Value, instructions: Option<&Value>) -> String {
    response_input_to_gemini_input(input, instructions).prompt
}

pub fn response_input_to_gemini_input(input: &Value, instructions: Option<&Value>) -> GeminiInput {
    let mut messages = Vec::new();
    for instruction in instructions_to_messages(instructions) {
        messages.push(instruction);
    }
    messages.extend(response_items_to_messages(input));
    messages_to_gemini_input(&messages)
}

fn instructions_to_messages(instructions: Option<&Value>) -> Vec<ChatMessage> {
    match instructions {
        Some(Value::String(text)) if !text.trim().is_empty() => vec![ChatMessage {
            role: "system".to_string(),
            content: Some(Value::String(text.clone())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
        Some(Value::Array(items)) => items.iter().flat_map(response_item_to_message).collect(),
        Some(Value::Object(_)) => response_item_to_message(instructions.unwrap()),
        _ => Vec::new(),
    }
}

fn response_items_to_messages(input: &Value) -> Vec<ChatMessage> {
    match input {
        Value::String(text) => vec![ChatMessage {
            role: "user".to_string(),
            content: Some(Value::String(text.clone())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
        Value::Array(items) => items.iter().flat_map(response_item_to_message).collect(),
        Value::Object(_) => response_item_to_message(input),
        _ => Vec::new(),
    }
}

fn response_item_to_message(item: &Value) -> Vec<ChatMessage> {
    let Some(obj) = item.as_object() else {
        return Vec::new();
    };
    let role = obj
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("user")
        .to_string();
    let content = obj.get("content").cloned().unwrap_or(Value::Null);
    let converted = if let Some(parts) = content.as_array() {
        Value::Array(
            parts
                .iter()
                .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                    Some("input_text") | Some("output_text") | Some("reasoning_text") => Some(json!({"type":"text", "text": part.get("text").and_then(Value::as_str).unwrap_or("")})),
                    Some("input_image") => Some(json!({"type":"image_url", "image_url":{"url": part.get("image_url").and_then(Value::as_str).unwrap_or("")}})),
                    Some("input_file") => Some(json!({"type":"file", "file":{"url": part.get("file_url").and_then(Value::as_str), "file_data": part.get("file_data"), "filename": part.get("filename").and_then(Value::as_str)}})),
                    _ => None,
                })
                .collect(),
        )
    } else {
        content
    };
    vec![ChatMessage {
        role,
        content: Some(converted),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }]
}

#[cfg(test)]
mod tests {
    use super::{ChatMessage, latest_user_plain_text, messages_to_gemini_input};
    use serde_json::json;

    #[test]
    fn image_data_url_becomes_attachment_without_prompt_base64() {
        let message = ChatMessage {
            role: "user".to_string(),
            content: Some(json!([
                {"type":"text", "text":"describe this image"},
                {"type":"image_url", "image_url":{"url":"data:image/png;base64,aGVsbG8="}}
            ])),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };

        let input = messages_to_gemini_input(&[message]);

        assert_eq!(input.attachments.len(), 1);
        assert!(input.prompt.contains("describe this image"));
        assert!(!input.prompt.contains("[Image attachment:"));
        assert!(!input.prompt.contains("data:image"));
        assert!(!input.prompt.contains("aGVsbG8="));
    }

    #[test]
    fn latest_user_plain_text_ignores_system_image_generation_words() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some(json!("you can create images when asked")),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some(json!([
                    {"type":"text", "text":"what is in this image?"},
                    {"type":"image_url", "image_url":{"url":"data:image/png;base64,aGVsbG8="}}
                ])),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        assert_eq!(latest_user_plain_text(&messages), "what is in this image?");
    }
}
