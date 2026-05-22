use serde::{Deserialize, Serialize};
use serde_json::Value;

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
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
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
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Serialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: Value,
    pub finish_reason: Option<&'static str>,
}

pub fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .filter_map(|message| {
            let text = content_to_text(message.content.as_ref())?;
            if text.trim().is_empty() {
                return None;
            }
            Some(format!("[{}]\n{}", message.role, text.trim()))
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn content_to_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    parts.push(text.to_string());
                }
            }
            Some(parts.join("\n"))
        }
        other => Some(other.to_string()),
    }
}

pub fn estimate_tokens(text: &str) -> u64 {
    ((text.chars().count() as f64) / 4.0).ceil() as u64
}

#[derive(Debug, Deserialize)]
pub struct ResponseCreateRequest {
    pub model: String,
    pub input: Value,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
}

pub fn chat_extra_instructions(request: &ChatCompletionRequest) -> String {
    let mut parts = Vec::new();
    if let Some(format) = &request.response_format {
        if format.get("type").and_then(Value::as_str) == Some("json_object") {
            parts.push("Return valid JSON only. Do not wrap it in Markdown.".to_string());
        }
    }
    if let Some(tools) = &request.tools {
        if !tools.is_empty() {
            parts.push(format!(
                "Tools are available. If a tool is needed, respond with a JSON object: {{\"tool_calls\":[{{\"name\":\"tool_name\",\"arguments\":{{}}}}]}}. Available tools: {}",
                serde_json::to_string(tools).unwrap_or_else(|_| "[]".to_string())
            ));
        }
    }
    if let Some(choice) = &request.tool_choice {
        if !choice.is_null() {
            parts.push(format!("Tool choice constraint: {}", choice));
        }
    }
    parts.join("\n")
}

pub fn response_extra_instructions(request: &ResponseCreateRequest) -> String {
    let mut parts = Vec::new();
    if let Some(format) = &request.response_format {
        if format.get("type").and_then(Value::as_str) == Some("json_object") {
            parts.push("Return valid JSON only. Do not wrap it in Markdown.".to_string());
        }
    }
    if let Some(tools) = &request.tools {
        if !tools.is_empty() {
            parts.push(format!(
                "Tools are available. If a tool is needed, respond with a JSON object: {{\"tool_calls\":[{{\"name\":\"tool_name\",\"arguments\":{{}}}}]}}. Available tools: {}",
                serde_json::to_string(tools).unwrap_or_else(|_| "[]".to_string())
            ));
        }
    }
    if let Some(choice) = &request.tool_choice {
        if !choice.is_null() {
            parts.push(format!("Tool choice constraint: {}", choice));
        }
    }
    parts.join("\n")
}
