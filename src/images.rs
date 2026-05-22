use anyhow::{Context, anyhow};
use base64::{Engine as _, engine::general_purpose};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::ImageGenerationConfig;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub output_format: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ImageGenerationResponse {
    pub created: i64,
    pub data: Vec<ImageData>,
}

#[derive(Debug, Serialize)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub b64_json: String,
    pub mime_type: String,
    pub revised_prompt: Option<String>,
}

pub async fn generate_images(
    http: &Client,
    config: &ImageGenerationConfig,
    request: &ImageGenerationRequest,
) -> anyhow::Result<Vec<GeneratedImage>> {
    if request.prompt.trim().is_empty() {
        return Err(anyhow!("prompt is required"));
    }
    if !config.is_enabled() {
        return Err(anyhow!(
            "image generation is disabled; configure image_generation.backend"
        ));
    }
    let api_key = config
        .resolved_api_key()
        .context("image generation api key is not configured")?;
    let model = choose_model(config, request);
    match config.backend.as_str() {
        "gemini_api" | "gemini" | "nano_banana" => {
            generate_with_gemini(http, config, request, &api_key, &model).await
        }
        "imagen_api" | "imagen" => {
            generate_with_imagen(http, config, request, &api_key, &model).await
        }
        other => Err(anyhow!("unsupported image generation backend: {other}")),
    }
}

fn choose_model(config: &ImageGenerationConfig, request: &ImageGenerationRequest) -> String {
    let requested = request.model.as_deref().unwrap_or("").trim();
    if requested.starts_with("gemini-") || requested.starts_with("imagen-") {
        requested.to_string()
    } else {
        config.model.clone()
    }
}

async fn generate_with_gemini(
    http: &Client,
    config: &ImageGenerationConfig,
    request: &ImageGenerationRequest,
    api_key: &str,
    model: &str,
) -> anyhow::Result<Vec<GeneratedImage>> {
    let count = image_count(request.n);
    let url = format!(
        "{}/v1beta/models/{}:generateContent",
        config.gemini_api_base_url.trim_end_matches('/'),
        model
    );
    let mut images = Vec::new();
    for _ in 0..count {
        let mut generation_config = json!({
            "responseModalities": ["TEXT", "IMAGE"],
        });
        if let Some(aspect_ratio) = aspect_ratio(request.size.as_deref()) {
            generation_config["imageConfig"] = json!({
                "aspectRatio": aspect_ratio,
                "imageSize": image_size(request.quality.as_deref()),
            });
        }
        let body = json!({
            "contents": [{
                "parts": [{"text": request.prompt}]
            }],
            "generationConfig": generation_config,
        });
        let response = http
            .post(&url)
            .header("x-goog-api-key", api_key)
            .json(&body)
            .send()
            .await?;
        let status = response.status();
        let value: Value = response.json().await.unwrap_or_else(|_| json!({}));
        if !status.is_success() {
            return Err(anyhow!(
                "Gemini image generation failed with status {status}: {}",
                value
            ));
        }
        images.extend(extract_gemini_images(&value));
    }
    if images.is_empty() {
        return Err(anyhow!("Gemini image generation returned no images"));
    }
    Ok(images)
}

async fn generate_with_imagen(
    http: &Client,
    config: &ImageGenerationConfig,
    request: &ImageGenerationRequest,
    api_key: &str,
    model: &str,
) -> anyhow::Result<Vec<GeneratedImage>> {
    let url = format!(
        "{}/v1beta/models/{}:predict",
        config.imagen_api_base_url.trim_end_matches('/'),
        model
    );
    let mut parameters = json!({
        "sampleCount": image_count(request.n),
    });
    if let Some(aspect_ratio) = aspect_ratio(request.size.as_deref()) {
        parameters["aspectRatio"] = json!(aspect_ratio);
    }
    let body = json!({
        "instances": [{"prompt": request.prompt}],
        "parameters": parameters,
    });
    let response = http
        .post(&url)
        .header("x-goog-api-key", api_key)
        .json(&body)
        .send()
        .await?;
    let status = response.status();
    let value: Value = response.json().await.unwrap_or_else(|_| json!({}));
    if !status.is_success() {
        return Err(anyhow!(
            "Imagen generation failed with status {status}: {}",
            value
        ));
    }
    let images = extract_imagen_images(&value);
    if images.is_empty() {
        return Err(anyhow!("Imagen returned no images"));
    }
    Ok(images)
}

fn extract_gemini_images(value: &Value) -> Vec<GeneratedImage> {
    value
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|candidate| {
            candidate
                .pointer("/content/parts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter_map(|part| {
            let inline = part.get("inlineData").or_else(|| part.get("inline_data"))?;
            let data = inline
                .get("data")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())?;
            let mime_type = inline
                .get("mimeType")
                .or_else(|| inline.get("mime_type"))
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            Some(GeneratedImage {
                b64_json: data.to_string(),
                mime_type: mime_type.to_string(),
                revised_prompt: None,
            })
        })
        .collect()
}

fn extract_imagen_images(value: &Value) -> Vec<GeneratedImage> {
    value
        .get("predictions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|prediction| {
            let data = prediction
                .get("bytesBase64Encoded")
                .or_else(|| prediction.get("bytes_base64_encoded"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())?;
            let mime_type = prediction
                .get("mimeType")
                .or_else(|| prediction.get("mime_type"))
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            let revised_prompt = prediction
                .get("prompt")
                .or_else(|| prediction.get("revisedPrompt"))
                .and_then(Value::as_str)
                .map(ToString::to_string);
            Some(GeneratedImage {
                b64_json: data.to_string(),
                mime_type: mime_type.to_string(),
                revised_prompt,
            })
        })
        .collect()
}

pub fn image_ext(mime_type: &str) -> &'static str {
    match mime_type {
        "image/jpeg" | "image/jpg" => "jpg",
        "image/webp" => "webp",
        _ => "png",
    }
}

pub fn decode_image_b64(image: &GeneratedImage) -> anyhow::Result<Vec<u8>> {
    Ok(general_purpose::STANDARD.decode(&image.b64_json)?)
}

fn image_count(n: Option<u32>) -> u32 {
    n.unwrap_or(1).clamp(1, 4)
}

fn aspect_ratio(size: Option<&str>) -> Option<&'static str> {
    match size.unwrap_or("").trim() {
        "" | "auto" => None,
        "1024x1024" | "1:1" => Some("1:1"),
        "1024x1536" | "2:3" => Some("2:3"),
        "1536x1024" | "3:2" => Some("3:2"),
        "1024x1792" | "9:16" => Some("9:16"),
        "1792x1024" | "16:9" => Some("16:9"),
        "768x1024" | "3:4" => Some("3:4"),
        "1024x768" | "4:3" => Some("4:3"),
        "1344x768" | "21:9" => Some("21:9"),
        _ => None,
    }
}

fn image_size(quality: Option<&str>) -> &'static str {
    match quality.unwrap_or("").trim() {
        "high" => "2K",
        _ => "1K",
    }
}
