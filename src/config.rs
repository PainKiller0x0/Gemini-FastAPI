use std::{collections::HashMap, env, fs, net::SocketAddr};

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    pub gemini: GeminiConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub image_generation: ImageGenerationConfig,
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        let path = env::var("CONFIG_PATH").unwrap_or_else(|_| "config/config.yaml".to_string());
        let content = fs::read_to_string(&path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub api_key: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            api_key: None,
        }
    }
}

impl ServerConfig {
    pub fn addr(&self) -> anyhow::Result<SocketAddr> {
        Ok(format!("{}:{}", self.host, self.port).parse()?)
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8000
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeminiConfig {
    pub clients: Vec<GeminiClientConfig>,
    #[serde(default)]
    pub models: Vec<GeminiModelConfig>,
    #[serde(default = "default_model_strategy")]
    pub model_strategy: String,
    #[serde(default = "default_chat_mode")]
    pub chat_mode: String,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default = "default_refresh_interval")]
    pub refresh_interval: u64,
    #[serde(default)]
    pub warm_generate: WarmGenerateConfig,
}

impl GeminiConfig {
    pub fn temporary_chat(&self) -> bool {
        self.chat_mode.eq_ignore_ascii_case("temporary")
    }
}

fn default_model_strategy() -> String {
    "append".to_string()
}

fn default_chat_mode() -> String {
    "normal".to_string()
}

fn default_timeout() -> u64 {
    600
}

fn default_refresh_interval() -> u64 {
    600
}

#[derive(Debug, Clone, Deserialize)]
pub struct WarmGenerateConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_warm_generate_interval")]
    pub interval: u64,
    #[serde(default = "default_warm_generate_initial_delay")]
    pub initial_delay: u64,
    #[serde(default = "default_warm_generate_model")]
    pub model: String,
    #[serde(default = "default_warm_generate_prompt")]
    pub prompt: String,
    #[serde(default)]
    pub active_periods: Vec<String>,
}

impl Default for WarmGenerateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: default_warm_generate_interval(),
            initial_delay: default_warm_generate_initial_delay(),
            model: default_warm_generate_model(),
            prompt: default_warm_generate_prompt(),
            active_periods: Vec::new(),
        }
    }
}

fn default_warm_generate_interval() -> u64 {
    300
}

fn default_warm_generate_initial_delay() -> u64 {
    20
}

fn default_warm_generate_model() -> String {
    "gemini-3.5-flash".to_string()
}

fn default_warm_generate_prompt() -> String {
    "只回复一个字：好".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeminiClientConfig {
    pub id: String,
    pub secure_1psid: String,
    pub secure_1psidts: String,
    #[serde(default)]
    pub secure_1psidcc: Option<String>,
    #[serde(default)]
    pub cookie_header: Option<String>,
    #[serde(default)]
    pub cookie_file: Option<String>,
    #[serde(default)]
    pub proxy: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeminiModelConfig {
    pub model_name: String,
    pub model_header: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "default_storage_path")]
    pub path: String,
    #[serde(default = "default_images_path")]
    pub images_path: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: default_storage_path(),
            images_path: default_images_path(),
        }
    }
}

fn default_storage_path() -> String {
    "data/lmdb".to_string()
}

fn default_images_path() -> String {
    "data/images".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageGenerationConfig {
    #[serde(default = "default_image_backend")]
    pub backend: String,
    #[serde(default = "default_image_model")]
    pub model: String,
    #[serde(default = "default_web_image_model")]
    pub web_model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_image_api_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_gemini_api_base_url")]
    pub gemini_api_base_url: String,
    #[serde(default = "default_imagen_api_base_url")]
    pub imagen_api_base_url: String,
    #[serde(default)]
    pub public_base_url: Option<String>,
}

impl Default for ImageGenerationConfig {
    fn default() -> Self {
        Self {
            backend: default_image_backend(),
            model: default_image_model(),
            web_model: default_web_image_model(),
            api_key: None,
            api_key_env: default_image_api_key_env(),
            gemini_api_base_url: default_gemini_api_base_url(),
            imagen_api_base_url: default_imagen_api_base_url(),
            public_base_url: None,
        }
    }
}

impl ImageGenerationConfig {
    pub fn resolved_api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .filter(|key| !key.trim().is_empty())
            .or_else(|| {
                env::var(&self.api_key_env)
                    .ok()
                    .filter(|key| !key.trim().is_empty())
            })
    }

    pub fn is_enabled(&self) -> bool {
        !matches!(self.backend.as_str(), "" | "disabled" | "none")
    }
}

fn default_image_backend() -> String {
    "disabled".to_string()
}

fn default_image_model() -> String {
    "gemini-3.1-flash-image-preview".to_string()
}

fn default_web_image_model() -> String {
    "gemini-3.5-flash".to_string()
}

fn default_image_api_key_env() -> String {
    "GEMINI_API_KEY".to_string()
}

fn default_gemini_api_base_url() -> String {
    "https://generativelanguage.googleapis.com".to_string()
}

fn default_imagen_api_base_url() -> String {
    "https://generativelanguage.googleapis.com".to_string()
}
