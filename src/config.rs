use std::{collections::HashMap, env, fs, net::SocketAddr};

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    pub gemini: GeminiConfig,
    #[serde(default)]
    pub storage: StorageConfig,
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
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default = "default_refresh_interval")]
    pub refresh_interval: u64,
}

fn default_model_strategy() -> String {
    "append".to_string()
}

fn default_timeout() -> u64 {
    600
}

fn default_refresh_interval() -> u64 {
    600
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeminiClientConfig {
    pub id: String,
    pub secure_1psid: String,
    pub secure_1psidts: String,
    #[serde(default)]
    pub secure_1psidcc: Option<String>,
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
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: default_storage_path(),
        }
    }
}

fn default_storage_path() -> String {
    "data/lmdb".to_string()
}
