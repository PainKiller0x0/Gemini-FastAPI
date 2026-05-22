use std::{fs::OpenOptions, io::Write, path::PathBuf, time::Instant};

use chrono::Utc;
use serde::Serialize;

#[derive(Clone)]
pub struct HistoryStore {
    path: PathBuf,
}

#[derive(Serialize)]
pub struct HistoryRecord<'a> {
    pub ts: String,
    pub kind: &'a str,
    pub model: &'a str,
    pub prompt_chars: usize,
    pub output_chars: usize,
    pub latency_ms: u128,
    pub ok: bool,
    pub error: Option<&'a str>,
}

impl HistoryStore {
    pub fn new(storage_path: impl Into<PathBuf>) -> Self {
        let mut path = storage_path.into();
        if path.extension().is_none() {
            path.push("rust-history.jsonl");
        }
        Self { path }
    }

    pub fn append(&self, record: &HistoryRecord<'_>) {
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(record) else {
            return;
        };
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = writeln!(file, "{line}");
        }
    }
}

pub fn started() -> Instant {
    Instant::now()
}

pub fn timestamp() -> String {
    Utc::now().to_rfc3339()
}
