use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::WhisburnResult;

/// Which models to load into GPU memory at startup.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PreloadModels {
    /// Do not load any model until the first transcribe request.
    #[default]
    None,
    /// Load every `burn_ready` model from the registry.
    All,
    /// Load only the named models.
    List(Vec<String>),
}

impl Serialize for PreloadModels {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::None => serializer.serialize_str("none"),
            Self::All => serializer.serialize_str("all"),
            Self::List(names) => names.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for PreloadModels {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, Visitor};
        use std::fmt;

        struct PreloadVisitor;

        impl<'de> Visitor<'de> for PreloadVisitor {
            type Value = PreloadModels;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("\"none\", \"all\", or a list of model names")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                if value.eq_ignore_ascii_case("none") {
                    Ok(PreloadModels::None)
                } else if value.eq_ignore_ascii_case("all") {
                    Ok(PreloadModels::All)
                } else if value.is_empty() {
                    Ok(PreloadModels::All)
                } else {
                    Ok(PreloadModels::List(vec![value.to_string()]))
                }
            }

            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut names = Vec::new();
                while let Some(name) = seq.next_element::<String>()? {
                    names.push(name);
                }
                if names.is_empty() {
                    Ok(PreloadModels::All)
                } else {
                    Ok(PreloadModels::List(names))
                }
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(PreloadModels::All)
            }

            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(PreloadModels::None)
            }

            fn visit_some<D: serde::Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> Result<Self::Value, D::Error> {
                deserializer.deserialize_any(self)
            }
        }

        deserializer.deserialize_any(PreloadVisitor)
    }
}

impl PreloadModels {
    pub fn from_cli_values(values: Option<Vec<String>>) -> Self {
        match values {
            None => Self::None,
            Some(v) if v.is_empty() => Self::All,
            Some(v) if v.len() == 1 && (v[0].is_empty() || v[0].eq_ignore_ascii_case("all")) => {
                Self::All
            }
            Some(v) => Self::List(v),
        }
    }

    pub fn resolve_names<'a>(&self, burn_ready: impl Iterator<Item = &'a str>) -> Vec<String> {
        match self {
            Self::None => Vec::new(),
            Self::All => burn_ready.map(str::to_string).collect(),
            Self::List(names) => names.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default = "default_model")]
    pub default_model: String,
    #[serde(default = "default_server_port")]
    pub server_port: u16,
    #[serde(default)]
    pub preload_models: PreloadModels,
    #[serde(default = "default_server_url")]
    pub server_url: String,
}

fn default_model() -> String {
    "tiny_en".to_string()
}

fn default_server_port() -> u16 {
    8787
}

fn default_server_url() -> String {
    "http://127.0.0.1:8787".to_string()
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            default_model: default_model(),
            server_port: default_server_port(),
            preload_models: PreloadModels::None,
            server_url: default_server_url(),
        }
    }
}

pub fn settings_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("WHISBURN_CONFIG") {
        return Some(PathBuf::from(dir));
    }
    dirs::config_dir().map(|d| d.join("whisburn"))
}

pub fn settings_path() -> Option<PathBuf> {
    settings_dir().map(|d| d.join("settings.toml"))
}

pub fn load_settings() -> Settings {
    let Some(path) = settings_path() else {
        return Settings::default();
    };
    load_settings_from(&path).unwrap_or_default()
}

pub fn load_settings_from(path: &Path) -> WhisburnResult<Settings> {
    if !path.exists() {
        return Ok(Settings::default());
    }
    let raw = fs::read_to_string(path)?;
    let settings: Settings = toml::from_str(&raw).map_err(|e| {
        crate::WhisburnError::InvalidRequest(format!("invalid settings at {}: {e}", path.display()))
    })?;
    Ok(settings)
}

pub fn save_settings(settings: &Settings) -> WhisburnResult<PathBuf> {
    let dir = settings_dir().ok_or_else(|| {
        crate::WhisburnError::InvalidRequest("could not resolve config directory".into())
    })?;
    fs::create_dir_all(&dir)?;
    let path = dir.join("settings.toml");
    let raw = toml::to_string_pretty(settings).map_err(|e| {
        crate::WhisburnError::InvalidRequest(format!("failed to serialize settings: {e}"))
    })?;
    fs::write(&path, raw)?;
    Ok(path)
}