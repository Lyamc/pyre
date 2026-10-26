use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageCode(pub String);

impl LanguageCode {
    pub fn new(code: impl Into<String>) -> Self {
        Self(code.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn english() -> Self {
        Self("en".to_string())
    }
}

impl Default for LanguageCode {
    fn default() -> Self {
        Self::english()
    }
}