use thiserror::Error;

pub type WhisburnResult<T> = Result<T, WhisburnError>;

#[derive(Debug, Error)]
pub enum WhisburnError {
    #[error("audio error: {0}")]
    Audio(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("unsupported capability: {capability} for model {model}")]
    UnsupportedCapability { model: String, capability: String },

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}