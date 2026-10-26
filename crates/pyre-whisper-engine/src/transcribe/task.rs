#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeTask {
    #[default]
    Transcribe,
    Translate,
}

impl DecodeTask {
    pub fn from_task_name(name: &str) -> Self {
        match name {
            "translate" | "translation" => Self::Translate,
            _ => Self::Transcribe,
        }
    }
}