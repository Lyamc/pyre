use pyre_whisper_core::{
    compose, OutputFormat, PipelineStep, SpeechTask, TaskOptions, TranscriptSegment,
    WhisburnResult,
};

struct DoubleStep;

impl PipelineStep<i32, i32> for DoubleStep {
    fn name(&self) -> &'static str {
        "double"
    }

    fn apply(&self, input: i32) -> WhisburnResult<i32> {
        Ok(input * 2)
    }
}

struct AddOneStep;

impl PipelineStep<i32, i32> for AddOneStep {
    fn name(&self) -> &'static str {
        "add_one"
    }

    fn apply(&self, input: i32) -> WhisburnResult<i32> {
        Ok(input + 1)
    }
}

#[test]
fn speech_task_parses_aliases() {
    assert_eq!(SpeechTask::parse("asr"), Some(SpeechTask::Transcribe));
    assert_eq!(SpeechTask::parse("stt"), Some(SpeechTask::Stt));
    assert_eq!(SpeechTask::parse("tts"), Some(SpeechTask::Tts));
    assert_eq!(SpeechTask::parse("nope"), None);
}

#[test]
fn output_format_parses_aliases() {
    assert_eq!(OutputFormat::parse("text"), Some(OutputFormat::Txt));
    assert_eq!(OutputFormat::parse("srt"), Some(OutputFormat::Srt));
    assert_eq!(OutputFormat::parse("json"), Some(OutputFormat::Json));
}

#[test]
fn pipeline_composition_is_functional() {
    let composed = compose(DoubleStep, AddOneStep);
    assert_eq!(composed.apply(3).unwrap(), 7);
}

#[test]
fn task_options_defaults_to_small_model_friendly_values() {
    let opts = TaskOptions::default();
    assert_eq!(opts.beam_size, 5);
    assert_eq!(opts.scout_model.as_deref(), Some("tiny_en"));
}

#[test]
fn transcript_segment_builder() {
    let seg = TranscriptSegment::new(0.0, 1.0, "hello").with_speaker("A");
    assert_eq!(seg.speaker.as_deref(), Some("A"));
}