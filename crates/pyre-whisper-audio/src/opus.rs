//! Pure-Rust Opus decoding for symphonia 0.5 (whose own registry has no Opus codec), so browser
//! microphone recordings (`audio/webm;codecs=opus`, `audio/ogg;codecs=opus`) decode for STT.
//! Wraps `opus-rs` (BSD-3, RFC 6716, no C), the same decoder Lyamc/Symphonia's opus codec uses.
use symphonia::core::audio::{AsAudioBufferRef, AudioBuffer, AudioBufferRef, Channels, Signal, SignalSpec};
use symphonia::core::codecs::{
    CodecDescriptor, CodecParameters, CodecRegistry, Decoder, DecoderOptions, FinalizeResult, CODEC_TYPE_OPUS,
};
use symphonia::core::errors::{decode_error, unsupported_error, Result};
use symphonia::core::formats::Packet;
use symphonia::core::support_codec;

const RATE: u32 = 48_000;
/// 120 ms at 48 kHz, the longest Opus packet.
const MAX_FRAME: usize = 5760;

pub struct OpusDecoder {
    params: CodecParameters,
    inner: opus_rs::OpusDecoder,
    channels: usize,
    pcm: Vec<f32>,
    buf: AudioBuffer<f32>,
    pre_skip: usize,
}

fn spec(channels: usize) -> SignalSpec {
    SignalSpec::new(RATE, if channels == 2 { Channels::FRONT_LEFT | Channels::FRONT_RIGHT } else { Channels::FRONT_LEFT })
}

/// `OpusHead` (RFC 7845 section 5.1): bytes 10..12 are the little-endian pre-skip.
fn pre_skip(extra: Option<&[u8]>) -> usize {
    match extra {
        Some(d) if d.len() >= 12 && &d[..8] == b"OpusHead" => u16::from_le_bytes([d[10], d[11]]) as usize,
        _ => 0,
    }
}

impl Decoder for OpusDecoder {
    fn try_new(params: &CodecParameters, _opts: &DecoderOptions) -> Result<Self> {
        if params.codec != CODEC_TYPE_OPUS {
            return unsupported_error("opus: invalid codec");
        }
        let channels = params.channels.map(|c| c.count()).unwrap_or(1);
        if !(1..=2).contains(&channels) {
            return unsupported_error("opus: only mono/stereo streams are supported");
        }
        let inner = opus_rs::OpusDecoder::new(RATE as i32, channels).map_err(|_| symphonia::core::errors::Error::Unsupported("opus: decoder init failed"))?;
        let mut p = params.clone();
        p.with_sample_rate(RATE);
        Ok(Self {
            params: p,
            inner,
            channels,
            pcm: vec![0.0; MAX_FRAME * channels],
            buf: AudioBuffer::new(MAX_FRAME as u64, spec(channels)),
            pre_skip: pre_skip(params.extra_data.as_deref()),
        })
    }

    fn supported_codecs() -> &'static [CodecDescriptor] {
        &[support_codec!(CODEC_TYPE_OPUS, "opus", "Opus (opus-rs)")]
    }

    fn reset(&mut self) {
        if let Ok(d) = opus_rs::OpusDecoder::new(RATE as i32, self.channels) {
            self.inner = d;
        }
    }

    fn codec_params(&self) -> &CodecParameters {
        &self.params
    }

    fn decode(&mut self, packet: &Packet) -> Result<AudioBufferRef<'_>> {
        let n = match self.inner.decode(packet.buf(), MAX_FRAME, &mut self.pcm) {
            Ok(n) if n > 0 && n <= MAX_FRAME => n,
            _ => return decode_error("opus: decode failed"),
        };
        let skip = self.pre_skip.min(n);
        self.pre_skip -= skip;
        let frames = n - skip;
        self.buf.clear();
        self.buf.render_reserved(Some(frames));
        for ch in 0..self.channels {
            let plane = self.buf.chan_mut(ch);
            for (i, s) in plane.iter_mut().enumerate() {
                *s = self.pcm[(skip + i) * self.channels + ch];
            }
        }
        Ok(self.buf.as_audio_buffer_ref())
    }

    fn finalize(&mut self) -> FinalizeResult {
        FinalizeResult::default()
    }

    fn last_decoded(&self) -> AudioBufferRef<'_> {
        self.buf.as_audio_buffer_ref()
    }
}

/// symphonia's default codecs plus Opus.
pub fn codecs() -> &'static CodecRegistry {
    static REG: std::sync::OnceLock<CodecRegistry> = std::sync::OnceLock::new();
    REG.get_or_init(|| {
        let mut r = CodecRegistry::new();
        symphonia::default::register_enabled_codecs(&mut r);
        r.register_all::<OpusDecoder>();
        r
    })
}
