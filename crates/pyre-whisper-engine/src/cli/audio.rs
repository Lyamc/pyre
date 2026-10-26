use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rtrb::RingBuffer;
use std::sync::mpsc;

pub const BUFFER_FRAME_COUNT: usize = 35;
pub const MINIMUM_SAMPLE_COUNT: usize = 1600 * 4; // @ 16kHz = 400ms
pub const MAXIMUM_SAMPLE_COUNT: usize = 1600 * 50;

/// Pure-Rust energy VAD (replaces webrtc-vad / libwebrtc, which needs the `cc` crate).
/// Frame is 10ms @ 16 kHz (160 samples). Threshold is RMS of i16 samples.
const ENERGY_VAD_RMS_THRESHOLD: f64 = 500.0;

fn is_voice_segment(frame: &[i16]) -> bool {
    if frame.is_empty() {
        return false;
    }
    let sum_sq: f64 = frame.iter().map(|&s| {
        let v = s as f64;
        v * v
    }).sum();
    (sum_sq / frame.len() as f64).sqrt() >= ENERGY_VAD_RMS_THRESHOLD
}

pub fn normalize_audio_data_to_16k(input_data: &[f32], input_sample_rate: &f32) -> Vec<f32> {
    let target_sample_rate = 16000f32;
    let resample_ratio = target_sample_rate / input_sample_rate;
    let mut resampled_data = Vec::new();
    for i in 0..(input_data.len() as f32 * resample_ratio) as usize {
        let x = i as f32 / resample_ratio;
        let x1 = x.floor() as usize;
        let x2 = x.ceil() as usize;

        if x2 >= input_data.len() {
            break;
        }

        let y1 = input_data[x1];
        let y2 = input_data[x2];

        let y = y1 + (y2 - y1) * (x - x1 as f32);
        resampled_data.push(y);
    }

    resampled_data
}

pub fn record_audio(sender: mpsc::Sender<Vec<i16>>) {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("Failed to get default input device");
    let config = device.default_input_config().expect("Failed to get default input config");
    let sample_rate = config.sample_rate() as f32;

    // Create a stream with the default input format
    let (mut producer, mut consumer) = RingBuffer::<i16>::new(16384);
    let stream = device.build_input_stream(
        config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let data_16k = normalize_audio_data_to_16k(data, &sample_rate);
            let vad_data_i16_16k: Vec<i16> = data_16k.iter().map(|x| (*x * 32767.0) as i16).collect();
            for sample in vad_data_i16_16k {
                if producer.push(sample).is_err() {
                    // Buffer full, dropping sample
                }
            }
        },
        move |err| eprintln!("Error: {}", err),
        None
    ).expect("Failed to build input stream");

    stream.play().expect("Failed to play stream");

    let mut unactive_count = 0;
    let mut speaking = false;
    let mut speech_segment = Vec::<i16>::new();
    loop {
        if consumer.slots() > 160 {
            let mut audio_frame = Vec::<i16>::new();
            for _ in 0..160 {
                match consumer.pop() {
                    Ok(value) => {
                        audio_frame.push(value);
                    }
                    Err(_) => {
                        break;
                    },
                }
            }

            let speech_active = is_voice_segment(&audio_frame);
            if speaking {
                if speech_active {
                    speech_segment.extend(audio_frame);
                    if speech_segment.len() > MAXIMUM_SAMPLE_COUNT {
                        sender.send(speech_segment.clone()).expect("Failed to send data");
                        speech_segment.clear();
                    }
                } else if unactive_count > BUFFER_FRAME_COUNT {
                    speaking = false;
                    if speech_segment.len() > MINIMUM_SAMPLE_COUNT {
                        sender.send(speech_segment.clone()).expect("Failed to send data");
                    }
                    speech_segment.clear();
                } else {
                    unactive_count += 1;
                }
            } else if speech_active {
                speaking = true;
                unactive_count = 0;
                speech_segment.extend(audio_frame);
            }
        } else {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}
