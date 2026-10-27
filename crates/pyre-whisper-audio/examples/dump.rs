fn main() {
    let a: Vec<String> = std::env::args().collect();
    let w = pyre_whisper_audio::decode::decode_to_mono_pcm(std::path::Path::new(&a[1])).unwrap();
    let s = &w.samples;
    let rms = (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt();
    println!("{} samples ({:.2}s @16k) rms {:.4}", s.len(), s.len() as f32 / 16000.0, rms);
    let spec = hound::WavSpec { channels: 1, sample_rate: 16000, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let mut wr = hound::WavWriter::create(&a[2], spec).unwrap();
    for x in s { wr.write_sample(*x).unwrap(); }
}

