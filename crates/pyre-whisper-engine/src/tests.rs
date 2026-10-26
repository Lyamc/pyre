#[cfg(test)]
mod tests {
    use crate::token::Language;
    use crate::transcribe::{get_output_path, format_timestamp, format_srt, TranscriptSegment};
    use strum::IntoEnumIterator;
    use std::path::PathBuf;

    #[test]
    fn test_languages() {
        for lang in Language::iter() {
            let s = lang.as_str();
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_naming_logic() {
        let input = PathBuf::from("audio.wav");
        let output = get_output_path(&input, &None, "txt");
        assert_eq!(output, PathBuf::from("audio.wav.txt"));

        let output_srt = get_output_path(&input, &None, "srt");
        assert_eq!(output_srt, PathBuf::from("audio.wav.srt"));

        let specified_out = Some(PathBuf::from("result.txt"));
        let output_spec = get_output_path(&input, &specified_out, "txt");
        assert_eq!(output_spec, PathBuf::from("result.txt"));
    }

    #[test]
    fn test_timestamp_formatting() {
        assert_eq!(format_timestamp(0.0), "00:00:00,000");
        assert_eq!(format_timestamp(1.5), "00:00:01,500");
        assert_eq!(format_timestamp(3661.02), "01:01:01,020");
    }

    #[test]
    fn test_srt_formatting() {
        let segments = vec![
            TranscriptSegment {
                start: 0.0,
                end: 2.0,
                text: "Hello".to_string(),
                diarization: None,
                location: None,
            },
            TranscriptSegment {
                start: 2.0,
                end: 4.0,
                text: "World".to_string(),
                diarization: None,
                location: None,
            },
        ];
        let srt = format_srt(&segments);
        assert!(srt.contains("1\n00:00:00,000 --> 00:00:02,000\nHello"));
        assert!(srt.contains("2\n00:00:02,000 --> 00:00:04,000\nWorld"));
    }

    #[test]
    fn test_npy_integration() {
        use npy::NpyData;
        // Create a minimal .npy header for a 1D array of 2 f32s
        // { 'descr': '<f4', 'fortran_order': False, 'shape': (2,), }
        let mut data = vec![0x93];
        data.extend_from_slice(b"NUMPY");
        data.extend_from_slice(&[0x01, 0x00]); // version
        
        let header = "{ 'descr': '<f4', 'fortran_order': False, 'shape': (2,), }";
        let mut header_bytes = header.as_bytes().to_vec();
        while (header_bytes.len() + 10 + 1) % 16 != 0 {
            header_bytes.push(b' ');
        }
        header_bytes.push(b'\n');
        
        let header_len = header_bytes.len() as u16;
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(&header_bytes);
        
        // Add 2 f32s: 1.0 and 2.0
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        
        let npy: NpyData<f32> = NpyData::from_bytes(&data).expect("Failed to parse npy");
        let vec = npy.to_vec();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[1], 2.0);
    }
}
