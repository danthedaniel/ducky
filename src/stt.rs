use anyhow::Result;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
pub struct Stt {
    ctx: WhisperContext,
}

impl Stt {
    pub fn new(model_path: &str) -> Result<Self> {
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
            .map_err(|e| anyhow::anyhow!("failed to load model: {}", e))?;

        Ok(Self { ctx })
    }

    pub fn transcribe(&self, audio_data: &[f32]) -> Result<String> {
        // Validate audio data
        if audio_data.is_empty() {
            anyhow::bail!("Audio data is empty");
        }

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(1);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let mut state = self.ctx.create_state()?;
        state.full(params, audio_data)?;

        let mut full_text = String::new();
        let num_segments = state.full_n_segments()?;
        for i in 0..num_segments {
            let segment_text = state.full_get_segment_text(i)?;
            full_text.push_str(&segment_text);
        }

        Ok(full_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound;

    fn read_audio_file(path: &str) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path).expect("Failed to open audio file");
        let hound::WavSpec { channels, .. } = reader.spec();
        let samples = reader
            .samples::<i16>()
            .map(|s| s.expect("Failed to read sample"))
            .collect::<Vec<_>>();

        let mut audio = vec![0.0f32; samples.len()];
        whisper_rs::convert_integer_to_float_audio(&samples, &mut audio).expect("Conversion error");

        match channels {
            1 => audio,
            2 => whisper_rs::convert_stereo_to_mono_audio(&audio)
                .expect("Stereo to mono conversion error"),
            _ => panic!(">2 channels unsupported"),
        }
    }

    #[test]
    fn test_transcribe_hello_world() {
        let audio = read_audio_file("tests/samples/hello_world.wav");
        let stt = Stt::new("models/ggerganov/whisper.cpp/ggml-base-q8_0.bin")
            .expect("Failed to create STT instance");

        let result = stt.transcribe(&audio).expect("Failed to transcribe audio");
        let result_trimmed = result.trim().to_lowercase();

        assert!(
            result_trimmed.contains("hello") && result_trimmed.contains("world"),
            "Expected transcription to contain 'hello' and 'world', but got: '{}'",
            result
        );
    }
}
