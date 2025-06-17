mod audio;
mod llm;
mod stt;
#[cfg(test)]
mod test_buffer;

use anyhow::Result;
use std::env;

fn main() -> Result<()> {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <models_directory>", args[0]);
        std::process::exit(1);
    }

    let models_dir = &args[1];
    let stt =
        stt::Stt::new(format!("{models_dir}/ggerganov/whisper.cpp/ggml-base-q8_0.bin").as_str())?;
    let audio = audio::Audio::new()?;

    let mut llm = llm::Llm::new(
        format!(
            "{models_dir}/bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf"
        )
        .as_str(),
        Box::new(std::io::stdout()),
    )?;

    loop {
        // Record audio until Enter is pressed
        let audio_data = match audio.record_until_enter() {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Recording error: {}", e);
                continue;
            }
        };

        // Skip if no audio data
        if audio_data.is_empty() {
            println!("No audio recorded, try again.");
            continue;
        }

        // Transcribe the audio
        let transcription = match stt.transcribe(&audio_data) {
            Ok(text) => {
                println!("Transcribed: {}", text);
                text
            }
            Err(e) => {
                eprintln!("Transcription error: {}", e);
                continue;
            }
        };

        // Send to LLM if we have transcribed text
        if !transcription.trim().is_empty() {
            llm.chat(&transcription)?;
            println!();
        }
    }
}
