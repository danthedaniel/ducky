# Ducky-RS: Voice Assistant with Whisper + Llama

A Rust-based voice assistant that uses Whisper for speech-to-text, Llama for text generation, and TTS for speech output.

## Current Status

✅ **Audio Pipeline**: Complete - microphone input with voice activity detection
🚧 **Whisper Integration**: TODO - requires correct API implementation
🚧 **Llama Integration**: TODO - requires correct API implementation
🚧 **TTS Integration**: TODO - requires correct API implementation

## Dependencies

The project includes the following dependencies:

- `cpal` - Cross-platform audio I/O for microphone input
- `whisper-rs` - Rust bindings for OpenAI Whisper (speech-to-text)
- `llama-cpp-2` - Rust bindings for llama.cpp (text generation)
- `tts_rust` - Text-to-speech functionality
- `tinyaudio` - Audio output (currently unused)

## Models

The application expects models to be downloaded in the following structure:

```
models/
├── openai/
│   └── whisper-base/
│       └── base.en.pt
└── bartowski/
    └── Llama-3.2-1B-Instruct-GGUF/
        └── Llama-3.2-1B-Instruct-Q4_0.gguf
```

## How It Works

1. **Audio Capture**: Uses `cpal` to capture audio from the default microphone
2. **Voice Activity Detection**: Simple RMS-based detection to identify when speech starts/stops
3. **Speech-to-Text**: (TODO) Process audio through Whisper model
4. **Text Generation**: (TODO) Generate responses using Llama model
5. **Text-to-Speech**: (TODO) Convert responses to speech

## Current Implementation

The current `main.rs` provides:

- ✅ Audio buffer management with circular buffer
- ✅ Voice activity detection using RMS threshold
- ✅ Audio stream setup and management
- ✅ Basic application structure and error handling
- 🚧 Placeholder implementations for AI components

## Next Steps

To complete the implementation, you need to:

### 1. Implement Whisper Integration

```rust
// In transcribe_audio() method
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

let whisper_ctx = WhisperContext::new_with_params(
    "models/openai/whisper-base/base.en.pt",
    WhisperContextParameters::default(),
)?;

let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
// Configure params...
// Run inference and extract text
```

### 2. Implement Llama Integration

```rust
// In generate_response() method
use llama_cpp_2::{/* correct imports */};

// Initialize model and context
// Tokenize input
// Generate response tokens
// Decode to text
```

### 3. Implement TTS Integration

```rust
// In speak_text() method
use tts_rust::Tts;

let tts = Tts::default()?;
tts.speak(text, false)?;
```

## API Documentation References

- **TinyAudio**: [docs.rs/tinyaudio](https://docs.rs/tinyaudio/1.1.0/tinyaudio/)
- **Llama-cpp-2**: [docs.rs/llama-cpp-2](https://docs.rs/crate/llama-cpp-2/latest)
- **Whisper-rs**: [docs.rs/whisper-rs](https://docs.rs/whisper-rs/0.14.3/whisper_rs/)

## Building and Running

```bash
# Check compilation
cargo check

# Build the project
cargo build --release

# Run the assistant (after implementing AI components)
cargo run
```

## Configuration

Key constants in `src/main.rs`:

- `SAMPLE_RATE`: 16000 Hz (required by Whisper)
- `CHANNELS`: 1 (mono audio)
- `BUFFER_SIZE`: 3 seconds of audio
- `SILENCE_THRESHOLD`: 0.01 (voice activity detection)
- `MIN_AUDIO_LENGTH`: 1 second minimum recording

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Microphone    │───▶│ Audio Buffer │───▶│ Voice Activity  │
│     Input       │    │              │    │   Detection     │
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                     │
                                                     ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   TTS Output    │◀───│    Llama     │◀───│    Whisper      │
│   (Speaker)     │    │ (Text Gen)   │    │ (Speech-to-Text)│
└─────────────────┘    └──────────────┘    └─────────────────┘
```

## License

This project uses dependencies with various licenses:
- MIT: Most Rust crates
- Apache-2.0: Some system libraries
- Unlicense: whisper-rs

Please review individual dependency licenses for compliance requirements.