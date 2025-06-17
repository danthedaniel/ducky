use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, Stream, StreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub struct OutputStream {
    stream: Stream,
    finished: Arc<AtomicBool>,
}

impl OutputStream {
    pub fn new(stream: Stream, finished: Arc<AtomicBool>) -> Self {
        Self { stream, finished }
    }

    pub fn play(&self) -> Result<()> {
        self.stream.play()?;
        Ok(())
    }

    pub fn stop(self) -> Result<()> {
        self.stream.pause()?;
        self.finished.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub fn wait(&self) {
        while !self.finished.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Give a bit more time for the last samples to play
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

pub struct Audio {
    input_device: Device,
    output_device: Device,
    input_config: StreamConfig,
    output_config: StreamConfig,
    input_sample_rate: u32,
}

impl Audio {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();

        // Get input device
        let input_device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device available"))?;
        println!("Audio input device: {}", input_device.name().unwrap());

        // Get output device
        let output_device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No output device available"))?;
        println!("Audio output device: {}", output_device.name().unwrap());

        // Use the device's default input config
        let input_config: StreamConfig = input_device.default_input_config()?.into();
        let input_sample_rate = input_config.sample_rate.0;

        println!(
            "Using input config: {} channels, {} Hz (will resample to 16kHz)",
            input_config.channels, input_sample_rate
        );

        let output_config = output_device.default_output_config()?.into();

        Ok(Self {
            input_device,
            output_device,
            input_config,
            output_config,
            input_sample_rate,
        })
    }

    /// Resample audio from source sample rate to target sample rate using linear interpolation
    fn resample_audio(&self, audio_data: &[f32], target_sample_rate: u32) -> Vec<f32> {
        if self.input_sample_rate == target_sample_rate {
            return audio_data.to_vec();
        }

        let ratio = self.input_sample_rate as f64 / target_sample_rate as f64;
        let output_length = (audio_data.len() as f64 / ratio) as usize;
        let mut resampled = Vec::with_capacity(output_length);

        for i in 0..output_length {
            let src_index = i as f64 * ratio;
            let src_index_floor = src_index.floor() as usize;
            let src_index_ceil = (src_index_floor + 1).min(audio_data.len() - 1);
            let fraction = src_index - src_index_floor as f64;

            if src_index_floor < audio_data.len() {
                let sample1 = audio_data[src_index_floor];
                let sample2 = audio_data[src_index_ceil];
                let interpolated = sample1 + (sample2 - sample1) * fraction as f32;
                resampled.push(interpolated);
            }
        }

        resampled
    }

    pub fn record_until_enter(&self) -> Result<Vec<f32>> {
        let channels = self.input_config.channels as usize;
        let audio_data = Arc::new(Mutex::new(Vec::new()));
        let recording = Arc::new(AtomicBool::new(true));

        {
            let stream =
                self.build_input_stream::<f32>(Arc::clone(&audio_data), Arc::clone(&recording))?;
            stream.play()?;

            println!("Recording... Press Enter to stop.");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();

            // Stop recording
            recording.store(false, Ordering::Relaxed);
        }

        let recorded_data = audio_data.lock().unwrap().clone();

        // Convert to mono if stereo
        let mono_data = if channels == 2 {
            recorded_data
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                .collect()
        } else {
            recorded_data
        };

        // Resample to 16kHz
        let resampled_data = self.resample_audio(&mono_data, 16000);
        println!(
            "Resampled {} samples from {}Hz to 16kHz",
            resampled_data.len(),
            self.input_sample_rate
        );

        Ok(resampled_data)
    }

    pub fn playback(&self, audio_data: &[f32]) -> Result<OutputStream> {
        let channels = self.output_config.channels as usize;

        // Convert mono to stereo if needed
        let playback_data: Vec<f32> = if channels == 2 {
            audio_data
                .iter()
                .flat_map(|&sample| [sample, sample])
                .collect()
        } else {
            audio_data.to_vec()
        };

        let playback_data = Arc::new(Mutex::new(playback_data));
        let playback_index = Arc::new(Mutex::new(0));
        let finished = Arc::new(AtomicBool::new(false));

        let playback_data_clone = Arc::clone(&playback_data);
        let playback_index_clone = Arc::clone(&playback_index);
        let finished_clone = Arc::clone(&finished);

        let stream = self.output_device.build_output_stream(
            &self.output_config,
            move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let data = playback_data_clone.lock().unwrap();
                let mut index = playback_index_clone.lock().unwrap();

                for sample in output.iter_mut() {
                    if *index < data.len() {
                        *sample = data[*index];
                        *index += 1;
                    } else {
                        *sample = 0.0;
                        finished_clone.store(true, Ordering::Relaxed);
                    }
                }
            },
            |err| eprintln!("An error occurred on the audio output stream: {}", err),
            None,
        )?;

        Ok(OutputStream::new(stream, finished))
    }

    fn build_input_stream<T>(
        &self,
        audio_data: Arc<Mutex<Vec<f32>>>,
        recording: Arc<AtomicBool>,
    ) -> Result<Stream>
    where
        T: Sample + cpal::SizedSample + Send + 'static,
        f32: cpal::FromSample<T>,
    {
        let config = self.input_config.clone();
        let stream = self.input_device.build_input_stream(
            &config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                if recording.load(Ordering::Relaxed) {
                    let mut audio_data = audio_data.lock().unwrap();
                    for &sample in data.iter() {
                        audio_data.push(f32::from_sample(sample));
                    }
                }
            },
            |err| eprintln!("An error occurred on the audio input stream: {}", err),
            None,
        )?;

        Ok(stream)
    }
}
