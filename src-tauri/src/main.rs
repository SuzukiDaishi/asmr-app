// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use once_cell::sync::Lazy;
use rodio::{Decoder, OutputStream, Sink, Source};
use std::{
    fs::File,
    io::BufReader,
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
};

mod phantomsilhouette;
mod hrtf;
mod crossfade;
mod distance_adjust;
use distance_adjust::DistanceAdjustProcessor;

// -----------------------------
// グローバルな pre/post gain (線形)
// -----------------------------
static PRE_GAIN: Lazy<Arc<Mutex<f32>>> = Lazy::new(|| Arc::new(Mutex::new(1.0)));
static POST_GAIN: Lazy<Arc<Mutex<f32>>> = Lazy::new(|| Arc::new(Mutex::new(1.0)));

// -----------------------------
// グローバルな 動的HRTFパラメータ（azimuth, elevation）
// -----------------------------
static DYNAMIC_HRTF_AZIMUTH: Lazy<Arc<Mutex<f32>>> =
    Lazy::new(|| Arc::new(Mutex::new(0.0)));
static DYNAMIC_HRTF_ELEVATION: Lazy<Arc<Mutex<f32>>> =
    Lazy::new(|| Arc::new(Mutex::new(0.0)));

// -----------------------------
// NoiseType の文字列一覧（フロント側へ返す用）
// -----------------------------
static NOISE_TYPES: Lazy<Vec<&'static str>> =
    Lazy::new(|| vec!["White", "Pink", "Velvet"]);

// -----------------------------
// PhantomSilhouette.rs 内の NoiseType 列挙型をインポート
// -----------------------------
use phantomsilhouette::NoiseType;

/// 文字列から NoiseType に変換します。大文字小文字は区別しません。
fn parse_noise_type(s: &str) -> NoiseType {
    match s.to_lowercase().as_str() {
        "white" => NoiseType::White,
        "pink" => NoiseType::Pink,
        "velvet" => NoiseType::Velvet,
        _ => NoiseType::White, // デフォルト
    }
}

/// 再生コマンド
enum AudioCommand {
    Play(String),
    AsmrPlayExtended(String, bool, String, bool), // (path, whisper, sourceNoise, getCloser)
    Stop,
}

/// AudioManager: 同時に1つのみ再生
pub struct AudioManager {
    command_sender: Sender<AudioCommand>,
}

impl AudioManager {
    pub fn new() -> Self {
        let (tx, rx): (Sender<AudioCommand>, Receiver<AudioCommand>) = mpsc::channel();
        thread::spawn(move || {
            let mut current_playback: Option<(OutputStream, Sink)> = None;
            for command in rx {
                match command {
                    AudioCommand::Stop => {
                        if let Some((_, sink)) = &current_playback {
                            println!("[DEBUG] Stopping current playback...");
                            sink.stop();
                        }
                        current_playback = None;
                    }
                    AudioCommand::Play(path) => {
                        println!("[DEBUG] Received DRY PLAY for path: {}", path);
                        if let Some((_, sink)) = &current_playback {
                            sink.stop();
                        }
                        match Self::play_dry_internal(&path) {
                            Ok((stream, sink)) => current_playback = Some((stream, sink)),
                            Err(e) => eprintln!("[ERROR] dry_play failed: {:?}", e),
                        }
                    }
                    AudioCommand::AsmrPlayExtended(path, whisper, sourceNoise, getCloser) => {
                        println!("[DEBUG] Received ASMR PLAY for path: {}", path);
                        if let Some((_, sink)) = &current_playback {
                            sink.stop();
                        }
                        match Self::play_asmr_internal(&path, whisper, sourceNoise, getCloser) {
                            Ok((stream, sink)) => current_playback = Some((stream, sink)),
                            Err(e) => eprintln!("[ERROR] asmr_play failed: {:?}", e),
                        }
                    }
                }
            }
        });
        AudioManager { command_sender: tx }
    }

    fn play_dry_internal(path: &str) -> Result<(OutputStream, Sink), String> {
        println!("[DEBUG] Dry internal start: {}", path);
        let file = File::open(path)
            .map_err(|e| format!("Failed to open file '{}': {:?}", path, e))?;
        let source_decoder = Decoder::new(BufReader::new(file))
            .map_err(|e| format!("Failed to decode file '{}': {:?}", path, e))?;
        let channels = source_decoder.channels();
        let sample_rate = source_decoder.sample_rate();
        let samples: Vec<f32> = source_decoder.convert_samples().collect();

        let dyn_source = DynamicGainSource::new(
            samples,
            channels,
            sample_rate,
            Arc::clone(&PRE_GAIN),
            Arc::clone(&POST_GAIN),
        );
        let (stream, stream_handle) = OutputStream::try_default()
            .map_err(|e| format!("Failed to get default output stream: {:?}", e))?;
        let sink = Sink::try_new(&stream_handle)
            .map_err(|e| format!("Failed to create sink: {:?}", e))?;
        sink.append(dyn_source);
        sink.play();
        println!("[DEBUG] Dry internal playback started with dynamic gain");
        Ok((stream, sink))
    }

    fn play_asmr_internal(
        path: &str,
        whisper: bool,
        sourceNoise: String,
        getCloser: bool,
    ) -> Result<(OutputStream, Sink), String> {
        println!("[DEBUG] ASMR internal start: {}", path);
        let mut reader = hound::WavReader::open(path)
            .map_err(|e| format!("Failed to open WAV '{}': {:?}", path, e))?;
        let spec = reader.spec();
        println!("[DEBUG] WAV format: {:?}", spec);
        if spec.channels != 1 {
            return Err(format!("Only mono WAV supported, found {} channels", spec.channels));
        }
        // WAVファイルからサンプルを読み込み (f64)
        let samples_f64: Vec<f64> = match (spec.bits_per_sample, spec.sample_format) {
            (16, hound::SampleFormat::Int) => reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f64 / i16::MAX as f64)
                .collect(),
            (24, hound::SampleFormat::Int) => reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f64 / (1 << 23) as f64)
                .collect(),
            (32, hound::SampleFormat::Float) => reader
                .samples::<f32>()
                .map(|s| s.unwrap() as f64)
                .collect(),
            _ => {
                return Err(format!(
                    "Unsupported WAV format bits_per_sample={} sample_format={:?}",
                    spec.bits_per_sample, spec.sample_format
                ));
            }
        };

        // whisper が true の場合は phantom silhouette 処理を適用、false の場合はバイパス
        let base_samples: Vec<f32> = if whisper {
            let mut processor = phantomsilhouette::PhantomSilhouetteProcessor::new(spec.sample_rate as usize);
            let noise = parse_noise_type(&sourceNoise);
            processor.set_noise_type(noise);
            processor.configure_dio(70.0, 600.0, 5.0);
            processor.configure_cheaptrick(-0.15, 1024);
            processor.configure_d4c(0.85);
            let processed = processor.process(&samples_f64)
                .map_err(|e| format!("PhantomSilhouette error: {:?}", e))?;
            let mut processed_f32: Vec<f32> = processed.iter().map(|&x| x as f32).collect();
            // whisper が true の場合、phantom silhouette 後の音声を 0.7 倍にする
            processed_f32.iter_mut().for_each(|s| *s *= 0.7);
            processed_f32
        } else {
            // バイパス：WAVから読み込んだ生のサンプルをそのまま使用
            samples_f64.iter().map(|&x| x as f32).collect()
        };

        // getCloser が true の場合は距離補正を適用、false の場合はバイパス
        let final_samples: Vec<f32> = if getCloser {
            let processed_stereo: Vec<[f32; 2]> = base_samples.iter().map(|&s| [s, s]).collect();
            // getCloser が true の場合、距離補正用の倍率を 1.0 とする
            let mut distance_processor =
                DistanceAdjustProcessor::new(spec.sample_rate as usize, Some(1.0))
                    .map_err(|e| format!("Distance processor error: {:?}", e))?;
            let distance_adjusted_stereo = distance_processor.process(&processed_stereo[..]);
            distance_adjusted_stereo.iter().map(|[l, _r]| *l).collect()
        } else {
            // バイパス：距離補正なし
            base_samples
        };

        let dynamic_hrtf_source = DynamicHrtfSourceOnnx::new(
            final_samples,
            2,
            spec.sample_rate,
            *DYNAMIC_HRTF_AZIMUTH.lock().unwrap(),
            *DYNAMIC_HRTF_ELEVATION.lock().unwrap(),
        );
        let (stream, stream_handle) = OutputStream::try_default()
            .map_err(|e| format!("Failed to get default output stream: {:?}", e))?;
        let sink = Sink::try_new(&stream_handle)
            .map_err(|e| format!("Failed to create sink: {:?}", e))?;
        sink.append(dynamic_hrtf_source);
        sink.play();
        println!("[DEBUG] ASMR playback started with dynamic HRTF (ONNX)");
        Ok((stream, sink))
    }

    // --------------------------
    // Public API
    // --------------------------
    pub fn play_dry(&self, path: String) -> Result<(), String> {
        self.command_sender
            .send(AudioCommand::Play(path))
            .map_err(|e| format!("Failed to send Play command: {:?}", e))
    }
    pub fn play_asmr(
        &self,
        path: String,
        whisper: bool,
        sourceNoise: String,
        getCloser: bool,
    ) -> Result<(), String> {
        self.command_sender
            .send(AudioCommand::AsmrPlayExtended(path, whisper, sourceNoise, getCloser))
            .map_err(|e| format!("Failed to send AsmrPlay command: {:?}", e))
    }
    pub fn stop(&self) -> Result<(), String> {
        self.command_sender
            .send(AudioCommand::Stop)
            .map_err(|e| format!("Failed to send Stop command: {:?}", e))
    }
}

// -------------------------------
// DynamicGainSource: 毎サンプルに pre_gain * sample * post_gain を掛ける（既存実装）
// -------------------------------
struct DynamicGainSource {
    data: Vec<f32>,
    position: usize,
    channels: u16,
    sample_rate: u32,
    pre_gain: Arc<Mutex<f32>>,
    post_gain: Arc<Mutex<f32>>,
}

impl DynamicGainSource {
    fn new(
        data: Vec<f32>,
        channels: u16,
        sample_rate: u32,
        pre_gain: Arc<Mutex<f32>>,
        post_gain: Arc<Mutex<f32>>,
    ) -> Self {
        Self {
            data,
            position: 0,
            channels,
            sample_rate,
            pre_gain,
            post_gain,
        }
    }
}

impl Iterator for DynamicGainSource {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.position >= self.data.len() {
            None
        } else {
            let sample = self.data[self.position];
            self.position += 1;
            let pre = *self.pre_gain.lock().unwrap();
            let post = *self.post_gain.lock().unwrap();
            Some(sample * pre * post)
        }
    }
}

impl Source for DynamicGainSource {
    fn current_frame_len(&self) -> Option<usize> { None }
    fn channels(&self) -> u16 { self.channels }
    fn sample_rate(&self) -> u32 { self.sample_rate }
    fn total_duration(&self) -> Option<std::time::Duration> { None }
}

// -------------------------------
// DynamicHrtfSourceOnnx: ONNX HRTFを用いて動的にHRIR畳み込みを行い、左右2チャンネル出力するソース
// 各ブロック間で短いクロスフェード (fade_length) を実施し、連続性を確保する。
pub struct DynamicHrtfSourceOnnx {
    data: Vec<f32>,              // 入力モノラル信号
    pos: usize,                  // 現在位置
    channels: u16,               // 出力は常に2チャンネル
    sample_rate: u32,
    block_size: usize,           // ブロック処理サイズ (例: 1秒分のサンプル数)
    fade_length: usize,          // クロスフェード長
    processed_buffer: Vec<[f32; 2]>, // 最新の処理済みブロック
    output_buffer: Vec<f32>,     // インターリーブ済み出力バッファ
    output_index: usize,         // 出力バッファの読み出し位置
    hrtf_processor: hrtf::HrtfProcessor, // ONNX HRTFプロセッサ
    prev_tail: Option<Vec<[f32; 2]>>, // 前ブロックの終わり部分
}

impl DynamicHrtfSourceOnnx {
    pub fn new(
        data: Vec<f32>,
        channels: u16,
        sample_rate: u32,
        _initial_azimuth: f32,
        _initial_elevation: f32,
    ) -> Self {
        let hrtf_processor = hrtf::HrtfProcessor::new("hrtf_model.onnx")
            .expect("Failed to create HRTF processor");
        Self {
            data,
            pos: 0,
            channels,
            sample_rate,
            block_size: sample_rate as usize, // 例: 1秒分のサンプル数
            fade_length: 64,
            processed_buffer: Vec::new(),
            output_buffer: Vec::new(),
            output_index: 0,
            hrtf_processor,
            prev_tail: None,
        }
    }

    /// sqrt-Hann 窓を生成します。窓の二乗和が1になるため、オーバーラップ時も音量が一定です。
    fn sqrt_hann_window(length: usize) -> Vec<f32> {
        (0..length)
            .map(|i| {
                let w = 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * i as f32) / ((length - 1) as f32)).cos();
                w.sqrt()
            })
            .collect()
    }

    /// ブロック単位で処理を行い、出力バッファを更新します。
    fn process_next_block(&mut self) {
        if self.pos >= self.data.len() {
            self.processed_buffer.clear();
            self.output_buffer.clear();
            return;
        }
        let end = std::cmp::min(self.pos + self.block_size, self.data.len());
        let mut block: Vec<f32> = self.data[self.pos..end].to_vec();
        if block.len() < self.block_size {
            block.resize(self.block_size, 0.0);
        }
        self.pos += self.block_size;

        // 距離補正を適用 (ONNX 推論前)
        let stereo_block: Vec<[f32; 2]> = block.iter().map(|&s| [s, s]).collect();
        let mut distance_processor =
            DistanceAdjustProcessor::new(self.sample_rate as usize, Some(0.5))
                .expect("Distance processor error");
        let distance_adjusted_stereo = distance_processor.process(&stereo_block[..]);
        let mut adjusted_block: Vec<f32> =
            distance_adjusted_stereo.iter().map(|[l, _r]| *l).collect();
        if adjusted_block.len() < self.block_size {
            let last = *adjusted_block.last().unwrap();
            adjusted_block.resize(self.block_size, last);
        } else if adjusted_block.len() > self.block_size {
            adjusted_block.truncate(self.block_size);
        }

        let azimuth = *DYNAMIC_HRTF_AZIMUTH.lock().unwrap();
        let elevation = *DYNAMIC_HRTF_ELEVATION.lock().unwrap();

        let mut processed = self
            .hrtf_processor
            .apply_hrtf(&adjusted_block, azimuth, elevation)
            .expect("Failed to apply HRTF");
        if processed.len() < self.block_size {
            let last = processed.last().unwrap().clone();
            processed.resize(self.block_size, last);
        } else if processed.len() > self.block_size {
            processed.truncate(self.block_size);
        }

        let window = Self::sqrt_hann_window(processed.len());
        for i in 0..processed.len() {
            processed[i][0] *= window[i];
            processed[i][1] *= window[i];
        }

        let fade = self.fade_length.min(processed.len());
        let mut output_block = Vec::new();
        if let Some(prev) = self.prev_tail.take() {
            let fade = fade.min(prev.len());
            for i in 0..fade {
                let alpha = i as f32 / fade as f32;
                let left = prev[i][0] * (1.0 - alpha) + processed[i][0] * alpha;
                let right = prev[i][1] * (1.0 - alpha) + processed[i][1] * alpha;
                output_block.push([left, right]);
            }
            for i in fade..processed.len() {
                output_block.push(processed[i]);
            }
        } else {
            output_block = processed.clone();
        }
        if processed.len() >= fade {
            self.prev_tail = Some(processed[processed.len() - fade..].to_vec());
        } else {
            self.prev_tail = Some(processed.clone());
        }

        // 動的ゲインを適用
        let pre = *PRE_GAIN.lock().unwrap();
        let post = *POST_GAIN.lock().unwrap();
        let total_gain = pre * post;
        for sample_pair in output_block.iter_mut() {
            sample_pair[0] *= total_gain;
            sample_pair[1] *= total_gain;
        }

        self.output_buffer = output_block
            .iter()
            .flat_map(|&[l, r]| vec![l, r])
            .collect();
        self.output_index = 0;
    }
}

impl Iterator for DynamicHrtfSourceOnnx {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.output_index >= self.output_buffer.len() {
            self.process_next_block();
            if self.output_buffer.is_empty() {
                return None;
            }
        }
        let sample = self.output_buffer[self.output_index];
        self.output_index += 1;
        Some(sample)
    }
}

impl Source for DynamicHrtfSourceOnnx {
    fn current_frame_len(&self) -> Option<usize> { None }
    fn channels(&self) -> u16 { self.channels }
    fn sample_rate(&self) -> u32 { self.sample_rate }
    fn total_duration(&self) -> Option<std::time::Duration> { None }
}

// -------------------------------
// AudioManager のグローバル管理
// -------------------------------
static AUDIO_MANAGER: Lazy<Mutex<AudioManager>> =
    Lazy::new(|| Mutex::new(AudioManager::new()));

//
// 以下、Tauriコマンド
//
use tauri::command;

#[command]
fn set_pre_gain(gain: f32) -> Result<String, String> {
    if let Ok(mut g) = PRE_GAIN.lock() {
        *g = gain;
        println!("[DEBUG] PRE_GAIN updated to {}", gain);
    }
    Ok(format!("pre_gain updated: {}", gain))
}

#[command]
fn set_post_gain(gain: f32) -> Result<String, String> {
    if let Ok(mut g) = POST_GAIN.lock() {
        *g = gain;
        println!("[DEBUG] POST_GAIN updated to {}", gain);
    }
    Ok(format!("post_gain updated: {}", gain))
}

#[command]
fn set_hrtf_azimuth(azimuth: f32) -> Result<String, String> {
    if let Ok(mut a) = DYNAMIC_HRTF_AZIMUTH.lock() {
        *a = azimuth;
        println!("[DEBUG] HRTF azimuth updated to {}", azimuth);
    }
    Ok(format!("HRTF azimuth updated: {}", azimuth))
}

#[command]
fn set_hrtf_elevation(elevation: f32) -> Result<String, String> {
    if let Ok(mut e) = DYNAMIC_HRTF_ELEVATION.lock() {
        *e = elevation;
        println!("[DEBUG] HRTF elevation updated to {}", elevation);
    }
    Ok(format!("HRTF elevation updated: {}", elevation))
}

#[command]
fn get_noise_types() -> Result<Vec<String>, String> {
    Ok(NOISE_TYPES.iter().map(|s| s.to_string()).collect())
}

#[command]
fn dry_play(path: &str) -> Result<String, String> {
    let manager = AUDIO_MANAGER.lock().map_err(|e| format!("Lock error: {:?}", e))?;
    manager.play_dry(path.to_string())?;
    Ok(format!("Playing dry audio at: {}", path))
}

#[command]
fn asmr_play(
    path: &str,
    whisper: bool,
    sourceNoise: String,
    getCloser: bool,
) -> Result<String, String> {
    let manager = AUDIO_MANAGER.lock().map_err(|e| format!("Lock error: {:?}", e))?;
    manager.play_asmr(path.to_string(), whisper, sourceNoise, getCloser)?;
    Ok(format!("Playing asmr audio at: {}", path))
}

#[command]
fn stop_audio() -> Result<String, String> {
    let manager = AUDIO_MANAGER.lock().map_err(|e| format!("Lock error: {:?}", e))?;
    manager.stop()?;
    Ok("Stopped audio".to_string())
}

// -------------------------------
// main関数 (Tauriアプリケーション)
// -------------------------------
fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            set_pre_gain,
            set_post_gain,
            set_hrtf_azimuth,
            set_hrtf_elevation,
            get_noise_types,
            dry_play,
            asmr_play,
            stop_audio
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
