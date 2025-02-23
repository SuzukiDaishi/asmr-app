// src/phantomsilhouette.rs
use rand::Rng;
use rand::rngs::ThreadRng;
use Rust_WORLD::rsworld::{cheaptrick, d4c, dio, stonemask, synthesis};
use Rust_WORLD::rsworld_sys::{CheapTrickOption, D4COption, DioOption};
use std::f64::consts::PI;

/// ノイズタイプ列挙型
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum NoiseType {
    White,
    Pink,
    Velvet,
}

const VELVET_DENSITY: usize = 16; // 1/16の確率でパルス発生
const MIN_PULSE_INTERVAL: usize = 4; // 最小パルス間隔

/// 音声変換処理のメイン構造体
pub struct PhantomSilhouetteProcessor {
    sample_rate: usize,
    dio_options: DioOptions,
    cheaptrick_options: CheapTrickOptions,
    d4c_options: D4COptions,
    noise_type: NoiseType,
}

/// ピンクノイズ生成用の状態管理構造体
struct PinkNoiseState {
    values: [f64; 7],
    index: usize,
}

/// ベルベットノイズ生成用の状態管理構造体
struct VelvetNoiseState {
    last_pulse_position: usize,
}

impl PinkNoiseState {
    fn new() -> Self {
        Self {
            values: [0.0; 7],
            index: 0,
        }
    }
}

impl VelvetNoiseState {
    fn new() -> Self {
        Self {
            last_pulse_position: 0,
        }
    }
}


impl PhantomSilhouetteProcessor {
    /// 新しいプロセッサを作成
    pub fn new(sample_rate: usize) -> Self {
        Self {
            sample_rate,
            dio_options: DioOptions::default(),
            cheaptrick_options: CheapTrickOptions::default(),
            d4c_options: D4COptions::default(),
            noise_type: NoiseType::White,
        }
    }

    /// ノイズタイプ設定
    pub fn set_noise_type(&mut self, noise_type: NoiseType) {
        self.noise_type = noise_type;
    }

    /// DIOパラメータ設定
    pub fn configure_dio(&mut self, f0_floor: f64, f0_ceil: f64, frame_period: f64) {
        self.dio_options = DioOptions {
            f0_floor,
            f0_ceil,
            frame_period,
        };
    }

    /// CheapTrickパラメータ設定
    pub fn configure_cheaptrick(&mut self, q1: f64, fft_size: i32) {
        self.cheaptrick_options = CheapTrickOptions { q1, fft_size };
    }

    /// D4Cパラメータ設定
    pub fn configure_d4c(&mut self, threshold: f64) {
        self.d4c_options = D4COptions { threshold };
    }

    /// 音声処理実行
    pub fn process(&self, input_samples: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let fs = self.sample_rate as i32;
        let padded_samples = self.add_padding(input_samples);
        
        let (temporal_positions, mut f0) = dio(
            &padded_samples,
            fs,
            &self.dio_options.to_world_options(),
        );

        let refined_f0 = stonemask(&padded_samples, fs, &temporal_positions, &f0);
        let mut spectrogram = cheaptrick(
            &padded_samples,
            fs,
            &temporal_positions,
            &refined_f0,
            &mut self.cheaptrick_options.to_world_options(fs),
        );

        let aperiodicity = d4c(
            &padded_samples,
            fs,
            &temporal_positions,
            &refined_f0,
            &self.d4c_options.to_world_options(),
        );

        self.apply_effects(&mut f0, &mut spectrogram);
        
        let synthesized = synthesis(
            &f0,
            &spectrogram,
            &aperiodicity,
            self.dio_options.frame_period,
            fs,
        );

        Ok(self.remove_padding(synthesized, input_samples.len()))
    }

    // 非公開メソッド群
    fn add_padding(&self, samples: &[f64]) -> Vec<f64> {
        let padding_length = (self.sample_rate / 2).max(2048);
        let mut padded = Vec::with_capacity(samples.len() + padding_length * 2);
        padded.extend(vec![0.0; padding_length]);
        padded.extend(samples);
        padded.extend(vec![0.0; padding_length]);
        padded
    }

    fn remove_padding(&self, samples: Vec<f64>, original_length: usize) -> Vec<f64> {
        let padding_length = (self.sample_rate / 2).max(2048);
        let start = padding_length.min(samples.len());
        let end = (start + original_length).min(samples.len());
        
        if start >= end {
            samples[..original_length.min(samples.len())].to_vec()
        } else {
            samples[start..end].to_vec()
        }
    }

    fn apply_effects(&self, f0: &mut [f64], spectrogram: &mut [Vec<f64>]) {
        self.convert_f0_to_noise(f0);
        for frame in spectrogram {
            self.process_spectral_frame(frame);
        }
    }

    fn convert_f0_to_noise(&self, f0: &mut [f64]) {
        let mut rng = rand::thread_rng();
        match self.noise_type {
            NoiseType::White => self.generate_white_noise(f0, &mut rng),
            NoiseType::Pink => self.generate_pink_noise(f0, &mut rng),
            NoiseType::Velvet => self.generate_velvet_noise(f0, &mut rng),
        }
    }
    
    fn generate_white_noise(&self, f0: &mut [f64], rng: &mut ThreadRng) {
        for x in f0.iter_mut() {
            *x = rng.gen_range(-1.0..1.0);
        }
    }

    fn generate_pink_noise(&self, f0: &mut [f64], rng: &mut ThreadRng) {
        let mut state = PinkNoiseState::new();
        for x in f0.iter_mut() {
            let mut sum = 0.0;
            
            if state.index % 2 == 0 { state.values[0] = rng.gen_range(-1.0..1.0); }
            if state.index % 4 == 0 { state.values[1] = rng.gen_range(-1.0..1.0); }
            if state.index % 8 == 0 { state.values[2] = rng.gen_range(-1.0..1.0); }
            if state.index % 16 == 0 { state.values[3] = rng.gen_range(-1.0..1.0); }
            if state.index % 32 == 0 { state.values[4] = rng.gen_range(-1.0..1.0); }
            if state.index % 64 == 0 { state.values[5] = rng.gen_range(-1.0..1.0); }
            if state.index % 128 == 0 { state.values[6] = rng.gen_range(-1.0..1.0); }
            
            for &val in &state.values {
                sum += val;
            }
            
            *x = sum / 7.0;
            state.index = (state.index + 1) % 128;
        }
    }

    /// ベルベットノイズ生成
    fn generate_velvet_noise<R: Rng>(&self, f0: &mut [f64], rng: &mut R) {
        let mut state = VelvetNoiseState::new();
        let sample_rate = self.sample_rate;
        
        for x in f0.iter_mut() {
            *x = 0.0;
            
            if rng.gen_ratio(1, VELVET_DENSITY as u32) {
                let min_interval = (sample_rate as f64 / 1000.0 * MIN_PULSE_INTERVAL as f64) as usize;
                
                if state.last_pulse_position >= min_interval {
                    *x = if rng.r#gen() { 1.0 } else { -1.0 };
                    state.last_pulse_position = 0;
                }
            }
            state.last_pulse_position += 1;
        }
        
        self.apply_cosine_phase(f0);
    }

    /// 余弦位相フィルタ適用
    fn apply_cosine_phase(&self, signal: &mut [f64]) {
        let n = signal.len();
        let mut spectrum: Vec<f64> = vec![0.0; n];
        
        // 余弦位相フィルタ生成
        for k in 0..n {
            let phase = (2.0 * PI * k as f64 / n as f64).cos();
            spectrum[k] = phase;
        }
        
        // FFT畳み込み
        let mut buffer = signal.to_vec();
        for i in 0..n {
            buffer[i] *= spectrum[i];
        }
        
        // 正規化
        let max = buffer.iter().fold(f64::MIN, |a, &b| a.max(b.abs()));
        for (i, x) in signal.iter_mut().enumerate() {
            *x = buffer[i] / max;
        }
    }

    fn process_spectral_frame(&self, frame: &mut [f64]) {
        let sr = self.sample_rate;
        let fft_size = (frame.len() - 1) * 2;

        let original = frame.to_vec();
        let max_freq = sr as f64 / 2.0 * 0.95;
        for (i, val) in frame.iter_mut().enumerate() {
            let f_i = i as f64 * sr as f64 / fft_size as f64;
            let new_f = self.calculate_shifted_frequency(f_i, max_freq);
            let new_bin = (new_f * fft_size as f64) / sr as f64;
            let j = new_bin.floor() as usize;
            let frac = new_bin - j as f64;

            *val = if j >= original.len() - 1 {
                original[original.len() - 1]
            } else {
                original[j] * (1.0 - frac) + original[j + 1] * frac
            };
        }

        for (i, val) in frame.iter_mut().enumerate() {
            let f = i as f64 * sr as f64 / fft_size as f64;
            
            let lpf_weight = if f > 1350.0 {
                1.0
            } else if f > 550.0 {
                ((f - 550.0) / 800.0).powf(std::f64::consts::E)
            } else {
                0.0
            };
            
            let hpf_weight = if f < 1000.0 {
                1.0
            } else if f < 10000.0 {
                1.0 + (f - 1000.0) / 9000.0
            } else {
                2.0
            };

            *val *= lpf_weight * hpf_weight;
            *val = val.max(1e-8);
        }
    }

    fn calculate_shifted_frequency(&self, f: f64, max_freq: f64) -> f64 {
        let erb = hz_to_erb(f);
        let target_erb = self.formant_shift_mapping(erb);
        erb_to_hz(target_erb).min(max_freq)
    }

    fn formant_shift_mapping(&self, erb: f64) -> f64 {
        let original_points = [0.0, hz_to_erb(1100.0), hz_to_erb(1600.0), hz_to_erb(20000.0)];
        let target_points = [0.0, hz_to_erb(1000.0), hz_to_erb(1600.0), hz_to_erb(20000.0)];
        
        for i in 0..original_points.len()-1 {
            if erb >= original_points[i] && erb <= original_points[i+1] {
                let t = (erb - original_points[i]) / (original_points[i+1] - original_points[i]);
                return target_points[i] + t * (target_points[i+1] - target_points[i]);
            }
        }
        target_points[target_points.len()-1]
    }
}

// ユーティリティ関数
fn hz_to_erb(hz: f64) -> f64 {
    21.4 * (0.00437 * hz + 1.0).ln() / 10f64.ln()
}

fn erb_to_hz(erb: f64) -> f64 {
    (10f64.powf(erb / 21.4) - 1.0) / 0.00437
}

// オプション構造体
struct DioOptions {
    f0_floor: f64,
    f0_ceil: f64,
    frame_period: f64,
}

impl Default for DioOptions {
    fn default() -> Self {
        Self {
            f0_floor: 50.0,
            f0_ceil: 600.0,
            frame_period: 5.0,
        }
    }
}

impl DioOptions {
    fn to_world_options(&self) -> DioOption {
        let mut opt = DioOption::new();
        opt.f0_floor = self.f0_floor;
        opt.f0_ceil = self.f0_ceil;
        opt.frame_period = self.frame_period;
        opt
    }
}

struct CheapTrickOptions {
    q1: f64,
    fft_size: i32,
}

impl Default for CheapTrickOptions {
    fn default() -> Self {
        Self {
            q1: -0.15,
            fft_size: 0,
        }
    }
}

impl CheapTrickOptions {
    fn to_world_options(&self, fs: i32) -> CheapTrickOption {
        let mut opt = CheapTrickOption::new(fs);
        opt.q1 = self.q1;
        opt.fft_size = self.fft_size;
        opt
    }
}

struct D4COptions {
    threshold: f64,
}

impl Default for D4COptions {
    fn default() -> Self {
        Self { threshold: 0.85 }
    }
}

impl D4COptions {
    fn to_world_options(&self) -> D4COption {
        let mut opt = D4COption::new();
        opt.threshold = self.threshold;
        opt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Helper function to generate a sine wave for testing
    fn generate_sine_wave(freq: f64, sample_rate: usize, duration_secs: f64) -> Vec<f64> {
        let total_samples = (sample_rate as f64 * duration_secs) as usize;
        (0..total_samples)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate as f64).sin())
            .collect()
    }

    #[test]
    fn test_processor_initialization() {
        let sample_rate = 48000;
        let processor = PhantomSilhouetteProcessor::new(sample_rate);

        assert_eq!(processor.sample_rate, sample_rate);
        assert_eq!(processor.noise_type, NoiseType::White);
        // Default configurations
        assert_eq!(processor.dio_options.f0_floor, 50.0);
        assert_eq!(processor.dio_options.f0_ceil, 600.0);
        assert_eq!(processor.dio_options.frame_period, 5.0);
        assert_eq!(processor.cheaptrick_options.q1, -0.15);
        assert_eq!(processor.cheaptrick_options.fft_size, 0);
        assert_eq!(processor.d4c_options.threshold, 0.85);
    }

    #[test]
    fn test_set_noise_type() {
        let sample_rate = 44100;
        let mut processor = PhantomSilhouetteProcessor::new(sample_rate);

        processor.set_noise_type(NoiseType::Pink);
        assert_eq!(processor.noise_type, NoiseType::Pink);

        processor.set_noise_type(NoiseType::Velvet);
        assert_eq!(processor.noise_type, NoiseType::Velvet);
    }

    #[test]
    fn test_configure_dio() {
        let sample_rate = 44100;
        let mut processor = PhantomSilhouetteProcessor::new(sample_rate);

        processor.configure_dio(60.0, 500.0, 10.0);
        assert_eq!(processor.dio_options.f0_floor, 60.0);
        assert_eq!(processor.dio_options.f0_ceil, 500.0);
        assert_eq!(processor.dio_options.frame_period, 10.0);
    }

    #[test]
    fn test_configure_cheaptrick() {
        let sample_rate = 44100;
        let mut processor = PhantomSilhouetteProcessor::new(sample_rate);

        processor.configure_cheaptrick(-0.2, 2048);
        assert_eq!(processor.cheaptrick_options.q1, -0.2);
        assert_eq!(processor.cheaptrick_options.fft_size, 2048);
    }

    #[test]
    fn test_configure_d4c() {
        let sample_rate = 44100;
        let mut processor = PhantomSilhouetteProcessor::new(sample_rate);

        processor.configure_d4c(0.9);
        assert_eq!(processor.d4c_options.threshold, 0.9);
    }

    #[test]
    fn test_process_with_white_noise() {
        let sample_rate = 48000;
        let processor = PhantomSilhouetteProcessor::new(sample_rate);
        let input_samples = generate_sine_wave(440.0, sample_rate, 1.0);

        let result = processor.process(&input_samples);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), input_samples.len());
        // 追加のアサーションをここに追加できます
    }

    #[test]
    fn test_process_with_pink_noise() {
        let sample_rate = 48000;
        let mut processor = PhantomSilhouetteProcessor::new(sample_rate);
        processor.set_noise_type(NoiseType::Pink);
        let input_samples = generate_sine_wave(440.0, sample_rate, 1.0);

        let result = processor.process(&input_samples);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), input_samples.len());
        // 追加のアサーションをここに追加できます
    }

    #[test]
    fn test_process_with_velvet_noise() {
        let sample_rate = 48000;
        let mut processor = PhantomSilhouetteProcessor::new(sample_rate);
        processor.set_noise_type(NoiseType::Velvet);
        let input_samples = generate_sine_wave(440.0, sample_rate, 1.0);

        let result = processor.process(&input_samples);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), input_samples.len());
        // 追加のアサーションをここに追加できます
    }

    #[test]
    fn test_white_noise_generation() {
        let sample_rate = 48000;
        let processor = PhantomSilhouetteProcessor::new(sample_rate);
        let mut f0 = vec![0.0; 1000];
        let mut rng = rand::thread_rng();

        processor.generate_white_noise(&mut f0, &mut rng);
        // 統計的特性をテスト
        let mean: f64 = f0.iter().sum::<f64>() / f0.len() as f64;
        let variance: f64 = f0.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / f0.len() as f64;

        // ホワイトノイズの平均は0に近いこと
        assert!(
            mean.abs() < 0.1,
            "ホワイトノイズの平均が0に近くありません: {}",
            mean
        );
        // 分散は-1から1の範囲の一様分布の場合約1/3であること
        assert!(
            (variance - (1.0 / 3.0)).abs() < 0.1,
            "ホワイトノイズの分散が期待値から外れています: {}",
            variance
        );
    }

    #[test]
    fn test_pink_noise_generation() {
        let sample_rate = 48000;
        let mut processor = PhantomSilhouetteProcessor::new(sample_rate);
        processor.set_noise_type(NoiseType::Pink);
        let mut f0 = vec![0.0; 128];
        let mut rng = rand::thread_rng();

        processor.generate_pink_noise(&mut f0, &mut rng);

        // ピンクノイズの値が期待範囲内であることを確認
        for &x in &f0 {
            assert!(
                x >= -1.0 && x <= 1.0,
                "ピンクノイズの値が範囲外です: {}",
                x
            );
        }

        // 追加のスペクトルテストをここに追加できます
    }

    #[test]
    fn test_velvet_noise_generation() {
        let sample_rate = 48000;
        let mut processor = PhantomSilhouetteProcessor::new(sample_rate);
        processor.set_noise_type(NoiseType::Velvet);
        let mut f0 = vec![0.0; 1000];
        let mut rng = StdRng::seed_from_u64(42); // 再現性のためシードを設定

        processor.generate_velvet_noise(&mut f0, &mut rng);

        // ベルベットノイズはスパースな非ゼロ値を持つこと
        let non_zero_count = f0.iter().filter(|&&x| x != 0.0).count();
        let expected_max = 1000 / VELVET_DENSITY;
        assert!(
            non_zero_count <= expected_max + 10,
            "非ゼロ値の数が期待値を超えています: 非ゼロ数={}, 期待最大数={}",
            non_zero_count,
            expected_max + 10
        );

        // パルス間隔が最小間隔以上であることを確認
        let mut last_pulse = None;
        for (i, &x) in f0.iter().enumerate() {
            if x != 0.0 {
                if let Some(last) = last_pulse {
                    let interval = i - last;
                    assert!(
                        interval >= MIN_PULSE_INTERVAL,
                        "パルス間隔が最小間隔より短い: 間隔={} < 最小間隔={}",
                        interval,
                        MIN_PULSE_INTERVAL
                    );
                }
                last_pulse = Some(i);
            }
        }
    }

    #[test]
    fn test_formant_shift_mapping() {
        // formant_shift_mapping 関数のテスト
        let processor = PhantomSilhouetteProcessor::new(48000);

        let test_frequencies = vec![0.0, 500.0, 1100.0, 1350.0, 1600.0, 1800.0, 20000.0, 25000.0];
        let mut test_cases = Vec::new();

        for hz in test_frequencies {
            let input_erb = hz_to_erb(hz);
            let expected_erb = calculate_expected_formant_shift(hz);
            test_cases.push((input_erb, expected_erb));
        }

        for (input, expected) in test_cases {
            let output = processor.formant_shift_mapping(input);
            assert!(
                (output - expected).abs() < 1e-6,
                "input_erb={}, expected_erb={}, got={}",
                input,
                expected,
                output
            );
        }
    }

    /// マッピングロジックに基づいて期待されるERB値を計算
    fn calculate_expected_formant_shift(hz: f64) -> f64 {
        let erb = hz_to_erb(hz);
        let original_points = [
            hz_to_erb(0.0),
            hz_to_erb(1100.0),
            hz_to_erb(1600.0),
            hz_to_erb(20000.0),
        ];
        let target_points = [
            hz_to_erb(0.0),
            hz_to_erb(1000.0),
            hz_to_erb(1600.0),
            hz_to_erb(20000.0),
        ];

        for i in 0..original_points.len() - 1 {
            if erb >= original_points[i] && erb <= original_points[i + 1] {
                let t = (erb - original_points[i]) / (original_points[i + 1] - original_points[i]);
                return target_points[i] + t * (target_points[i + 1] - target_points[i]);
            }
        }
        // 範囲外の場合は最後のターゲットポイントにクランプ
        target_points[target_points.len() - 1]
    }

    #[test]
    fn test_hz_to_erb_and_back() {
        // hz_to_erb と erb_to_hz 関数のテスト
        let frequencies = vec![0.0, 100.0, 500.0, 1000.0, 1600.0, 20000.0];
        for &hz in &frequencies {
            let erb = hz_to_erb(hz);
            let converted_hz = erb_to_hz(erb);
            if hz == 0.0 {
                assert_eq!(
                    converted_hz, 0.0,
                    "hz=0.0 の場合、converted_hz は 0.0 でなければなりません。実際の値: {}",
                    converted_hz
                );
            } else {
                assert!(
                    (converted_hz - hz).abs() < 1.0,
                    "hz={}, converted_hz={} が一致しません",
                    hz,
                    converted_hz
                );
            }
        }
    }

    #[test]
    fn test_add_and_remove_padding() {
        let sample_rate = 48000;
        let processor = PhantomSilhouetteProcessor::new(sample_rate);
        let input_samples = generate_sine_wave(440.0, sample_rate, 1.0);

        let padded = processor.add_padding(&input_samples);
        let expected_padding = (sample_rate / 2).max(2048).min(1_000_000); // 上限を設定
        assert_eq!(
            padded.len(),
            input_samples.len() + 2 * expected_padding,
            "パディング後の長さが一致しません"
        );
        assert!(
            padded[..expected_padding].iter().all(|&x| x == 0.0),
            "先頭のパディングがゼロではありません"
        );
        assert!(
            padded[padded.len() - expected_padding..].iter().all(|&x| x == 0.0),
            "末尾のパディングがゼロではありません"
        );

        let removed = processor.remove_padding(padded, input_samples.len());
        assert_eq!(removed.len(), input_samples.len(), "パディング除去後の長さが一致しません");
        assert_eq!(removed, input_samples, "パディング除去後のサンプルが元のサンプルと一致しません");
    }

    #[test]
    fn test_apply_cosine_phase() {
        let sample_rate = 48000;
        let processor = PhantomSilhouetteProcessor::new(sample_rate);
        let mut signal = vec![1.0, -1.0, 1.0, -1.0]; // 単純な交互信号

        processor.apply_cosine_phase(&mut signal);
        // 余弦位相フィルタの期待される出力を計算
        let expected_signal = vec![1.0, 0.0, -1.0, 0.0];
        for (output, expected) in signal.iter().zip(expected_signal.iter()) {
            assert!(
                (*output - *expected).abs() < 1e-6,
                "出力: {}, 期待値: {}",
                output,
                expected
            );
        }
    }
}
