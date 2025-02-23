// src/crossfade.rs

use anyhow::Result;
use std::f64::consts::PI;

/// 音声波形を等電力クロスフェードするための構造体
pub struct CrossFader {
    volume1_db: f64,
    volume2_db: f64,
}

#[allow(dead_code)]
impl CrossFader {
    /// 新しい CrossFader を作成します。
    ///
    /// # 引数
    ///
    /// * `volume1_db` - 入力音声1の音量（デシベル単位）
    /// * `volume2_db` - 入力音声2の音量（デシベル単位）
    ///
    /// # 戻り値
    ///
    /// CrossFader 構造体のインスタンス
    pub fn new(volume1_db: f64, volume2_db: f64) -> Self {
        Self {
            volume1_db,
            volume2_db,
        }
    }

    /// モノラル音声を等電力クロスフェードします（f64用）。
    ///
    /// # 引数
    ///
    /// * `input1` - 入力音声1のサンプル配列（モノラル）
    /// * `input2` - 入力音声2のサンプル配列（モノラル）
    ///
    /// # 戻り値
    ///
    /// クロスフェードされたサンプル配列（モノラル）
    pub fn crossfade_mono(&self, input1: &[f64], input2: &[f64]) -> Result<Vec<f64>> {
        // デシベルをリニアスケールに変換
        let volume1_linear = 10f64.powf(self.volume1_db / 20.0);
        let volume2_linear = 10f64.powf(self.volume2_db / 20.0);

        // 2つの入力音声の長さの最大値を取得
        let max_len = input1.len().max(input2.len());

        // 出力用のベクターを初期化
        let mut output = Vec::with_capacity(max_len);

        for i in 0..max_len {
            // 各入力音声のサンプルを取得（範囲外は0.0）
            let sample1 = if i < input1.len() { input1[i] * volume1_linear } else { 0.0 };
            let sample2 = if i < input2.len() { input2[i] * volume2_linear } else { 0.0 };

            // フェードの進行度を0.0から1.0に正規化
            let fade_progress = if max_len > 1 {
                i as f64 / (max_len - 1) as f64
            } else {
                1.0
            };

            // θを計算（0からπ/2に変化）
            let theta = fade_progress * (PI / 2.0);

            // 等電力クロスフェードのウェイトを計算
            let weight1 = theta.cos();
            let weight2 = theta.sin();

            // クロスフェードされたサンプルを計算
            let crossfaded_sample = sample1 * weight1 + sample2 * weight2;

            output.push(crossfaded_sample);
        }

        Ok(output)
    }

    /// ステレオ音声を等電力クロスフェードします（f64用）。
    ///
    /// # 引数
    ///
    /// * `input1` - 入力音声1のサンプル配列（ステレオ）
    /// * `input2` - 入力音声2のサンプル配列（ステレオ）
    ///
    /// # 戻り値
    ///
    /// クロスフェードされたサンプル配列（ステレオ）
    pub fn crossfade_stereo(&self, input1: &[[f64; 2]], input2: &[[f64; 2]]) -> Result<Vec<[f64; 2]>> {
        // デシベルをリニアスケールに変換
        let volume1_linear = 10f64.powf(self.volume1_db / 20.0);
        let volume2_linear = 10f64.powf(self.volume2_db / 20.0);

        // 2つの入力音声の長さの最大値を取得
        let max_len = input1.len().max(input2.len());

        // 出力用のベクターを初期化
        let mut output = Vec::with_capacity(max_len);

        for i in 0..max_len {
            // 各入力音声のサンプルを取得（範囲外は0.0）
            let sample1_left = if i < input1.len() { input1[i][0] * volume1_linear } else { 0.0 };
            let sample1_right = if i < input1.len() { input1[i][1] * volume1_linear } else { 0.0 };
            let sample2_left = if i < input2.len() { input2[i][0] * volume2_linear } else { 0.0 };
            let sample2_right = if i < input2.len() { input2[i][1] * volume2_linear } else { 0.0 };

            // フェードの進行度を0.0から1.0に正規化
            let fade_progress = if max_len > 1 {
                i as f64 / (max_len - 1) as f64
            } else {
                1.0
            };

            // θを計算（0からπ/2に変化）
            let theta = fade_progress * (PI / 2.0);

            // 等電力クロスフェードのウェイトを計算
            let weight1 = theta.cos();
            let weight2 = theta.sin();

            // クロスフェードされたサンプルを計算
            let crossfaded_left = sample1_left * weight1 + sample2_left * weight2;
            let crossfaded_right = sample1_right * weight1 + sample2_right * weight2;

            output.push([crossfaded_left, crossfaded_right]);
        }

        Ok(output)
    }

    /// モノラル音声を等電力クロスフェードします（f32用）。
    ///
    /// # 引数
    ///
    /// * `input1` - 入力音声1のサンプル配列（モノラル）
    /// * `input2` - 入力音声2のサンプル配列（モノラル）
    ///
    /// # 戻り値
    ///
    /// クロスフェードされたサンプル配列（モノラル）
    pub fn crossfade_mono_f32(&self, input1: &[f32], input2: &[f32]) -> Result<Vec<f32>> {
        // デシベルをリニアスケールに変換
        let volume1_linear = 10f32.powf((self.volume1_db / 20.0) as f32);
        let volume2_linear = 10f32.powf((self.volume2_db / 20.0) as f32);

        // 2つの入力音声の長さの最大値を取得
        let max_len = input1.len().max(input2.len());

        // 出力用のベクターを初期化
        let mut output = Vec::with_capacity(max_len);

        for i in 0..max_len {
            // 各入力音声のサンプルを取得（範囲外は0.0）
            let sample1 = if i < input1.len() { input1[i] * volume1_linear } else { 0.0 };
            let sample2 = if i < input2.len() { input2[i] * volume2_linear } else { 0.0 };

            // フェードの進行度を0.0から1.0に正規化
            let fade_progress = if max_len > 1 {
                i as f32 / ((max_len - 1) as f32)
            } else {
                1.0
            };

            // θを計算（0からπ/2に変化）
            let theta = fade_progress * (PI / 2.0) as f32;

            // 等電力クロスフェードのウェイトを計算
            let weight1 = theta.cos();
            let weight2 = theta.sin();

            // クロスフェードされたサンプルを計算
            let crossfaded_sample = sample1 * weight1 + sample2 * weight2;

            output.push(crossfaded_sample);
        }

        Ok(output)
    }

    /// ステレオ音声を等電力クロスフェードします（f32用）。
    ///
    /// # 引数
    ///
    /// * `input1` - 入力音声1のサンプル配列（ステレオ）
    /// * `input2` - 入力音声2のサンプル配列（ステレオ）
    ///
    /// # 戻り値
    ///
    /// クロスフェードされたサンプル配列（ステレオ）
    pub fn crossfade_stereo_f32(&self, input1: &[[f32; 2]], input2: &[[f32; 2]]) -> Result<Vec<[f32; 2]>> {
        // デシベルをリニアスケールに変換
        let volume1_linear = 10f32.powf((self.volume1_db / 20.0) as f32);
        let volume2_linear = 10f32.powf((self.volume2_db / 20.0) as f32);

        // 2つの入力音声の長さの最大値を取得
        let max_len = input1.len().max(input2.len());

        // 出力用のベクターを初期化
        let mut output = Vec::with_capacity(max_len);

        for i in 0..max_len {
            // 各入力音声のサンプルを取得（範囲外は0.0）
            let sample1_left = if i < input1.len() { input1[i][0] * volume1_linear } else { 0.0 };
            let sample1_right = if i < input1.len() { input1[i][1] * volume1_linear } else { 0.0 };
            let sample2_left = if i < input2.len() { input2[i][0] * volume2_linear } else { 0.0 };
            let sample2_right = if i < input2.len() { input2[i][1] * volume2_linear } else { 0.0 };

            // フェードの進行度を0.0から1.0に正規化
            let fade_progress = if max_len > 1 {
                i as f32 / ((max_len - 1) as f32)
            } else {
                1.0
            };

            // θを計算（0からπ/2に変化）
            let theta = fade_progress * (PI / 2.0) as f32;

            // 等電力クロスフェードのウェイトを計算
            let weight1 = theta.cos();
            let weight2 = theta.sin();

            // クロスフェードされたサンプルを計算
            let crossfaded_left = sample1_left * weight1 + sample2_left * weight2;
            let crossfaded_right = sample1_right * weight1 + sample2_right * weight2;

            output.push([crossfaded_left, crossfaded_right]);
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use rubato::Sample;

    use super::*;

    #[test]
    fn test_crossfade_mono_same_length() {
        let crossfader = CrossFader::new(0.0, 0.0); // 0 dB, 音量変化なし
        let input1 = vec![1.0, 1.0, 1.0, 1.0];
        let input2 = vec![0.0, 0.0, 0.0, 0.0];
        let output = crossfader.crossfade_mono(&input1, &input2).unwrap();
        let expected = vec![
            1.0 * (0.0).cos() + 0.0 * (0.0).sin(),                   // θ=0: weight1=1, weight2=0
            1.0 * ((1.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((1.0 / 3.0) * PI / 2.0).sin(),
            1.0 * ((2.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((2.0 / 3.0) * PI / 2.0).sin(),
            1.0 * (PI / 2.0).cos() + 0.0 * (PI / 2.0).sin(),         // θ=π/2: weight1=0, weight2=1
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "o: {}, e: {}", o, e);
        }
    }

    #[test]
    fn test_crossfade_mono_different_volumes() {
        let crossfader = CrossFader::new(-6.0, 0.0); // 入力1は-6 dB, 入力2は0 dB
        let input1 = vec![1.0, 1.0, 1.0, 1.0];
        let input2 = vec![1.0, 1.0, 1.0, 1.0];
        let output = crossfader.crossfade_mono(&input1, &input2).unwrap();

        // -6 dBは約0.5011872336272722のリニアスケールに対応
        let volume1_linear = 10f64.powf(-6.0 / 20.0);
        let expected = vec![
            (1.0 * volume1_linear) * (0.0).cos() + (1.0) * (0.0).sin(),
            (1.0 * volume1_linear) * ((1.0 / 3.0) * PI / 2.0).cos() + (1.0) * ((1.0 / 3.0) * PI / 2.0).sin(),
            (1.0 * volume1_linear) * ((2.0 / 3.0) * PI / 2.0).cos() + (1.0) * ((2.0 / 3.0) * PI / 2.0).sin(),
            (1.0 * volume1_linear) * (PI / 2.0).cos() + (1.0) * (PI / 2.0).sin(),
        ];

        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "o: {}, e: {}", o, e);
        }
    }

    #[test]
    fn test_crossfade_mono_different_lengths() {
        let crossfader = CrossFader::new(0.0, 0.0); // 0 dB
        let input1 = vec![1.0, 1.0, 1.0];
        let input2 = vec![0.0, 0.0, 0.0, 0.0];
        let output = crossfader.crossfade_mono(&input1, &input2).unwrap();
        let expected = vec![
            1.0 * (0.0).cos() + 0.0 * (0.0).sin(),
            1.0 * ((1.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((1.0 / 3.0) * PI / 2.0).sin(),
            1.0 * ((2.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((2.0 / 3.0) * PI / 2.0).sin(),
            0.0 * (PI / 2.0).cos() + 0.0 * (PI / 2.0).sin(),
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "o: {}, e: {}", o, e);
        }
    }

    #[test]
    fn test_crossfade_mono_with_negative_db() {
        let crossfader = CrossFader::new(-6.0, -6.0);
        let input1 = vec![1.0, 1.0];
        let input2 = vec![1.0, 1.0];
        let output = crossfader.crossfade_mono(&input1, &input2).unwrap();
        let volume_linear = 10f64.powf(-6.0 / 20.0);
        let expected = vec![
            (1.0 * volume_linear) * (0.0).cos() + (1.0 * volume_linear) * (0.0).sin(),
            (1.0 * volume_linear) * ((1.0 / 1.0) * PI / 2.0).cos() + (1.0 * volume_linear) * ((1.0 / 1.0) * PI / 2.0).sin(),
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "o: {}, e: {}", o, e);
        }
    }

    #[test]
    fn test_crossfade_mono_zero_length() {
        let crossfader = CrossFader::new(0.0, 0.0);
        let input1: Vec<f64> = vec![];
        let input2: Vec<f64> = vec![];
        let output = crossfader.crossfade_mono(&input1, &input2).unwrap();
        let expected: Vec<f64> = vec![];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_crossfade_stereo_same_length() {
        let crossfader = CrossFader::new(0.0, 0.0); // 0 dB
        let input1 = vec![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let input2 = vec![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let output = crossfader.crossfade_stereo(&input1, &input2).unwrap();
        let expected = vec![
            [
                1.0 * (0.0).cos() + 0.0 * (0.0).sin(),
                1.0 * (0.0).cos() + 0.0 * (0.0).sin(),
            ],
            [
                1.0 * ((1.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((1.0 / 3.0) * PI / 2.0).sin(),
                1.0 * ((1.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((1.0 / 3.0) * PI / 2.0).sin(),
            ],
            [
                1.0 * ((2.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((2.0 / 3.0) * PI / 2.0).sin(),
                1.0 * ((2.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((2.0 / 3.0) * PI / 2.0).sin(),
            ],
            [
                1.0 * (PI / 2.0).cos() + 0.0 * (PI / 2.0).sin(),
                1.0 * (PI / 2.0).cos() + 0.0 * (PI / 2.0).sin(),
            ],
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!(
                (o[0] - e[0]).abs() < 1e-6 && (o[1] - e[1]).abs() < 1e-6,
                "o: {:?}, e: {:?}",
                o,
                e
            );
        }
    }

    #[test]
    fn test_crossfade_stereo_different_volumes() {
        let crossfader = CrossFader::new(-6.0, 0.0); // 入力1は-6 dB, 入力2は0 dB
        let input1 = vec![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let input2 = vec![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let output = crossfader.crossfade_stereo(&input1, &input2).unwrap();

        // -6 dBは約0.5011872336272722のリニアスケールに対応
        let volume1_linear = 10f64.powf(-6.0 / 20.0);
        let expected = vec![
            [
                (1.0 * volume1_linear) * (0.0).cos() + (1.0) * (0.0).sin(),
                (1.0 * volume1_linear) * (0.0).cos() + (1.0) * (0.0).sin(),
            ],
            [
                (1.0 * volume1_linear) * ((1.0 / 3.0) * PI / 2.0).cos() + (1.0) * ((1.0 / 3.0) * PI / 2.0).sin(),
                (1.0 * volume1_linear) * ((1.0 / 3.0) * PI / 2.0).cos() + (1.0) * ((1.0 / 3.0) * PI / 2.0).sin(),
            ],
            [
                (1.0 * volume1_linear) * ((2.0 / 3.0) * PI / 2.0).cos() + (1.0) * ((2.0 / 3.0) * PI / 2.0).sin(),
                (1.0 * volume1_linear) * ((2.0 / 3.0) * PI / 2.0).cos() + (1.0) * ((2.0 / 3.0) * PI / 2.0).sin(),
            ],
            [
                (1.0 * volume1_linear) * (PI / 2.0).cos() + (1.0) * (PI / 2.0).sin(),
                (1.0 * volume1_linear) * (PI / 2.0).cos() + (1.0) * (PI / 2.0).sin(),
            ],
        ];

        for (o, e) in output.iter().zip(expected.iter()) {
            assert!(
                (o[0] - e[0]).abs() < 1e-6 && (o[1] - e[1]).abs() < 1e-6,
                "o: {:?}, e: {:?}",
                o,
                e
            );
        }
    }

    #[test]
    fn test_crossfade_stereo_different_lengths() {
        let crossfader = CrossFader::new(0.0, 0.0); // 0 dB
        let input1 = vec![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let input2 = vec![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let output = crossfader.crossfade_stereo(&input1, &input2).unwrap();
        let expected = vec![
            [
                1.0 * (0.0).cos() + 0.0 * (0.0).sin(),
                1.0 * (0.0).cos() + 0.0 * (0.0).sin(),
            ],
            [
                1.0 * ((1.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((1.0 / 3.0) * PI / 2.0).sin(),
                1.0 * ((1.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((1.0 / 3.0) * PI / 2.0).sin(),
            ],
            [
                1.0 * ((2.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((2.0 / 3.0) * PI / 2.0).sin(),
                1.0 * ((2.0 / 3.0) * PI / 2.0).cos() + 0.0 * ((2.0 / 3.0) * PI / 2.0).sin(),
            ],
            [
                0.0 * (PI / 2.0).cos() + 0.0 * (PI / 2.0).sin(),
                0.0 * (PI / 2.0).cos() + 0.0 * (PI / 2.0).sin(),
            ],
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!(
                (o[0] - e[0]).abs() < 1e-6 && (o[1] - e[1]).abs() < 1e-6,
                "o: {:?}, e: {:?}",
                o,
                e
            );
        }
    }

    #[test]
    fn test_crossfade_stereo_with_negative_db() {
        let crossfader = CrossFader::new(-6.0, -6.0);
        let input1 = vec![[1.0, 1.0], [1.0, 1.0]];
        let input2 = vec![[1.0, 1.0], [1.0, 1.0]];
        let output = crossfader.crossfade_stereo(&input1, &input2).unwrap();
        let volume_linear = 10f64.powf(-6.0 / 20.0);
        let expected = vec![
            [
                (1.0 * volume_linear) * (0.0).cos() + (1.0 * volume_linear) * (0.0).sin(),
                (1.0 * volume_linear) * (0.0).cos() + (1.0 * volume_linear) * (0.0).sin(),
            ],
            [
                (1.0 * volume_linear) * ((1.0 / 1.0) * PI / 2.0).cos() + (1.0 * volume_linear) * ((1.0 / 1.0) * PI / 2.0).sin(),
                (1.0 * volume_linear) * ((1.0 / 1.0) * PI / 2.0).cos() + (1.0 * volume_linear) * ((1.0 / 1.0) * PI / 2.0).sin(),
            ],
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!(
                (o[0] - e[0]).abs() < 1e-6 && (o[1] - e[1]).abs() < 1e-6,
                "o: {:?}, e: {:?}",
                o,
                e
            );
        }
    }

    #[test]
    fn test_crossfade_stereo_zero_length() {
        let crossfader = CrossFader::new(0.0, 0.0);
        let input1: Vec<[f64; 2]> = vec![];
        let input2: Vec<[f64; 2]> = vec![];
        let output = crossfader.crossfade_stereo(&input1, &input2).unwrap();
        let expected: Vec<[f64; 2]> = vec![];
        assert_eq!(output, expected);
    }

    // 以下に f32 用のテストケースを追加します

    #[test]
    fn test_crossfade_mono_f32_same_length() {
        let crossfader = CrossFader::new(0.0, 0.0); // 0 dB, 音量変化なし
        let input1 = vec![1.0f32, 1.0, 1.0, 1.0];
        let input2 = vec![0.0f32, 0.0, 0.0, 0.0];
        let output = crossfader.crossfade_mono_f32(&input1, &input2).unwrap();
        let expected = vec![
            1.0 * (0.0f32).cos() + 0.0 * (0.0f32).sin(),                   // θ=0: weight1=1, weight2=0
            1.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            1.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            1.0 * (PI as f32 / 2.0).cos() + 0.0 * (PI as f32 / 2.0).sin(), // θ=π/2: weight1=0, weight2=1
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "o: {}, e: {}", o, e);
        }
    }

    #[test]
    fn test_crossfade_mono_f32_different_volumes() {
        let crossfader = CrossFader::new(-6.0, 0.0); // 入力1は-6 dB, 入力2は0 dB
        let input1 = vec![1.0f32, 1.0, 1.0, 1.0];
        let input2 = vec![1.0f32, 1.0, 1.0, 1.0];
        let output = crossfader.crossfade_mono_f32(&input1, &input2).unwrap();

        // -6 dBは約0.5011872のリニアスケールに対応
        let volume1_linear = 10f32.powf((-6.0f64 / 20.0) as f32);
        let expected = vec![
            (1.0 * volume1_linear) * (0.0f32).cos() + (1.0) * (0.0f32).sin(),
            (1.0 * volume1_linear) * ((1.0f32 / 3.0) * (PI / 2.0) as f32).cos() + (1.0) * ((1.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            (1.0 * volume1_linear) * ((2.0f32 / 3.0) * (PI / 2.0) as f32).cos() + (1.0) * ((2.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            (1.0 * volume1_linear) * (PI as f32 / 2.0).cos() + (1.0) * (PI as f32 / 2.0).sin(),
        ];

        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "o: {}, e: {}", o, e);
        }
    }

    #[test]
    fn test_crossfade_mono_f32_different_lengths() {
        let crossfader = CrossFader::new(0.0, 0.0); // 0 dB
        let input1 = vec![1.0f32, 1.0, 1.0];
        let input2 = vec![0.0f32, 0.0, 0.0, 0.0];
        let output = crossfader.crossfade_mono_f32(&input1, &input2).unwrap();
        let expected = vec![
            1.0 * (0.0f32).cos() + 0.0 * (0.0f32).sin(),
            1.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            1.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            0.0 * (PI as f32 / 2.0).cos() + 0.0 * (PI as f32 / 2.0).sin(),
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "o: {}, e: {}", o, e);
        }
    }

    #[test]
    fn test_crossfade_mono_f32_with_negative_db() {
        let crossfader = CrossFader::new(-6.0, -6.0);
        let input1 = vec![1.0f32, 1.0];
        let input2 = vec![1.0f32, 1.0];
        let output = crossfader.crossfade_mono_f32(&input1, &input2).unwrap();
        let volume_linear = 10f32.powf((-6.0f64 / 20.0) as f32);
        let expected = vec![
            (1.0 * volume_linear) * (0.0f32).cos() + (1.0 * volume_linear) * (0.0f32).sin(),
            (1.0 * volume_linear) * ((1.0f32 / 1.0) * (PI / 2.0) as f32).cos() + (1.0 * volume_linear) * ((1.0f32 / 1.0) * (PI / 2.0) as f32).sin(),
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "o: {}, e: {}", o, e);
        }
    }

    #[test]
    fn test_crossfade_mono_f32_zero_length() {
        let crossfader = CrossFader::new(0.0, 0.0);
        let input1: Vec<f32> = vec![];
        let input2: Vec<f32> = vec![];
        let output = crossfader.crossfade_mono_f32(&input1, &input2).unwrap();
        let expected: Vec<f32> = vec![];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_crossfade_stereo_f32_same_length() {
        let crossfader = CrossFader::new(0.0, 0.0); // 0 dB
        let input1 = vec![[1.0f32, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let input2 = vec![[0.0f32, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let output = crossfader.crossfade_stereo_f32(&input1, &input2).unwrap();
        let expected = vec![
            [
                1.0 * (0.0f32).cos() + 0.0 * (0.0f32).sin(),
                1.0 * (0.0f32).cos() + 0.0 * (0.0f32).sin(),
            ],
            [
                1.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
                1.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            ],
            [
                1.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
                1.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            ],
            [
                1.0 * (PI as f32 / 2.0).cos() + 0.0 * (PI as f32 / 2.0).sin(),
                1.0 * (PI as f32 / 2.0).cos() + 0.0 * (PI as f32 / 2.0).sin(),
            ],
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!(
                (o[0] - e[0]).abs() < 1e-6 && (o[1] - e[1]).abs() < 1e-6,
                "o: {:?}, e: {:?}",
                o,
                e
            );
        }
    }

    #[test]
    fn test_crossfade_stereo_f32_different_volumes() {
        let crossfader = CrossFader::new(-6.0, 0.0); // 入力1は-6 dB, 入力2は0 dB
        let input1 = vec![[1.0f32, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let input2 = vec![[1.0f32, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let output = crossfader.crossfade_stereo_f32(&input1, &input2).unwrap();

        // -6 dBは約0.5011872のリニアスケールに対応
        let volume1_linear = 10f32.powf((-6.0f64 / 20.0) as f32);
        let expected = vec![
            [
                (1.0 * volume1_linear) * (0.0f32).cos() + (1.0 * volume1_linear) * (0.0f32).sin(),
                (1.0 * volume1_linear) * (0.0f32).cos() + (1.0 * volume1_linear) * (0.0f32).sin(),
            ],
            [
                (1.0 * volume1_linear) * ((1.0f32 / 3.0) * (PI / 2.0) as f32).cos() + (1.0 * volume1_linear) * ((1.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
                (1.0 * volume1_linear) * ((1.0f32 / 3.0) * (PI / 2.0) as f32).cos() + (1.0 * volume1_linear) * ((1.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            ],
            [
                (1.0 * volume1_linear) * ((2.0f32 / 3.0) * (PI / 2.0) as f32).cos() + (1.0 * volume1_linear) * ((2.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
                (1.0 * volume1_linear) * ((2.0f32 / 3.0) * (PI / 2.0) as f32).cos() + (1.0 * volume1_linear) * ((2.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            ],
            [
                (1.0 * volume1_linear) * (PI as f32 / 2.0).cos() + (1.0 * volume1_linear) * (PI as f32 / 2.0).sin(),
                (1.0 * volume1_linear) * (PI as f32 / 2.0).cos() + (1.0 * volume1_linear) * (PI as f32 / 2.0).sin(),
            ],
        ];

        for (o, e) in output.iter().zip(expected.iter()) {
            assert!(
                (o[0] - e[0]).abs() < 1e-6 && (o[1] - e[1]).abs() < 1e-6,
                "o: {:?}, e: {:?}",
                o,
                e
            );
        }
    }

    #[test]
    fn test_crossfade_stereo_f32_different_lengths() {
        let crossfader = CrossFader::new(0.0, 0.0); // 0 dB
        let input1 = vec![[1.0f32, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let input2 = vec![[0.0f32, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let output = crossfader.crossfade_stereo_f32(&input1, &input2).unwrap();
        let expected = vec![
            [
                1.0 * (0.0f32).cos() + 0.0 * (0.0f32).sin(),
                1.0 * (0.0f32).cos() + 0.0 * (0.0f32).sin(),
            ],
            [
                1.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
                1.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((1.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            ],
            [
                1.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
                1.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).cos() + 0.0 * ((2.0f32 / 3.0) * (PI / 2.0) as f32).sin(),
            ],
            [
                0.0 * (PI as f32 / 2.0).cos() + 0.0 * (PI as f32 / 2.0).sin(),
                0.0 * (PI as f32 / 2.0).cos() + 0.0 * (PI as f32 / 2.0).sin(),
            ],
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!(
                (o[0] - e[0]).abs() < 1e-6 && (o[1] - e[1]).abs() < 1e-6,
                "o: {:?}, e: {:?}",
                o,
                e
            );
        }
    }

    #[test]
    fn test_crossfade_stereo_f32_with_negative_db() {
        let crossfader = CrossFader::new(-6.0, -6.0);
        let input1 = vec![[1.0f32, 1.0], [1.0, 1.0]];
        let input2 = vec![[1.0f32, 1.0], [1.0, 1.0]];
        let output = crossfader.crossfade_stereo_f32(&input1, &input2).unwrap();
        let volume_linear = 10f32.powf((-6.0f64 / 20.0) as f32);
        let expected = vec![
            [
                (1.0 * volume_linear) * (0.0f32).cos() + (1.0 * volume_linear) * (0.0f32).sin(),
                (1.0 * volume_linear) * (0.0f32).cos() + (1.0 * volume_linear) * (0.0f32).sin(),
            ],
            [
                (1.0 * volume_linear) * ((1.0f32 / 1.0) * (PI / 2.0) as f32).cos() + (1.0 * volume_linear) * ((1.0f32 / 1.0) * (PI / 2.0) as f32).sin(),
                (1.0 * volume_linear) * ((1.0f32 / 1.0) * (PI / 2.0) as f32).cos() + (1.0 * volume_linear) * ((1.0f32 / 1.0) * (PI / 2.0) as f32).sin(),
            ],
        ];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!(
                (o[0] - e[0]).abs() < 1e-6 && (o[1] - e[1]).abs() < 1e-6,
                "o: {:?}, e: {:?}",
                o,
                e
            );
        }
    }

    #[test]
    fn test_crossfade_stereo_f32_zero_length() {
        let crossfader = CrossFader::new(0.0, 0.0);
        let input1: Vec<[f32; 2]> = vec![];
        let input2: Vec<[f32; 2]> = vec![];
        let output = crossfader.crossfade_stereo_f32(&input1, &input2).unwrap();
        let expected: Vec<[f32; 2]> = vec![];
        assert_eq!(output, expected);
    }
}