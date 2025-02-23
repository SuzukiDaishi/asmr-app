// src/distance_adjust.rs

use biquad::{Biquad, Coefficients, DirectForm1, ToHertz, Q_BUTTERWORTH_F32, Type};

#[allow(dead_code)]
pub struct DistanceAdjustProcessor {
    sample_rate: usize,
    gain: Option<f32>,
    // 左チャンネル用フィルター
    low_shelf_left: DirectForm1<f32>,
    high_shelf_left: DirectForm1<f32>,
    reverb_reduction_left: DirectForm1<f32>,
    // 右チャンネル用フィルター
    low_shelf_right: DirectForm1<f32>,
    high_shelf_right: DirectForm1<f32>,
    reverb_reduction_right: DirectForm1<f32>,
}

#[allow(dead_code)]
impl DistanceAdjustProcessor {
    /// 新しい DistanceAdjustProcessor を作成します。
    ///
    /// # 引数
    ///
    /// * `sample_rate` - サンプルレート（Hz）
    /// * `gain` - オプションのゲイン値（f32）。`Some(gain)` でゲインを適用し、`None` で適用しません。
    ///
    /// # 戻り値
    ///
    /// フィルター生成に失敗した場合はエラーを返します。
    pub fn new(sample_rate: usize, gain: Option<f32>) -> Result<Self, biquad::Errors> {
        // 左チャンネルのフィルター設定
        let low_coeffs_left = Coefficients::<f32>::from_params(
            Type::LowShelf(6.0), // +6dB
            sample_rate.hz(),
            100.0.hz(),
            Q_BUTTERWORTH_F32,
        )?;
        let low_shelf_left = DirectForm1::<f32>::new(low_coeffs_left);

        let high_coeffs_left = Coefficients::<f32>::from_params(
            Type::HighShelf(3.0), // +3dB
            sample_rate.hz(),
            5000.0.hz(),
            Q_BUTTERWORTH_F32,
        )?;
        let high_shelf_left = DirectForm1::<f32>::new(high_coeffs_left);

        let reverb_coeffs_left = Coefficients::<f32>::from_params(
            Type::HighPass,
            sample_rate.hz(),
            200.0.hz(),
            Q_BUTTERWORTH_F32,
        )?;
        let reverb_reduction_left = DirectForm1::<f32>::new(reverb_coeffs_left);

        // 右チャンネルのフィルター設定
        let low_coeffs_right = Coefficients::<f32>::from_params(
            Type::LowShelf(6.0), // +6dB
            sample_rate.hz(),
            100.0.hz(),
            Q_BUTTERWORTH_F32,
        )?;
        let low_shelf_right = DirectForm1::<f32>::new(low_coeffs_right);

        let high_coeffs_right = Coefficients::<f32>::from_params(
            Type::HighShelf(3.0), // +3dB
            sample_rate.hz(),
            5000.0.hz(),
            Q_BUTTERWORTH_F32,
        )?;
        let high_shelf_right = DirectForm1::<f32>::new(high_coeffs_right);

        let reverb_coeffs_right = Coefficients::<f32>::from_params(
            Type::HighPass,
            sample_rate.hz(),
            200.0.hz(),
            Q_BUTTERWORTH_F32,
        )?;
        let reverb_reduction_right = DirectForm1::<f32>::new(reverb_coeffs_right);

        Ok(Self {
            sample_rate,
            gain,
            low_shelf_left,
            high_shelf_left,
            reverb_reduction_left,
            low_shelf_right,
            high_shelf_right,
            reverb_reduction_right,
        })
    }

    /// 音声データを処理して近接感を高めます。
    ///
    /// # 引数
    ///
    /// * `samples` - 処理対象のステレオ音声サンプル（左と右のペア）
    ///
    /// # 戻り値
    ///
    /// 処理後のステレオ音声サンプル
    pub fn process(&mut self, samples: &[[f32; 2]]) -> Vec<[f32; 2]> {
        samples
            .iter()
            .map(|&[left, right]| {
                // 左チャンネルの処理
                let processed_left = if let Some(g) = self.gain {
                    left * g
                } else {
                    left
                };
                let processed_left = self.low_shelf_left.run(processed_left);
                let processed_left = self.high_shelf_left.run(processed_left);
                let processed_left = self.reverb_reduction_left.run(processed_left);

                // 右チャンネルの処理
                let processed_right = if let Some(g) = self.gain {
                    right * g
                } else {
                    right
                };
                let processed_right = self.low_shelf_right.run(processed_right);
                let processed_right = self.high_shelf_right.run(processed_right);
                let processed_right = self.reverb_reduction_right.run(processed_right);

                [processed_left, processed_right]
            })
            .collect()
    }
}
