use anyhow::Result;
use ndarray::{Array, Array3, Axis, CowArray};
use ort::{
    execution_providers::CPUExecutionProviderOptions, Environment, ExecutionProvider,
    GraphOptimizationLevel, SessionBuilder, Value,
};
use std::sync::Arc;

pub struct HrtfProcessor {
    session: ort::Session,
}

impl HrtfProcessor {
    pub fn new(model_path: &str) -> Result<Self> {
        let environment = Environment::builder()
            .with_execution_providers([ExecutionProvider::CPU(
                CPUExecutionProviderOptions::default(),
            )])
            .build()?;

        let environment = Arc::new(environment);

        Ok(Self {
            session: SessionBuilder::new(&environment)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_model_from_file(model_path)?,
        })
    }

    pub fn apply_hrtf(
        &mut self,
        audio: &[f32],
        azimuth: f32,
        elevation: f32,
    ) -> Result<Vec<[f32; 2]>> {
        // 入力テンソルの準備
        let audio_array: CowArray<f32, _> =
            Array3::from_shape_vec((1, 1, audio.len()), audio.to_vec())?.into_dyn().into();
        let azimuth_array: CowArray<f32, _> = Array::from_vec(vec![azimuth]).into_dyn().into();
        let elevation_array: CowArray<f32, _> = Array::from_vec(vec![elevation]).into_dyn().into();

        // 推論実行
        let inputs = vec![
            Value::from_array(self.session.allocator(), &audio_array)?,
            Value::from_array(self.session.allocator(), &azimuth_array)?,
            Value::from_array(self.session.allocator(), &elevation_array)?,
        ];

        let outputs = self.session.run(inputs)?;
        let output_tensor = outputs[0].try_extract()?;
        let mut output_view = output_tensor.view().to_owned();

        // 形状変換
        output_view.swap_axes(1, 2);

        Ok(output_view
            .clone()
            .into_shape((output_view.shape()[1], 2))?
            .axis_iter(Axis(0))
            .map(|row| [row[0], row[1]])
            .collect())
    }
}