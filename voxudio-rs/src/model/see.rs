use {
    super::get_session_builder,
    crate::OperationError,
    ndarray::Array2,
    ort::{
        inputs,
        session::{RunOptions, Session},
        value::TensorRef,
    },
    std::{array::from_fn, path::Path},
};

/// 说话人嵌入提取器，用于从音频中提取说话人特征（音色）向量
///
/// # 示例
/// ```
/// use voxudio::{SpeakerEmbeddingExtractor, load_audio};
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
/// let mut see = SpeakerEmbeddingExtractor::new("../checkpoint/speaker_embedding_extractor.onnx")?;
/// let (audio_data, channels) = load_audio::<22050, _>("../asset/hello_in_cn.mp3", true).await?;
/// let embeddings = see.extract(&audio_data, channels).await?;
/// Ok(())
/// }
/// ```
///
/// # 错误
/// - 模型加载失败时返回 `OperationError`
/// - 音频处理失败时返回 `OperationError`
///
/// # 注意事项
/// - 只能处理22050采样率的音频
/// - 输入音频数据应为单声道或双声道PCM格式
/// - 每个声道会输出一个256维的特征向量
pub struct SpeakerEmbeddingExtractor {
    model: Session,
}

impl SpeakerEmbeddingExtractor {
    /// 创建一个新的说话人嵌入提取器实例
    ///
    /// # 参数
    /// - `model_path`: ONNX模型文件路径
    ///
    /// # 返回值
    /// - `Result<Self, OperationError>`: 成功返回提取器实例，失败返回错误
    ///
    /// # 示例
    /// ```
    /// use voxudio::SpeakerEmbeddingExtractor;
    /// let extractor = SpeakerEmbeddingExtractor::new("../checkpoint/speaker_embedding_extractor.onnx")?;
    /// ```
    pub fn new<P>(model_path: P) -> Result<Self, OperationError>
    where
        P: AsRef<Path>,
    {
        let model = get_session_builder()?.commit_from_file(model_path)?;

        Ok(Self { model })
    }

    /// 从音频数据中提取说话人嵌入特征
    ///
    /// # 参数
    /// - `audio`: 音频数据，采样率必须为22050Hz
    ///
    /// # 返回值
    /// - `Result<Vec<[f32; 256]>, OperationError>`: 成功返回每个声道的256维特征向量，失败返回错误
    ///
    /// # 示例
    /// ```
    /// use voxudio::SpeakerEmbeddingExtractor;
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let mut see = SpeakerEmbeddingExtractor::new("../checkpoint/speaker_embedding_extractor.onnx")?;
    ///     let embeddings = see.extract(&[0.0; 22050], 1).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn extract(
        &mut self,
        audio: &[f32],
        channels: usize,
    ) -> Result<Vec<[f32; 256]>, OperationError> {
        let len = audio.len();
        let x = Array2::from_shape_vec((len / channels, channels), audio.to_vec())?;
        let options = RunOptions::new()?;
        let outputs = self
            .model
            .run_async(
                inputs![
                    "audio" => TensorRef::from_array_view(&x)?
                ],
                &options,
            )?
            .await?;
        let se = outputs["se"].try_extract_array::<f32>()?;

        let mut out = Vec::with_capacity(channels);
        // 使用数组推导式更简洁地填充结果
        for i in 0..channels {
            let row: [f32; 256] = from_fn(|j| se[[i, j]]);
            out.push(row);
        }

        Ok(out)
    }
}
