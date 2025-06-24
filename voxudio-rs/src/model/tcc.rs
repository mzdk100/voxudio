use {
    super::get_session_builder,
    crate::{OperationError, resample},
    ndarray::Array2,
    ort::{inputs, session::RunOptions, session::Session, value::TensorRef},
    std::{
        path::Path,
        time::{Duration, SystemTime},
    },
};

/// 音色转换器，用于将源音频转换为目标音色
///
/// # 示例
/// ```
/// use voxudio::ToneColorConverter;
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
/// let mut converter = ToneColorConverter::new("../checkpoint/tone_color_converter.onnx")?;
/// let src_audio = vec![0f32; 22050];
/// let src_se = vec![[0f32; 256]; 1];
/// let tgt_se = vec![[0f32; 256]; 1];
/// let (converted_audio, channels, duration) = converter.convert(&src_audio, &src_se, &tgt_se).await?;
/// Ok(())
/// }
/// ```
///
/// # 错误
/// 可能返回`OperationError`表示模型加载或推理过程中的错误
pub struct ToneColorConverter {
    model: Session,
}

impl ToneColorConverter {
    /// 创建一个新的音色转换器实例
    ///
    /// # 参数
    /// * `model_path` - ONNX 模型文件路径
    ///
    /// # 返回值
    /// * `Result<Self, OperationError>` - 成功返回音色转换器实例，失败返回操作错误
    ///
    /// # 示例
    /// ```
    /// use voxudio::ToneColorConverter;
    /// fn main() -> anyhow::Result<()> {
    /// let converter = ToneColorConverter::new("../checkpoint/tone_color_converter.onnx")?;
    /// 
    /// Ok(())
    /// }
    /// ```
    pub fn new<P>(model_path: P) -> Result<Self, OperationError>
    where
        P: AsRef<Path>,
    {
        let model = get_session_builder()?.commit_from_file(model_path)?;

        Ok(Self { model })
    }

    /// 执行音色转换
    ///
    /// # 参数
    /// * `src_audio` - 源音频数据（如果是双声道则样本交错存储）
    /// * `src_se` - 源音色嵌入，形状为[src_channels, 256]的数组
    /// * `tgt_se` - 目标音色嵌入，形状为[tgt_channels, 256]的数组
    ///
    /// # 返回值
    /// * `Result<(Vec<f32>, usize, Duration), OperationError>` -
    ///   成功返回转换后的音频数据（如果是双声道则样本交错存储）、声道数和推理耗时，
    ///   失败返回操作错误
    ///
    /// # 注意
    /// * 输入音频采样率必须为22050Hz
    /// * 当源音色声道数小于目标音色时，会自动进行声道扩展
    pub async fn convert(
        &mut self,
        src_audio: &[f32],
        src_se: &[[f32; 256]],
        tgt_se: &[[f32; 256]],
    ) -> Result<(Vec<f32>, usize, Duration), OperationError> {
        let max_channels = tgt_se.len().max(src_se.len());
        let audio = if src_se.len() < max_channels {
            resample::<22050, 22050>(&src_audio, src_se.len(), max_channels)
        } else {
            src_audio.to_vec()
        };
        let audio = Array2::from_shape_vec((audio.len() / max_channels, max_channels), audio)?;
        let src_se = Array2::from(src_se.to_vec());
        let tgt_se = Array2::from(tgt_se.to_vec());

        let options = RunOptions::new()?;
        let start = SystemTime::now();
        let outputs = self
            .model
            .run_async(
                inputs![
                    "src_audio" => TensorRef::from_array_view(&audio)?,
                    "src_se" => TensorRef::from_array_view(&src_se)?,
                    "tgt_se" => TensorRef::from_array_view(&tgt_se)?,
                ],
                &options,
            )?
            .await?;
        let (_, audio) = outputs["audio"].try_extract_tensor::<f32>()?;

        Ok((audio.to_vec(), max_channels, start.elapsed()?))
    }
}
