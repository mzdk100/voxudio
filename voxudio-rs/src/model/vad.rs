use {
    super::get_session_builder,
    crate::{OperationError, resample},
    ndarray::{Array0, Array2, Array3, Axis, concatenate, s},
    ort::{
        inputs,
        session::{RunOptions, Session},
        value::TensorRef,
    },
    std::{
        ops::{Deref, DerefMut},
        path::Path,
    },
};

/// 定义语音活动检测器配置结构体
///
/// 该结构体用于配置语音活动检测器的参数。
///
/// # Fields
///
/// * `threshold`: 检测阈值，用于判断音频是否为语音。
/// * `min_speech_duration_ms`: 最小语音持续时间，单位为毫秒。
/// * `max_speech_duration_s`: 最大语音持续时间，单位为秒。
/// * `min_silence_duration_ms`: 最小静音持续时间，单位为毫秒。
/// * `speech_pad_ms`: 语音填充时间，单位为毫秒。
#[derive(Debug, Clone)]
pub struct VoiceActivityDetectorConfig {
    pub threshold: f32,
    pub min_speech_duration_ms: u64,
    pub max_speech_duration_s: f64,
    pub min_silence_duration_ms: u64,
    pub speech_pad_ms: u64,
}

/// 为 VoiceActivityDetectorConfig 实现 Default trait
///
/// 提供语音活动检测器的默认配置参数：
/// - 检测阈值: 0.5
/// - 最小语音持续时间: 100 毫秒
/// - 最大语音持续时间: 无限制
/// - 最小静音持续时间: 1000 毫秒
/// - 语音填充时间: 30 毫秒
impl Default for VoiceActivityDetectorConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 100,
            max_speech_duration_s: f64::INFINITY,
            min_silence_duration_ms: 1000,
            speech_pad_ms: 30,
        }
    }
}

/// 语音活动检测器构建器
///
/// 用于构建语音活动检测器实例，允许自定义配置参数。
///
/// # Fields
///
/// * `config`: 语音活动检测器的配置参数
///
pub struct VoiceActivityDetectorBuilder {
    config: VoiceActivityDetectorConfig,
}

impl VoiceActivityDetectorBuilder {
    /// 设置语音活动检测阈值
    ///
    /// # 参数
    ///
    /// * `threshold`: 检测阈值，范围应在0.0到1.0之间
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// 设置最小语音持续时间
    ///
    /// # 参数
    ///
    /// * `duration_ms`: 最小持续时间，单位为毫秒
    pub fn with_min_speech_duration(mut self, duration_ms: u64) -> Self {
        self.config.min_speech_duration_ms = duration_ms;
        self
    }

    /// 设置最大语音持续时间
    ///
    /// # 参数
    ///
    /// * `duration_s`: 最大持续时间，单位为秒
    pub fn with_max_speech_duration(mut self, duration_s: f64) -> Self {
        self.config.max_speech_duration_s = duration_s;
        self
    }

    /// 设置最小静音持续时间
    ///
    /// # 参数
    ///
    /// * `duration_ms`: 最小持续时间，单位为毫秒
    pub fn with_min_silence_duration(mut self, duration_ms: u64) -> Self {
        self.config.min_silence_duration_ms = duration_ms;
        self
    }

    /// 设置语音填充时间
    ///
    /// # 参数
    ///
    /// * `duration_ms`: 填充时间，单位为毫秒
    pub fn with_speech_pad(mut self, duration_ms: u64) -> Self {
        self.config.speech_pad_ms = duration_ms;
        self
    }

    /// 构建语音活动检测器实例
    ///
    /// 根据配置参数和模型路径创建语音活动检测器。
    ///
    /// # 参数
    ///
    /// * `model_path`: ONNX 模型文件路径
    ///
    /// # 返回值
    ///
    /// 返回包含语音活动检测器的 `Result`，如果构建失败则返回 `OperationError`
    ///
    /// # 示例
    ///
    /// ```
    /// use voxudio::VoiceActivityDetectorBuilder;
    /// let vad = VoiceActivityDetectorBuilder::default()
    ///     .with_threshold(0.6)
    ///     .build("../checkpoint/voice_activity_detector.onnx")?;
    /// ```
    pub fn build<P>(&self, model_path: P) -> Result<VoiceActivityDetector, OperationError>
    where
        P: AsRef<Path>,
    {
        let model = get_session_builder()?.commit_from_file(model_path)?;
        let last_batch_size = 0;
        let state = Array3::default((2, last_batch_size, 128));
        let context = Default::default();
        let last_sr = 0;

        Ok(VoiceActivityDetector {
            model,
            config: self.config.to_owned(),
            state,
            context,
            last_sr,
            last_batch_size,
        })
    }
}

impl Default for VoiceActivityDetectorBuilder {
    fn default() -> Self {
        Self {
            config: Default::default(),
        }
    }
}

/// 语音活动检测器
///
/// 该结构体实现了基于ONNX模型的语音活动检测功能，能够识别音频中的语音片段。
///
/// # Fields
///
/// * `model`: ONNX模型会话
/// * `config`: 语音活动检测配置参数
/// * `state`: 模型状态数组，维度为(2, batch_size, 128)
/// * `context`: 上下文数组，用于保存历史音频特征
/// * `last_sr`: 上一次处理的音频采样率
/// * `last_batch_size`: 上一次处理的音频批次大小
///
/// # 方法
///
/// * `new`: 使用默认配置创建语音活动检测器
/// * `builder`: 获取配置构建器
/// * `detect`: 检测单帧音频是否为语音
/// * `get_speech_timestamps`: 获取音频中所有语音片段的时间戳
/// * `retain_speech_only`: 仅保留音频中的语音部分，非语音会被删除
///
/// # 示例
///
/// ```
/// use voxudio::{VoiceActivityDetector,load_audio};
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
/// let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;
/// let audio_data = load_audio::<16000, _>("../asset/hello_in_cn.mp3", true).await?;
/// let segments = vad.get_speech_segments::<16000>(&audio_data[0]).await?;
/// Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct VoiceActivityDetector {
    model: Session,
    config: VoiceActivityDetectorConfig,
    state: Array3<f32>,
    context: Array2<f32>,
    last_sr: usize,
    last_batch_size: usize,
}

impl VoiceActivityDetector {
    /// 创建语音活动检测器实例
    ///
    /// 使用默认配置参数和指定的ONNX模型路径创建语音活动检测器。
    ///
    /// # 参数
    ///
    /// * `model_path`: ONNX模型文件路径
    ///
    /// # 返回值
    ///
    /// 返回包含语音活动检测器的 `Result`，如果构建失败则返回 `OperationError`
    ///
    /// # 示例
    ///
    /// ```
    /// let vad = voxudio::VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;
    /// ```
    pub fn new<P>(model_path: P) -> Result<Self, OperationError>
    where
        P: AsRef<Path>,
    {
        Self::builder().build(model_path)
    }

    /// 获取语音活动检测器构建器
    ///
    /// 返回一个配置构建器实例，可用于自定义语音活动检测器的参数。
    ///
    /// # 返回值
    ///
    /// 返回 `VoiceActivityDetectorBuilder` 实例
    ///
    /// # 示例
    ///
    /// ```
    /// let builder = voxudio::VoiceActivityDetector::builder();
    /// ```
    pub fn builder() -> VoiceActivityDetectorBuilder {
        Default::default()
    }

    fn validate_input<const SR: usize>(
        &self,
        x: &[f32],
    ) -> Result<(Array2<f32>, usize), OperationError> {
        let (x, sr) = if SR != 16000 && (SR % 16000 == 0) {
            let step = SR / 16000;
            let x = x.iter().step_by(step).collect::<Vec<_>>();
            (Array2::from_shape_fn((1, x.len()), |(_, i)| *x[i]), 16000)
        } else {
            (Array2::from_shape_fn((1, x.len()), |(_, i)| x[i]), SR)
        };
        if ![8000, 16000].contains(&sr) {
            return Err(OperationError::InputInvalid(format!(
                "Unsupported sample rate. Supported sampling rates: [8000, 16000] (or multiply of 16000), current is {}",
                sr
            )));
        }
        if sr as f32 / x.shape()[1] as f32 > 31.25 {
            return Err(OperationError::InputTooShort);
        }

        Ok((x, sr))
    }

    fn reset_states(&mut self, batch_size: usize) {
        self.state = Array3::default((2, batch_size, 128));
        self.context = Default::default();
        self.last_sr = 0;
        self.last_batch_size = 0;
    }

    /// 检测单帧音频是否为语音
    ///
    /// 该方法会对输入的音频帧进行语音活动检测，返回语音概率值。
    ///
    /// # 参数
    ///
    /// * `x`: 音频帧数据，长度必须为512(16kHz)或256(8kHz)个采样点
    ///
    /// # 返回值
    ///
    /// 返回语音概率值(0.0-1.0)，越接近1.0表示越可能是语音
    ///
    /// # 错误
    ///
    /// 可能返回以下错误：
    /// - `OperationError::InputTooShort`: 输入音频过短
    /// - `OperationError::InputInvalid`: 输入音频长度不符合要求
    ///
    /// # 示例
    ///
    /// ```
    /// use voxudio::{VoiceActivityDetector, load_audio};
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    /// let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;
    /// let audio_data = load_audio::<16000, _>("../asset/hello_in_cn.mp3", true).await?;
    /// let prob = vad.detect::<16000>(&audio_data[0][..512]).await?;
    /// if prob > 0.5 {
    ///     println!("检测到语音");
    /// }
    /// Ok(())
    /// }
    /// ```
    pub async fn detect<const SR: usize>(&mut self, x: &[f32]) -> Result<f32, OperationError> {
        let (x, sr) = self.validate_input::<SR>(x)?;
        let num_samples = if sr == 16000 { 512 } else { 256 };
        if x.shape()[1] != num_samples {
            return Err(OperationError::InputInvalid(format!(
                "Provided number of samples is {} (Supported values: 256 for 8000 sample rate, 512 for 16000)",
                x.shape()[1]
            )));
        }
        let context_size = if sr == 16000 { 64 } else { 32 };
        let batch_size = x.shape()[0];
        if self.last_batch_size < 1 {
            self.reset_states(batch_size);
        }
        if self.last_sr > 0 && self.last_sr != sr {
            self.reset_states(x.shape()[0]);
        }
        if self.last_batch_size > 0 && self.last_batch_size != batch_size {
            self.reset_states(batch_size);
        }
        if self.context.is_empty() {
            self.context = Array2::from_elem((batch_size, context_size), 0f32);
        }
        let x = concatenate(Axis(1), &[self.context.view(), x.view()])?;
        let sr2 = Array0::from_elem((), sr as i64);
        let state = self.state.to_owned();
        let options = RunOptions::new()?;
        let outputs = self
            .model
            .run_async(
                inputs![
                    "input" => TensorRef::from_array_view(&x)?,
                    "sr" => TensorRef::from_array_view(&sr2)?,
                    "state" => TensorRef::from_array_view(&state)?,
                ],
                &options,
            )?
            .await?;
        let ((_, out), state) = (
            outputs["output"].try_extract_tensor()?,
            outputs["stateN"].try_extract_array()?,
        );
        self.state = state.into_dimensionality()?.to_owned();
        self.context = x.slice(s!(.., x.shape()[1] - context_size..)).to_owned();
        self.last_sr = sr;
        self.last_batch_size = batch_size;

        Ok(out[0])
    }

    /// 获取音频中的语音片段
    ///
    /// 该方法会对整个音频进行分析，返回所有检测到的语音片段。
    ///
    /// # 参数
    ///
    /// * `audio`: 音频数据，采样率由泛型参数 `SR` 指定
    ///
    /// # 返回值
    ///
    /// 返回包含语音片段起始和结束位置的元组列表(以`SR`为参考的样本索引）
    ///
    /// # 错误
    ///
    /// 可能返回以下错误：
    /// - `OperationError::InputTooShort`: 输入音频过短
    ///
    /// # 示例
    ///
    /// ```
    /// use voxudio::{load_audio,VoiceActivityDetector};
    /// async fn main() -> anyhow::Result<()> {
    /// let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;
    /// let audio_data = load_audio::<16000, _>("../asset/hello_in_cn.mp3", true).await?;
    /// let segments = vad.get_speech_segments::<16000>(&audio_data[0]).await?;
    /// for (start, end) in segments {
    ///     println!("语音片段: {}ms - {}ms", start*1000/16000, end*1000/16000);
    /// }
    /// Ok(())
    /// }
    /// ```
    pub async fn get_speech_segments<const SR: usize>(
        &mut self,
        audio: &[f32],
    ) -> Result<Vec<(usize, usize)>, OperationError> {
        let window_size_samples = if SR == 16000 { 512 } else { 256 };
        self.reset_states(1);

        let min_speech_samples = (SR as u64 * self.min_speech_duration_ms) / 1000;
        let speech_pad_samples = (SR as u64 * self.speech_pad_ms) / 1000;
        let max_speech_samples = (SR as f64 * self.max_speech_duration_s) as u64
            - window_size_samples as u64
            - 2 * speech_pad_samples;
        let min_silence_samples = (SR as u64 * self.min_silence_duration_ms) / 1000;
        let min_silence_samples_at_max_speech = (SR as u64 * 98) / 1000;

        let audio_length = audio.len();
        let mut speech_prob_list = Vec::new();

        for chunk_start in (0..audio_length).step_by(window_size_samples) {
            let chunk_end = chunk_start + window_size_samples;
            let chunk = if chunk_end <= audio_length {
                audio[chunk_start..chunk_end].to_vec()
            } else {
                let mut padded = vec![0.0; window_size_samples];
                let actual_len = audio_length - chunk_start;
                padded[..actual_len].copy_from_slice(&audio[chunk_start..]);
                padded
            };

            speech_prob_list.push(self.detect::<SR>(&chunk).await?);
        }

        let mut triggered = false;
        let mut speeches = Vec::new();
        let mut current_speech: Option<(usize, usize)> = None;
        let neg_threshold = self.threshold.max(0.15) - 0.15;
        let mut temp_end = 0;
        let mut prev_end = 0;
        let mut next_start = 0;

        for (i, &prob) in speech_prob_list.iter().enumerate() {
            let current_sample = i * window_size_samples;

            if prob >= self.threshold && temp_end > 0 {
                temp_end = 0;
                if next_start < prev_end {
                    next_start = current_sample;
                }
            }

            if prob >= self.threshold && !triggered {
                triggered = true;
                current_speech = Some((current_sample, 0));
            }

            if let Some((start, _)) = current_speech {
                if triggered && (current_sample - start) as u64 > max_speech_samples {
                    if prev_end > 0 {
                        let mut speech = current_speech.take().unwrap();
                        speech.1 = prev_end;
                        speeches.push(speech);
                        current_speech = Some((next_start, 0));
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                    } else {
                        let mut speech = current_speech.take().unwrap();
                        speech.1 = current_sample;
                        speeches.push(speech);
                        current_speech = None;
                        triggered = false;
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                        continue;
                    }
                }

                if prob < neg_threshold && triggered {
                    if temp_end == 0 {
                        temp_end = current_sample;
                    }

                    if (current_sample - temp_end) as u64 > min_silence_samples_at_max_speech {
                        prev_end = temp_end;
                    }

                    if ((current_sample - temp_end) as u64) < min_silence_samples {
                        continue;
                    } else {
                        let mut speech = current_speech.take().unwrap();
                        speech.1 = temp_end;
                        if (speech.1 - speech.0) as u64 > min_speech_samples {
                            speeches.push(speech);
                        }
                        current_speech = None;
                        triggered = false;
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                        continue;
                    }
                }
            }
        }

        if let Some((start, _)) = current_speech {
            if (audio_length - start) as u64 > min_speech_samples {
                speeches.push((start, audio_length));
            }
        }

        // 应用边界填充
        for i in 0..speeches.len() {
            if i == 0 {
                speeches[i].0 = speeches[i].0.saturating_sub(speech_pad_samples as usize);
            }

            if i < speeches.len() - 1 {
                let silence_duration = speeches[i + 1].0 - speeches[i].1;
                if silence_duration < (2 * speech_pad_samples) as usize {
                    let half_silence = silence_duration / 2;
                    speeches[i].1 += half_silence;
                    speeches[i + 1].0 = speeches[i + 1]
                        .0
                        .saturating_sub(silence_duration - half_silence);
                } else {
                    speeches[i].1 = (speeches[i].1 as u64 + speech_pad_samples) as usize;
                    speeches[i].1 = speeches[i].1.min(audio_length);
                    speeches[i + 1].0 = speeches[i + 1]
                        .0
                        .saturating_sub(speech_pad_samples as usize);
                }
            } else {
                speeches[i].1 = (speeches[i].1 as u64 + speech_pad_samples) as usize;
                speeches[i].1 = speeches[i].1.min(audio_length);
            }
        }

        Ok(speeches)
    }

    /// 仅保留音频中的语音部分
    ///
    /// 该方法会分析音频并只保留被识别为语音的片段，移除所有静音或者非语音部分。
    ///
    /// # 参数
    ///
    /// * `audio`: 多通道音频数据，采样率由泛型参数 `SR` 指定
    ///
    /// # 返回值
    ///
    /// 返回只包含语音片段的新音频数据，保持原始通道数
    ///
    /// # 示例
    ///
    /// ```
    /// use voxudio::{load_audio,VoiceActivityDetector};
    /// async fn main() -> anyhow::Result<()> {
    /// let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;
    /// let (audio_data, channels) = load_audio::<22050, _>("../asset/hello_in_cn.mp3", false).await?;
    /// let speech_only = vad.retain_speech_only::<22050>(&audio_data, channels).await?;
    /// Ok(())
    /// }
    /// ```
    pub async fn retain_speech_only<const SR: usize>(
        &mut self,
        audio: &[f32],
        channels: usize,
    ) -> Result<Vec<f32>, OperationError> {
        let audio_16k = resample::<SR, 16000>(audio, channels, 1);
        let segments = self.get_speech_segments::<16000>(&audio_16k).await?;

        let len = audio.len();
        let mut result = Vec::with_capacity(len);
        for (start, end) in segments.iter() {
            let start_sample = start * SR / 16000;
            let end_sample = end * SR / 16000;
            result.extend_from_slice(
                &audio[start_sample * channels..(end_sample * channels + 1).min(len)],
            );
        }
        if result.len() % channels != 0 {
            result.push(0f32);
        }

        Ok(result)
    }
}

impl Deref for VoiceActivityDetector {
    type Target = VoiceActivityDetectorConfig;

    fn deref(&self) -> &Self::Target {
        &self.config
    }
}

impl DerefMut for VoiceActivityDetector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.config
    }
}
