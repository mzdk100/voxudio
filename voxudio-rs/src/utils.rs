//! 音频处理工具模块
//!
//! 提供音频加载和重采样等功能
//!
//! # 主要功能
//! - `GenericSample`: 音频样本类型特征，统一 f32/i16 样本格式
//! - `load_audio`: 从文件加载音频数据并转换为指定采样率和声道数
//! - `resample`: 对音频数据进行重采样处理
//! - `spatial_audio`: 生成空间音频
//! - `speed`: 对音频数据进行变速处理（变速变调）
//!
//! # 错误处理
//! 使用`OperationError`作为统一的错误类型
//!
//! # 示例
//! ```
//! use voxudio::{decode_audio,resample};
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//! use voxudio::load_audio;
//! let (audio, channels) = load_audio::<44100, f32, _>("../asset/hello_in_cn.mp3", false).await?;
//! let resampled = resample::<44100, 48000, f32>(&audio, channels, 2)?;
//! Ok(())
//! }
//! ```

use {
    crate::OperationError,
    rodio::{
        ChannelCount, Decoder, SampleRate, Source,
        buffer::SamplesBuffer,
        source::{Spatial, UniformSourceIterator},
    },
    std::{
        io::{Cursor, Error as IoError},
        path::Path,
        slice::from_raw_parts,
    },
    tokio::fs::read,
};

/// 音频样本类型特征
///
/// 定义了可用于音频处理的样本类型必须实现的方法。
/// 这个特征允许各种音频处理器与不同的音频样本格式（如 f32 和 i16）一起工作。
///
/// # 实现
/// 目前支持以下类型：
/// - `f32`: 32位浮点样本
/// - `i16`: 16位整数样本
pub trait GenericSample: Clone {
    /// 检查样本类型是否为 f32
    fn is_f32() -> bool {
        false
    }

    /// 检查样本类型是否为 i16
    fn is_i16() -> bool {
        false
    }

    /// 创建一个值为零的样本
    fn zero() -> Self;

    /// 将样本转换为 f32
    fn to_f32(&self) -> f32;

    /// 从 f32 值创建样本
    fn from_f32(value: f32) -> Self;
}

impl GenericSample for f32 {
    fn is_f32() -> bool {
        true
    }

    fn zero() -> Self {
        0.0
    }

    fn to_f32(&self) -> f32 {
        *self
    }

    fn from_f32(value: f32) -> Self {
        value
    }
}

impl GenericSample for i16 {
    fn is_i16() -> bool {
        true
    }

    fn zero() -> Self {
        0
    }

    fn to_f32(&self) -> f32 {
        *self as f32 / i16::MAX as f32
    }

    fn from_f32(value: f32) -> Self {
        (value * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }
}

/// 将样本数组转换为 f32 数组（零拷贝优化：f32 输入直接返回）
pub(super) fn samples_to_f32<S>(samples: &[S]) -> Vec<f32>
where
    S: GenericSample,
{
    if S::is_f32() {
        // SAFETY: GenericSample::is_f32() 为 true 时，S 就是 f32
        unsafe { from_raw_parts(samples.as_ptr() as *const f32, samples.len()) }.to_vec()
    } else {
        samples.iter().map(|s| s.to_f32()).collect()
    }
}

/// 将 f32 数组转换为目标样本类型数组（零拷贝优化：f32 输出直接返回）
fn f32_to_samples<S>(samples: &[f32]) -> Vec<S>
where
    S: GenericSample,
{
    if S::is_f32() {
        // SAFETY: GenericSample::is_f32() 为 true 时，S 就是 f32
        unsafe { from_raw_parts(samples.as_ptr() as *const S, samples.len()) }.to_vec()
    } else {
        samples.iter().map(|&v| S::from_f32(v)).collect()
    }
}

/// 异步加载音频文件
///
/// # 参数
/// - `SR`: 目标采样率
/// - `S`: 样本类型，需实现 [`GenericSample`] 特征
/// - `P`: 音频文件路径类型，需实现`AsRef<Path>`
/// - `audio_path`: 音频文件路径
/// - `mono`: 是否转换为单声道
///
/// # 返回
/// `Result<(Vec<S>, usize), OperationError>`: 样本数组和声道数量（双声道样本交错排列），或者错误
///
/// # 示例
/// ```
/// use voxudio::load_audio;
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
/// let (samples, channels) = load_audio::<44100, f32, _>("../asset/hello_in_cn.mp3", false).await?;
/// Ok(())
/// }
/// ```
pub async fn load_audio<const SR: usize, S, P>(
    audio_path: P,
    mono: bool,
) -> Result<(Vec<S>, usize), OperationError>
where
    S: GenericSample,
    P: AsRef<Path>,
{
    decode_audio::<SR, S, _>(read(audio_path).await?, mono)
}

/// 解码音频数据
///
/// # 参数
/// - `SR`: 目标采样率
/// - `S`: 样本类型，需实现 [`GenericSample`] 特征
/// - `D`: 音频数据类型，需实现`AsRef<[u8]> + Send + Sync + 'static`
/// - `audio_data`: 音频数据字节数组
/// - `mono`: 是否转换为单声道
///
/// # 返回
/// `Result<(Vec<S>, usize), OperationError>`: 样本数组和声道数量（双声道样本交错排列），或者错误
///
/// # 示例
/// ```
/// use voxudio::decode_audio;
/// fn main() -> anyhow::Result<()> {
///     let (samples, channels) = decode_audio::<44100, f32, _>(include_bytes!("../../asset/hello_in_cn.mp3"), false)?;
///     Ok(())
/// }
/// ```
pub fn decode_audio<const SR: usize, S, D>(
    audio_data: D,
    mono: bool,
) -> Result<(Vec<S>, usize), OperationError>
where
    S: GenericSample,
    D: AsRef<[u8]> + Send + Sync + 'static,
{
    let file = Cursor::new(audio_data);
    let decoder = Decoder::new(file)?;
    let channels = if mono { 1 } else { decoder.channels().into() } as usize;
    let samples: Vec<f32> = UniformSourceIterator::new(
        decoder,
        ChannelCount::new(channels as _).ok_or(IoError::other("Invalid channel count."))?,
        SampleRate::new(SR as _).ok_or(IoError::other("Invalid sample rate."))?,
    )
    .collect();

    if samples.is_empty() {
        return Err(OperationError::InputInvalid("Audio is empty.".to_owned()));
    }

    Ok((f32_to_samples(&samples), channels))
}

/// 对音频数据进行重采样处理
///
/// # 参数
/// - `SSR`: 源采样率
/// - `TSR`: 目标采样率
/// - `S`: 样本类型，需实现 [`GenericSample`] 特征
/// - `samples`: 样本数组，双声道样本交错排列
/// - `src_channels`: 原始声道数量
/// - `tgt_channels`: 目标声道数量
///
/// # 返回
/// `Result<Vec<S>, OperationError>`: 重采样后的样本数组
///
/// # 示例
/// ```
/// use voxudio::resample;
/// fn main() -> anyhow::Result<()> {
///     let samples = vec![0.1f32, 0.2, 0.3, 0.4];
///     let resampled = resample::<44100, 48000, f32>(&samples, 1, 2)?;
///
///     Ok(())
/// }
/// ```
pub fn resample<const SSR: usize, const TSR: usize, S>(
    samples: &[S],
    src_channels: usize,
    tgt_channels: usize,
) -> Result<Vec<S>, OperationError>
where
    S: GenericSample,
{
    resample_dynamic(samples, SSR, TSR, src_channels, tgt_channels)
}

/// 动态采样率版本的重采样处理
///
/// 参数与返回值同静态版本
///
/// # 示例
/// ```
/// use voxudio::resample_dynamic;
/// fn main() -> anyhow::Result<()> {
///     let samples = vec![0.1f32, 0.2, 0.3, 0.4];
///     let resampled = resample_dynamic(&samples, 44100, 48000, 1, 2)?;
///
///     Ok(())
/// }
/// ```
pub fn resample_dynamic<S>(
    samples: &[S],
    ssr: usize,
    tsr: usize,
    src_channels: usize,
    tgt_channels: usize,
) -> Result<Vec<S>, OperationError>
where
    S: GenericSample,
{
    if samples.is_empty() || src_channels == 0 || tgt_channels == 0 {
        return Ok(Default::default());
    }

    let f32_samples = samples_to_f32(samples);
    let resampled: Vec<f32> = UniformSourceIterator::new(
        SamplesBuffer::new(
            ChannelCount::new(src_channels as _).ok_or(IoError::other("Invalid channel count."))?,
            SampleRate::new(ssr as _).ok_or(IoError::other("Invalid sample rate."))?,
            f32_samples,
        ),
        ChannelCount::new(tgt_channels as _).ok_or(IoError::other("Invalid channel count."))?,
        SampleRate::new(tsr as _).ok_or(IoError::other("Invalid sample rate."))?,
    )
    .collect();

    Ok(f32_to_samples(&resampled))
}

/// 对音频数据进行空间化处理（3D音效）
///
/// # 参数
/// - `SR`: 采样率
/// - `S`: 样本类型，需实现 [`GenericSample`] 特征
/// - `audio`: 样本数组，双声道样本交错排列
/// - `channels`: 声道数量
/// - `emitter_position`: 声源位置坐标[x, y, z]
/// - `left_ear`: 左耳位置坐标[x, y, z]
/// - `right_ear`: 右耳位置坐标[x, y, z]
///
/// # 返回
/// `Result<Vec<S>, OperationError>`: 处理后的样本数组
///
/// # 示例
/// ```
/// use voxudio::spatial_audio;
/// fn main() -> anyhow::Result<()> {
///     let samples = vec![0.1f32, 0.2, 0.3, 0.4];
///     let processed = spatial_audio::<44100, f32>(
///         &samples,
///         2,
///         [0.0, 0.0, 0.0],
///         [-0.1, 0.0, 0.0],
///         [0.1, 0.0, 0.0]
///     )?;
///
///     Ok(())
/// }
/// ```
pub fn spatial_audio<const SR: usize, S>(
    audio: &[S],
    channels: usize,
    emitter_position: [f32; 3],
    left_ear: [f32; 3],
    right_ear: [f32; 3],
) -> Result<Vec<S>, OperationError>
where
    S: GenericSample,
{
    let f32_audio = samples_to_f32(audio);
    let result: Vec<f32> = Spatial::new(
        SamplesBuffer::new(
            ChannelCount::new(channels as _).ok_or(IoError::other("Invalid channel count."))?,
            SampleRate::new(SR as _).ok_or(IoError::other("Invalid sample rate."))?,
            f32_audio,
        ),
        emitter_position,
        left_ear,
        right_ear,
    )
    .collect();

    Ok(f32_to_samples(&result))
}

/// 对音频数据添加房间混响效果
///
/// # 参数
/// - `SR`: 采样率
/// - `S`: 样本类型，需实现 [`GenericSample`] 特征
/// - `audio`: 样本数组，双声道样本交错排列
/// - `channels`: 声道数量
/// - `room_size`: 房间大小系数 (0.0-1.0)
/// - `damping`: 高频衰减系数 (0.0-1.0)
/// - `wet`: 混响强度 (0.0-1.0)
///
/// # 返回
/// `Vec<S>`: 添加混响后的样本数组
///
/// # 示例
/// ```
/// use voxudio::reverb;
/// let samples = vec![0.1f32, 0.2, 0.3, 0.4];
/// let processed = reverb::<44100, f32>(&samples, 2, 0.7, 0.5, 0.3);
/// ```
pub fn reverb<const SR: usize, S>(
    audio: &[S],
    channels: usize,
    room_size: f32,
    damping: f32,
    wet: f32,
) -> Vec<S>
where
    S: GenericSample,
{
    if audio.is_empty() || channels == 0 {
        return Vec::new();
    }

    let f32_audio = samples_to_f32(audio);
    let mut output = vec![0.0; f32_audio.len()];
    let mut delay_lines = vec![vec![0.0; (SR as f32 * 0.1) as usize]; 8]; // 8个延迟线
    let mut delay_pos = [0; 8];
    let delay_lengths = [
        (SR as f32 * 0.0297) as usize,
        (SR as f32 * 0.0371) as usize,
        (SR as f32 * 0.0411) as usize,
        (SR as f32 * 0.0437) as usize,
        (SR as f32 * 0.005) as usize,
        (SR as f32 * 0.0157) as usize,
        (SR as f32 * 0.0201) as usize,
        (SR as f32 * 0.0263) as usize,
    ];

    for i in 0..f32_audio.len() {
        let sample = f32_audio[i];
        let mut reverb_sample = 0.0;

        // 处理8个延迟线
        for j in 0..8 {
            let delayed = delay_lines[j][delay_pos[j]];
            reverb_sample += delayed * 0.125; // 平均分配
            delay_lines[j][delay_pos[j]] = sample + delayed * damping;
            delay_pos[j] = (delay_pos[j] + 1) % delay_lengths[j];
        }

        // 混合原始信号和混响信号
        output[i] = sample * (1.0 - wet) + reverb_sample * wet * room_size;
    }

    f32_to_samples(&output)
}

/// 对音频数据进行变速处理（变速变调）
///
/// 通过调整采样率实现变速效果，速度改变的同时音高也会相应变化。
/// factor > 1.0 加速（音高升高），factor < 1.0 减速（音高降低）。
///
/// # 参数
/// - `SR`: 采样率
/// - `S`: 样本类型，需实现 [`GenericSample`] 特征
/// - `audio`: 样本数组，双声道样本交错排列
/// - `channels`: 声道数量
/// - `factor`: 变速因子（必须大于0，1.0为原始速度）
///
/// # 返回
/// `Result<Vec<S>, OperationError>`: 变速后的样本数组
///
/// # 示例
/// ```
/// use voxudio::speed;
/// fn main() -> anyhow::Result<()> {
///     let samples = vec![0.1f32, 0.2, 0.3, 0.4];
///     let sped_up = speed::<44100, f32>(&samples, 2, 1.5)?; // 1.5倍速
///     let slowed_down = speed::<44100, f32>(&samples, 2, 0.75)?; // 0.75倍速
///
///     Ok(())
/// }
/// ```
pub fn speed<const SR: usize, S>(
    audio: &[S],
    channels: usize,
    factor: f32,
) -> Result<Vec<S>, OperationError>
where
    S: GenericSample,
{
    if audio.is_empty() || channels == 0 {
        return Ok(Default::default());
    }
    if factor <= 0.0 {
        return Err(OperationError::InputInvalid(
            "Speed factor must be greater than 0.".to_owned(),
        ));
    }
    if (factor - 1.0).abs() < f32::EPSILON {
        return Ok(audio.to_vec());
    }

    let f32_audio = samples_to_f32(audio);
    let src_channels =
        ChannelCount::new(channels as _).ok_or(IoError::other("Invalid channel count."))?;
    let src_sample_rate = SampleRate::new(SR as _).ok_or(IoError::other("Invalid sample rate."))?;

    let source = SamplesBuffer::new(src_channels, src_sample_rate, f32_audio).speed(factor);

    // 变速后采样率改变，重采样回原始采样率以保持采样率一致
    let result: Vec<f32> =
        UniformSourceIterator::new(source, src_channels, src_sample_rate).collect();
    Ok(f32_to_samples(&result))
}
