//! 音频处理工具模块
//!
//! 提供音频加载和重采样等功能
//!
//! # 主要功能
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
//! let (audio, channels) = load_audio::<44100, _>("../asset/hello_in_cn.mp3", false).await?;
//! let resampled = resample::<44100, 48000>(&audio, channels, 2);
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
    },
    tokio::fs::read,
};

/// 异步加载音频文件
///
/// # 参数
/// - `SR`: 目标采样率
/// - `P`: 音频文件路径类型，需实现`AsRef<Path>`
/// - `audio_path`: 音频文件路径
/// - `mono`: 是否转换为单声道
///
/// # 返回
/// `Result<(Vec<f32>, usize), OperationError>`: 样本数组和声道数量（双声道样本交错排列），或者错误
///
/// # 示例
/// ```
/// use voxudio::load_audio;
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
/// let (samples, channels) = load_audio::<44100, _>("../asset/hello_in_cn.mp3", false).await?;
/// Ok(())
/// }
/// ```
pub async fn load_audio<const SR: usize, P>(
    audio_path: P,
    mono: bool,
) -> Result<(Vec<f32>, usize), OperationError>
where
    P: AsRef<Path>,
{
    decode_audio::<SR, _>(read(audio_path).await?, mono)
}

/// 解码音频数据
///
/// # 参数
/// - `SR`: 目标采样率
/// - `D`: 音频数据类型，需实现`AsRef<[u8]> + Send + Sync + 'static`
/// - `audio_data`: 音频数据字节数组
/// - `mono`: 是否转换为单声道
///
/// # 返回
/// `Result<(Vec<f32>, usize), OperationError>`: 样本数组和声道数量（双声道样本交错排列），或者错误
///
/// # 示例
/// ```
/// use voxudio::decode_audio;
/// fn main() -> anyhow::Result<()> {
///     let (samples, channels) = decode_audio::<44100, _>(include_bytes!("../../asset/hello_in_cn.mp3"), false)?;
///     Ok(())
/// }
/// ```
pub fn decode_audio<const SR: usize, D>(
    audio_data: D,
    mono: bool,
) -> Result<(Vec<f32>, usize), OperationError>
where
    D: AsRef<[u8]> + Send + Sync + 'static,
{
    let file = Cursor::new(audio_data);
    let decoder = Decoder::new(file)?;
    let channels = if mono { 1 } else { decoder.channels().into() } as usize;
    let samples = UniformSourceIterator::new(
        decoder,
        ChannelCount::new(channels as _).ok_or(IoError::other("Invalid channel count."))?,
        SampleRate::new(SR as _).ok_or(IoError::other("Invalid sample rate."))?,
    )
    .collect::<Vec<f32>>();

    if samples.is_empty() {
        return Err(OperationError::InputInvalid("Audio is empty.".to_owned()));
    }

    Ok((samples, channels))
}

/// 对音频数据进行重采样处理
///
/// # 参数
/// - `SSR`: 源采样率
/// - `TSR`: 目标采样率
/// - `samples`: 样本数组，双声道样本交错排列
/// - `src_channels`: 原始声道数量
/// - `tgt_channels`: 目标声道数量
///
/// # 返回
/// `Result<Vec<f32>, OperationError>`: 重采样后的样本数组
///
/// # 示例
/// ```
/// use voxudio::resample;
/// fn main() -> anyhow::Result<()> {
///     let samples = vec![0.1, 0.2, 0.3, 0.4];
///     let resampled = resample::<44100, 48000>(&samples, 1, 2)?;
///
///     Ok(())
/// }
/// ```
pub fn resample<const SSR: usize, const TSR: usize>(
    samples: &[f32],
    src_channels: usize,
    tgt_channels: usize,
) -> Result<Vec<f32>, OperationError> {
    if samples.is_empty() || src_channels == 0 || tgt_channels == 0 {
        return Ok(Default::default());
    }

    Ok(UniformSourceIterator::new(
        SamplesBuffer::new(
            ChannelCount::new(src_channels as _).ok_or(IoError::other("Invalid channel count."))?,
            SampleRate::new(SSR as _).ok_or(IoError::other("Invalid sample rate."))?,
            samples,
        ),
        ChannelCount::new(tgt_channels as _).ok_or(IoError::other("Invalid channel count."))?,
        SampleRate::new(TSR as _).ok_or(IoError::other("Invalid sample rate."))?,
    )
    .collect())
}

/// 对音频数据进行空间化处理（3D音效）
///
/// # 参数
/// - `SR`: 采样率
/// - `audio`: 样本数组，双声道样本交错排列
/// - `channels`: 声道数量
/// - `emitter_position`: 声源位置坐标[x, y, z]
/// - `left_ear`: 左耳位置坐标[x, y, z]
/// - `right_ear`: 右耳位置坐标[x, y, z]
///
/// # 返回
/// `Result<Vec<f32>, OperationError>`: 处理后的样本数组
///
/// # 示例
/// ```
/// use voxudio::spatial_audio;
/// fn main() -> anyhow::Result<()> {
///     let samples = vec![0.1, 0.2, 0.3, 0.4];
///     let processed = spatial_audio::<44100>(
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
pub fn spatial_audio<const SR: usize>(
    audio: &[f32],
    channels: usize,
    emitter_position: [f32; 3],
    left_ear: [f32; 3],
    right_ear: [f32; 3],
) -> Result<Vec<f32>, OperationError> {
    Ok(Spatial::new(
        SamplesBuffer::new(
            ChannelCount::new(channels as _).ok_or(IoError::other("Invalid channel count."))?,
            SampleRate::new(SR as _).ok_or(IoError::other("Invalid sample rate."))?,
            audio,
        ),
        emitter_position,
        left_ear,
        right_ear,
    )
    .collect())
}

/// 对音频数据添加房间混响效果
///
/// # 参数
/// - `SR`: 采样率
/// - `audio`: 样本数组，双声道样本交错排列
/// - `channels`: 声道数量
/// - `room_size`: 房间大小系数 (0.0-1.0)
/// - `damping`: 高频衰减系数 (0.0-1.0)
/// - `wet`: 混响强度 (0.0-1.0)
///
/// # 返回
/// `Vec<f32>`: 添加混响后的样本数组
///
/// # 示例
/// ```
/// use voxudio::reverb;
/// let samples = vec![0.1, 0.2, 0.3, 0.4];
/// let processed = reverb::<44100>(&samples, 2, 0.7, 0.5, 0.3);
/// ```
pub fn reverb<const SR: usize>(
    audio: &[f32],
    channels: usize,
    room_size: f32,
    damping: f32,
    wet: f32,
) -> Vec<f32> {
    if audio.is_empty() || channels == 0 {
        return Vec::new();
    }

    let mut output = vec![0.0; audio.len()];
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

    for i in 0..audio.len() {
        let sample = audio[i];
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

    output
}

/// 对音频数据进行变速处理（变速变调）
///
/// 通过调整采样率实现变速效果，速度改变的同时音高也会相应变化。
/// factor > 1.0 加速（音高升高），factor < 1.0 减速（音高降低）。
///
/// # 参数
/// - `SR`: 采样率
/// - `audio`: 样本数组，双声道样本交错排列
/// - `channels`: 声道数量
/// - `factor`: 变速因子（必须大于0，1.0为原始速度）
///
/// # 返回
/// `Result<Vec<f32>, OperationError>`: 变速后的样本数组
///
/// # 示例
/// ```
/// use voxudio::speed;
/// fn main() -> anyhow::Result<()> {
///     let samples = vec![0.1, 0.2, 0.3, 0.4];
///     let sped_up = speed::<44100>(&samples, 2, 1.5)?; // 1.5倍速
///     let slowed_down = speed::<44100>(&samples, 2, 0.75)?; // 0.75倍速
///
///     Ok(())
/// }
/// ```
pub fn speed<const SR: usize>(
    audio: &[f32],
    channels: usize,
    factor: f32,
) -> Result<Vec<f32>, OperationError> {
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

    let src_channels =
        ChannelCount::new(channels as _).ok_or(IoError::other("Invalid channel count."))?;
    let src_sample_rate = SampleRate::new(SR as _).ok_or(IoError::other("Invalid sample rate."))?;

    let source = SamplesBuffer::new(src_channels, src_sample_rate, audio).speed(factor);

    // 变速后采样率改变，重采样回原始采样率以保持采样率一致
    Ok(UniformSourceIterator::new(source, src_channels, src_sample_rate).collect())
}
