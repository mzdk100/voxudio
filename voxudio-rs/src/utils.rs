//! 音频处理工具模块
//!
//! 提供音频加载和重采样等功能
//!
//! # 主要功能
//! - `load_audio`: 从文件加载音频数据并转换为指定采样率和声道数
//! - `resample`: 对音频数据进行重采样处理
//! - `spatial_audio`: 生成空间音频
//!
//! # 错误处理
//! 使用`OperationError`作为统一的错误类型
//!
//! # 示例
//! ```
//! use voxudio::{load_audio,resample};
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//! let (audio, channels) = load_audio::<44100, _>("../asset/hello_in_cn.mp3", false).await?;
//! let resampled = resample::<44100, 48000>(&audio, channels, 2);
//! Ok(())
//! }
//! ```

use {
    crate::OperationError,
    rodio::{
        Decoder, Source,
        buffer::SamplesBuffer,
        source::{Spatial, UniformSourceIterator},
    },
    std::{io::Cursor, path::Path},
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
    let file = Cursor::new(read(audio_path).await?);
    let decoder = Decoder::new(file)?;
    let channels = if mono { 1 } else { decoder.channels() } as usize;
    let samples = UniformSourceIterator::new(decoder, channels as _, SR as _).collect::<Vec<f32>>();

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
/// `Vec<f32>`: 重采样后的样本数组
///
/// # 示例
/// ```
/// use voxudio::resample;
/// let samples = vec![0.1, 0.2, 0.3, 0.4];
/// let resampled = resample::<44100, 48000>(&samples, 1, 2);
/// ```
pub fn resample<const SSR: usize, const TSR: usize>(
    samples: &[f32],
    src_channels: usize,
    tgt_channels: usize,
) -> Vec<f32> {
    if samples.is_empty() || src_channels == 0 || tgt_channels == 0 {
        return Vec::new();
    }

    UniformSourceIterator::new(
        SamplesBuffer::new(src_channels as _, SSR as _, samples),
        tgt_channels as _,
        TSR as _,
    )
    .collect()
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
/// `Vec<f32>`: 空间化处理后的样本数组
///
/// # 示例
/// ```
/// use voxudio::spatial_audio;
/// let samples = vec![0.1, 0.2, 0.3, 0.4];
/// let processed = spatial_audio::<44100>(
///     &samples,
///     2,
///     [0.0, 0.0, 0.0],
///     [-0.1, 0.0, 0.0],
///     [0.1, 0.0, 0.0]
/// );
/// ```
pub fn spatial_audio<const SR: usize>(
    audio: &[f32],
    channels: usize,
    emitter_position: [f32; 3],
    left_ear: [f32; 3],
    right_ear: [f32; 3],
) -> Vec<f32> {
    Spatial::new(
        SamplesBuffer::new(channels as _, SR as _, audio),
        emitter_position,
        left_ear,
        right_ear,
    )
    .collect()
}
