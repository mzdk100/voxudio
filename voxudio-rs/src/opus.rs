mod raw;

use {crate::error::OperationError, std::ptr::null};

/// 封装Opus应用类型为枚举
///
/// 定义了Opus编码器的不同应用场景，每种场景针对不同的音频类型和使用场景进行了优化。
/// 选择合适的应用类型可以显著影响编码质量和延迟。
#[derive(Debug, Clone, Copy)]
pub enum OpusApplication {
    /// 最适合大多数音频信号的通用编码
    ///
    /// 为音乐和混合内容提供最佳质量，适用于广播和存储应用。
    Audio,
    /// 针对语音信号优化的编码
    ///
    /// 为语音通话提供更好的性能，适用于VoIP和视频会议应用。
    Voip,
    /// 针对低延迟应用的编码
    ///
    /// 提供最低的算法延迟，适用于实时交互式应用。
    RestrictedLowdelay,
}

impl OpusApplication {
    // 转换为Opus C API使用的整数值
    fn to_opus_int(&self) -> i32 {
        match self {
            Self::Audio => raw::OPUS_APPLICATION_AUDIO as i32,
            Self::Voip => raw::OPUS_APPLICATION_VOIP as i32,
            Self::RestrictedLowdelay => raw::OPUS_APPLICATION_RESTRICTED_LOWDELAY as i32,
        }
    }
}

/// 封装Opus带宽设置为枚举
///
/// 定义了Opus编码器支持的不同带宽级别，影响音频质量、比特率和计算复杂度。
/// 带宽越高，音频质量越好，但需要更高的比特率和计算资源。
#[derive(Debug, Clone, Copy)]
pub enum OpusBandwidth {
    /// 4kHz带宽
    ///
    /// 适用于低比特率语音通信，类似于电话质量。
    Narrowband,
    /// 6kHz带宽
    ///
    /// 提供比窄带更好的语音质量，但仍然主要针对语音优化。
    Mediumband,
    /// 8kHz带宽
    ///
    /// 提供良好的语音质量和基本的音乐质量，适合大多数VoIP应用。
    Wideband,
    /// 12kHz带宽
    ///
    /// 提供高质量的语音和音乐，适合高质量的音频会议和音乐流媒体。
    Superwideband,
    /// 20kHz带宽（全频带）
    ///
    /// 提供最高质量的音频，覆盖人类听力的全部范围，适合高保真音乐。
    Fullband,
    /// 自定义带宽值
    ///
    /// 允许使用自定义的带宽设置，用于特殊应用场景。
    Custom(i32),
}

impl OpusBandwidth {
    // 转换为Opus C API使用的整数值
    fn to_opus_int(&self) -> i32 {
        match self {
            Self::Narrowband => raw::OPUS_BANDWIDTH_NARROWBAND as i32,
            Self::Mediumband => raw::OPUS_BANDWIDTH_MEDIUMBAND as i32,
            Self::Wideband => raw::OPUS_BANDWIDTH_WIDEBAND as i32,
            Self::Superwideband => raw::OPUS_BANDWIDTH_SUPERWIDEBAND as i32,
            Self::Fullband => raw::OPUS_BANDWIDTH_FULLBAND as i32,
            Self::Custom(value) => *value,
        }
    }
}

/// Opus音频样本类型特征
///
/// 定义了可用于Opus编解码的音频样本类型必须实现的方法。
/// 这个特征允许编解码器与不同的音频样本格式（如i16和f32）一起工作。
pub trait OpusSample {
    /// 检查样本类型是否为f32
    ///
    /// 用于内部决定使用哪个Opus API函数进行编解码。
    fn is_f32() -> bool {
        false
    }

    /// 检查样本类型是否为i16
    ///
    /// 用于内部决定使用哪个Opus API函数进行编解码。
    fn is_i16() -> bool {
        false
    }

    /// 创建一个值为零的样本
    ///
    /// 用于初始化音频缓冲区。
    fn zero() -> Self;
}

impl OpusSample for i16 {
    fn is_i16() -> bool {
        true
    }

    fn zero() -> Self {
        0
    }
}

impl OpusSample for f32 {
    fn is_f32() -> bool {
        true
    }

    fn zero() -> Self {
        0f32
    }
}

/// 安全封装Opus编码器
///
/// 提供高质量、低延迟的音频编码功能，支持多种采样率和通道配置。
/// 可以通过各种设置方法调整编码参数，如比特率、复杂度和带宽。
pub struct OpusEncoder {
    encoder: *mut raw::OpusEncoder,
    channels: usize,
}

impl OpusEncoder {
    // 创建一个新的编码器，使用枚举类型作为应用类型参数
    pub(self) fn new(
        sample_rate: usize,
        channels: usize,
        application: OpusApplication,
    ) -> Result<Self, OperationError> {
        let mut error = 0;
        let encoder = unsafe {
            raw::opus_encoder_create(
                sample_rate as _,
                channels as _,
                application.to_opus_int(),
                &mut error,
            )
        };

        if error != raw::OPUS_OK as i32 {
            return Err(OperationError::Opus(format!(
                "Failed to create Opus encoder: {}",
                error
            )));
        }

        if encoder.is_null() {
            return Err(OperationError::Opus("Opus encoder is null".to_owned()));
        }

        Ok(Self { encoder, channels })
    }

    /// 设置编码器位速率
    ///
    /// # 参数
    ///
    /// * `bitrate` - 目标位速率（比特/秒）。可以设置为特定值（如64000表示64kbps）
    ///   或使用以下特殊值：
    ///   - 0: 自动位速率（由编码器根据信号特性选择）
    ///   - 1: 最大位速率（质量优先）
    ///
    /// # 返回值
    ///
    /// 成功时返回Ok(())，失败时返回错误
    ///
    /// # 示例
    ///
    /// ```no_run
    /// # use voxudio::{OpusEncoder, OpusApplication, OpusCodec};
    /// # let mut encoder = OpusCodec::new_encoder(48000, 2, OpusApplication::Audio).unwrap();
    /// // 设置为64kbps
    /// encoder.set_bitrate(64000).unwrap();
    /// ```
    pub fn set_bitrate(&mut self, bitrate: usize) -> Result<(), OperationError> {
        let result = unsafe {
            raw::opus_encoder_ctl(self.encoder, raw::OPUS_SET_BITRATE_REQUEST as _, bitrate)
        };

        if result != raw::OPUS_OK as i32 {
            return Err(OperationError::Opus(format!(
                "Failed to set bitrate: {}",
                result
            )));
        }

        Ok(())
    }

    /// 设置编码器复杂度
    ///
    /// 复杂度控制编码器的计算复杂性，较高的值提供更好的质量但需要更多CPU资源。
    ///
    /// # 参数
    ///
    /// * `complexity` - 复杂度值，范围从0（最低复杂度）到10（最高复杂度）
    ///
    /// # 返回值
    ///
    /// 成功时返回Ok(())，失败时返回错误
    ///
    /// # 示例
    ///
    /// ```no_run
    /// # use voxudio::{OpusEncoder, OpusApplication, OpusCodec};
    /// # let mut encoder = OpusCodec::new_encoder(48000, 2, OpusApplication::Audio).unwrap();
    /// // 设置为中等复杂度
    /// encoder.set_complexity(5).unwrap();
    /// ```
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), OperationError> {
        let result = unsafe {
            raw::opus_encoder_ctl(
                self.encoder,
                raw::OPUS_SET_COMPLEXITY_REQUEST as _,
                complexity,
            )
        };

        if result != raw::OPUS_OK as i32 {
            return Err(OperationError::Opus(format!(
                "Failed to set complexity: {}",
                result
            )));
        }

        Ok(())
    }

    /// 设置信号带宽
    ///
    /// 带宽控制编码音频的频率范围，影响音质和压缩效率。
    ///
    /// # 参数
    ///
    /// * `bandwidth` - 带宽设置，参见[`OpusBandwidth`]枚举
    ///
    /// # 返回值
    ///
    /// 成功时返回Ok(())，失败时返回错误
    ///
    /// # 示例
    ///
    /// ```no_run
    /// # use voxudio::{OpusEncoder, OpusApplication, OpusCodec, OpusBandwidth};
    /// # let mut encoder = OpusCodec::new_encoder(48000, 2, OpusApplication::Audio).unwrap();
    /// // 设置为宽带（8kHz带宽）
    /// encoder.set_bandwidth(OpusBandwidth::Wideband).unwrap();
    /// ```
    pub fn set_bandwidth(&mut self, bandwidth: OpusBandwidth) -> Result<(), OperationError> {
        let result = unsafe {
            raw::opus_encoder_ctl(
                self.encoder,
                raw::OPUS_SET_BANDWIDTH_REQUEST as _,
                bandwidth.to_opus_int(),
            )
        };

        if result != raw::OPUS_OK as i32 {
            return Err(OperationError::Opus(format!(
                "Failed to set bandwidth: {}",
                result
            )));
        }

        Ok(())
    }

    /// 编码音频数据
    ///
    /// 将PCM音频数据编码为Opus压缩格式。
    ///
    /// # 参数
    ///
    /// * `pcm` - 输入PCM音频数据，类型必须实现[`OpusSample`]特征
    /// * `frame_size` - 每个通道的帧大小（样本数），通常为2.5, 5, 10, 20, 40或60毫秒对应的样本数
    /// * `max_data_bytes` - 输出缓冲区的最大大小（字节），建议使用[`OpusCodec::MAX_PACKET_SIZE`]
    ///
    /// # 返回值
    ///
    /// 成功时返回编码后的数据，失败时返回错误
    ///
    /// # 示例
    ///
    /// ```no_run
    /// # use voxudio::{OpusEncoder, OpusApplication, OpusCodec};
    /// # let encoder = OpusCodec::new_encoder(48000, 2, OpusApplication::Audio).unwrap();
    /// # let pcm_data: Vec<i16> = vec![0; 960 * 2]; // 20ms at 48kHz stereo
    /// // 编码20ms的音频数据
    /// let frame_size = 960; // 48000 Hz * 20ms / 1000 = 960 samples
    /// let encoded = encoder.encode(&pcm_data, frame_size, OpusCodec::MAX_PACKET_SIZE).unwrap();
    /// ```
    pub fn encode<S: OpusSample>(
        &self,
        pcm: &[S],
        frame_size: usize,
        max_data_bytes: usize,
    ) -> Result<Vec<u8>, OperationError> {
        let samples_per_channel = pcm.len() / self.channels;

        // 检查输入数据是否足够
        if samples_per_channel < frame_size as usize {
            return Err(OperationError::Opus(format!(
                "Input data too short for encoding: need {} samples per channel, got {}",
                frame_size, samples_per_channel
            )));
        }

        // 额外空间给TOC头部
        let mut encoded = vec![0u8; max_data_bytes + 1];

        let encoded_len = unsafe {
            if S::is_i16() {
                raw::opus_encode(
                    self.encoder,
                    pcm.as_ptr() as _,
                    frame_size as _,
                    encoded.as_mut_ptr(),
                    max_data_bytes as _,
                )
            } else {
                raw::opus_encode_float(
                    self.encoder,
                    pcm.as_ptr() as _,
                    frame_size as _,
                    encoded.as_mut_ptr(),
                    max_data_bytes as _,
                )
            }
        };

        if encoded_len < 0 {
            return Err(OperationError::Opus(format!(
                "Failed to encode: {}",
                encoded_len
            )));
        }

        // 包含TOC头部
        encoded.truncate(encoded_len as usize + 1);
        Ok(encoded)
    }

    /// 获取当前编码器的通道数
    ///
    /// # 返回值
    ///
    /// 返回编码器配置的通道数
    ///
    /// # 示例
    ///
    /// ```no_run
    /// # use voxudio::{OpusEncoder, OpusApplication, OpusCodec};
    /// # let encoder = OpusCodec::new_encoder(48000, 2, OpusApplication::Audio).unwrap();
    /// assert_eq!(encoder.channels(), 2);
    /// ```
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Drop for OpusEncoder {
    fn drop(&mut self) {
        if !self.encoder.is_null() {
            unsafe {
                raw::opus_encoder_destroy(self.encoder);
            }
        }
    }
}

/// 安全封装Opus解码器
///
/// 提供高质量、低延迟的音频解码功能，支持多种采样率和通道配置。
/// 能够处理丢包情况，并提供丢包隐藏(PLC)功能。
pub struct OpusDecoder {
    decoder: *mut raw::OpusDecoder,
    channels: usize,
}

impl OpusDecoder {
    // 创建一个新的解码器
    pub(self) fn new(sample_rate: usize, channels: usize) -> Result<Self, OperationError> {
        let mut error = 0;
        let decoder =
            unsafe { raw::opus_decoder_create(sample_rate as _, channels as _, &mut error) };

        if error != raw::OPUS_OK as i32 {
            return Err(OperationError::Opus(format!(
                "Failed to create Opus decoder: {}",
                error
            )));
        }

        if decoder.is_null() {
            return Err(OperationError::Opus("Opus decoder is null".to_owned()));
        }

        Ok(Self { decoder, channels })
    }

    /// 解码音频数据
    ///
    /// 将Opus压缩格式解码为PCM音频数据。支持丢包隐藏(PLC)功能。
    ///
    /// # 参数
    ///
    /// * `data` - 输入的Opus编码数据，如果为None则执行丢包隐藏
    /// * `frame_size` - 每个通道的帧大小（样本数）
    ///
    /// # 返回值
    ///
    /// 成功时返回解码后的PCM数据，失败时返回错误
    ///
    /// # 示例
    ///
    /// ```no_run
    /// # use voxudio::{OpusDecoder, OpusCodec};
    /// # let decoder = OpusCodec::new_decoder(48000, 2).unwrap();
    /// # let encoded_data = vec![0u8; 100];
    /// // 解码音频数据
    /// let frame_size = 960; // 48000 Hz * 20ms / 1000 = 960 samples
    /// let decoded = decoder.decode::<i16>(Some(&encoded_data), frame_size).unwrap();
    ///
    /// // 处理丢包情况
    /// let concealed = decoder.decode::<i16>(None, frame_size).unwrap();
    /// ```
    pub fn decode<S: Clone + OpusSample>(
        &self,
        data: Option<&[u8]>,
        frame_size: usize,
    ) -> Result<Vec<S>, OperationError> {
        // 为输出分配足够的空间（frame_size * channels）
        let mut decoded = vec![S::zero(); frame_size * self.channels];

        let decoded_samples = unsafe {
            if let Some(data) = data {
                raw::opus_decode(
                    self.decoder,
                    data.as_ptr(),
                    data.len() as _,
                    decoded.as_mut_ptr() as _,
                    frame_size as _, // 这里传入每个通道的输出样本数
                    0,               // 不使用FEC
                )
            } else {
                // 解码丢失的包（PLC - Packet Loss Concealment）
                raw::opus_decode(
                    self.decoder,
                    null(),
                    0,
                    decoded.as_mut_ptr() as _,
                    frame_size as _, // 这里传入每个通道的输出样本数
                    0,
                )
            }
        };

        if decoded_samples < 0 {
            return Err(OperationError::Opus(format!(
                "Failed to decode: {}",
                decoded_samples
            )));
        }
        // 调整输出大小为实际解码的样本数 * 通道数
        decoded.truncate(decoded_samples as usize * self.channels);

        Ok(decoded)
    }
}

impl Drop for OpusDecoder {
    fn drop(&mut self) {
        if !self.decoder.is_null() {
            unsafe {
                raw::opus_decoder_destroy(self.decoder);
            }
        }
    }
}

/// Opus编解码器的主要接口
///
/// 提供创建Opus编码器和解码器的方法，以及一些实用功能如：
/// - 计算帧大小
/// - 获取最大带宽
/// - 获取版本信息
///
/// # 示例
/// ```
/// use voxudio::*;
///
/// // 创建编码器
/// let _encoder = OpusCodec::new_encoder(48000, 2, OpusApplication::Audio).unwrap();
///
/// // 创建解码器
/// let _decoder = OpusCodec::new_decoder(48000, 2).unwrap();
///
/// // 计算20ms的帧大小
/// let _frame_size = OpusCodec::calculate_frame_size(48000, 20);
///
/// // 获取版本
/// let _version = OpusCodec::version();
/// ```
pub struct OpusCodec;

impl OpusCodec {
    /// 最大包大小（字节）
    pub const MAX_PACKET_SIZE: usize = 1500;

    /// 创建一个新的Opus编码器
    ///
    /// # 参数
    ///
    /// * `sample_rate` - 采样率，必须是8000, 12000, 16000, 24000或48000 Hz
    /// * `channels` - 通道数，支持1(单声道)或2(立体声)
    /// * `application` - 应用类型，参见[`OpusApplication`]枚举
    ///
    /// # 返回值
    ///
    /// 成功时返回Opus编码器实例，失败时返回错误
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use voxudio::{OpusCodec, OpusApplication};
    ///
    /// // 创建一个48kHz立体声音频编码器
    /// let encoder = OpusCodec::new_encoder(48000, 2, OpusApplication::Audio).unwrap();
    /// ```
    pub fn new_encoder(
        sample_rate: usize,
        channels: usize,
        application: OpusApplication,
    ) -> Result<OpusEncoder, OperationError> {
        OpusEncoder::new(sample_rate, channels, application)
    }

    /// 创建一个新的Opus解码器
    ///
    /// # 参数
    ///
    /// * `sample_rate` - 采样率，必须是8000, 12000, 16000, 24000或48000 Hz
    /// * `channels` - 通道数，支持1(单声道)或2(立体声)
    ///
    /// # 返回值
    ///
    /// 成功时返回Opus解码器实例，失败时返回错误
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use voxudio::OpusCodec;
    ///
    /// // 创建一个48kHz立体声音频解码器
    /// let decoder = OpusCodec::new_decoder(48000, 2).unwrap();
    /// ```
    pub fn new_decoder(sample_rate: usize, channels: usize) -> Result<OpusDecoder, OperationError> {
        OpusDecoder::new(sample_rate, channels)
    }

    /// 计算指定采样率和时长的帧大小（样本数）
    ///
    /// # 参数
    ///
    /// * `sample_rate` - 音频采样率（Hz）
    /// * `duration_ms` - 音频帧时长（毫秒）
    ///
    /// # 返回值
    ///
    /// 返回每个通道的样本数
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use voxudio::OpusCodec;
    ///
    /// // 计算48kHz采样率下20ms的帧大小
    /// let frame_size = OpusCodec::calculate_frame_size(48000, 20);
    /// assert_eq!(frame_size, 960); // 48000 * 20 / 1000 = 960
    /// ```
    pub fn calculate_frame_size(sample_rate: usize, duration_ms: u64) -> usize {
        (sample_rate * duration_ms as usize) / 1000
    }

    /// 获取采样率对应的最大带宽
    ///
    /// 根据给定的采样率返回Opus支持的最大带宽设置。
    /// 采样率越高，支持的带宽越大，音频质量越好。
    ///
    /// # 参数
    ///
    /// * `sample_rate` - 音频采样率（Hz）
    ///
    /// # 返回值
    ///
    /// 返回对应采样率的最大带宽设置，参见[`OpusBandwidth`]枚举
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use voxudio::{OpusCodec, OpusBandwidth};
    ///
    /// // 获取48kHz采样率对应的最大带宽
    /// let bandwidth = OpusCodec::get_max_bandwidth_for_sample_rate(48000);
    /// assert!(matches!(bandwidth, OpusBandwidth::Fullband));
    /// ```
    pub fn get_max_bandwidth_for_sample_rate(sample_rate: usize) -> OpusBandwidth {
        match sample_rate {
            8000 => OpusBandwidth::Narrowband,
            12000 => OpusBandwidth::Mediumband,
            16000 => OpusBandwidth::Wideband,
            24000 => OpusBandwidth::Superwideband,
            _ => OpusBandwidth::Fullband, // 48000及其他采样率使用全频带
        }
    }

    /// 获取版本
    pub fn version() -> String {
        unsafe {
            let version_ptr = raw::opus_get_version_string();
            let c_str = std::ffi::CStr::from_ptr(version_ptr);
            c_str.to_string_lossy().to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cmp::min, f32::consts::PI};

    // 计算两个音频信号之间的信噪比（SNR）
    fn calculate_snr(original: &[i16], decoded: &[i16]) -> f32 {
        let len = std::cmp::min(original.len(), decoded.len());
        let mut signal_power = 0.0;
        let mut noise_power = 0.0;

        for i in 0..len {
            let signal = original[i] as f32;
            let noise = signal - decoded[i] as f32;

            signal_power += signal * signal;
            noise_power += noise * noise;
        }

        if noise_power > 0.0 && signal_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f32::INFINITY // 如果没有噪声，SNR为无穷大
        }
    }

    // 生成一个更复杂的音频信号（包含多个频率）
    fn generate_complex_audio(duration_ms: usize, sample_rate: usize) -> Vec<i16> {
        let num_samples = sample_rate * duration_ms / 1000;
        let mut samples = Vec::with_capacity(num_samples * 2); // 立体声

        // 根据采样率调整频率，确保不会超过奈奎斯特频率
        // 对于8kHz采样率，最高频率应低于4kHz
        let max_freq = (sample_rate as f32 / 2.0) * 0.8; // 留出20%的余量

        // 根据采样率选择合适的频率
        let frequencies = [261.63, 329.63, 392.00, 523.25, 659.25]; // C4, E4, G4, C5, E5

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;

            // 组合多个频率
            let mut value = 0.0;
            for (j, &freq) in frequencies.iter().enumerate() {
                // 确保频率不超过最大允许频率
                if freq < max_freq {
                    // 每个频率的振幅逐渐减小
                    let amplitude = 0.1 / (j + 1) as f32; // 进一步降低振幅以避免溢出
                    value += amplitude * f32::sin(2.0 * PI * freq * t);
                }
            }

            // 添加一些确定性"噪声"，但确保噪声频率不超过最大允许频率
            let noise_freq = max_freq * 0.5; // 噪声频率为最大允许频率的一半
            let noise = f32::sin(t * noise_freq) * f32::cos(t * (noise_freq * 0.7) * 0.005);
            value += noise;

            // 确保值在[-0.8, 0.8]范围内，进一步避免溢出
            value = value.max(-0.8).min(0.8);

            // 归一化并转换为i16，使用安全的转换方式
            let sample = (value * 30000.0) as i16; // 使用30000而不是32767，留出更多余量

            // 左右声道略有不同
            samples.push(sample); // 左声道
            samples.push((sample as f32 * 0.95) as i16); // 右声道（稍微安静一点）
        }

        samples
    }

    // 测试基本的编解码功能
    fn test_basic_encoding_decoding(
        sample_rate: usize,
        channels: usize,
        bitrate: usize,
    ) -> Result<(), OperationError> {
        println!("\n--- Testing basic encode/decode ({} Hz) ---", sample_rate);

        let frame_size = sample_rate * 20 / 1000;

        let input_audio = generate_complex_audio(500, sample_rate);

        let mut encoder = OpusCodec::new_encoder(sample_rate, channels, OpusApplication::Audio)?;
        encoder.set_bitrate(bitrate)?;
        encoder.set_complexity(10)?;
        encoder.set_bandwidth(OpusCodec::get_max_bandwidth_for_sample_rate(sample_rate))?;

        let decoder = OpusDecoder::new(sample_rate, channels)?;

        println!("Encoding and decoding multiple frames...");
        let frame_count = 5;
        let mut all_input = Vec::new();
        let mut all_decoded = Vec::new();

        for i in 0..frame_count {
            let start = i * frame_size * channels;
            let end = start + frame_size * channels;
            let end = end.min(input_audio.len());

            if start >= input_audio.len() {
                break;
            }

            let frame_input = &input_audio[start..end];
            all_input.extend_from_slice(frame_input);

            let encoded = encoder.encode(frame_input, frame_size, OpusCodec::MAX_PACKET_SIZE)?;
            println!("  Frame {}: encoded {} bytes", i + 1, encoded.len());

            let decoded = decoder.decode::<i16>(Some(&encoded), frame_size)?;
            println!("  Frame {}: decoded {} samples", i + 1, decoded.len());
            all_decoded.extend(decoded);
        }

        println!("Frame size: {} samples (20 ms)", frame_size);
        println!("Total input samples: {}", all_input.len());
        println!("Total decoded samples: {}", all_decoded.len());

        if !all_decoded.is_empty() {
            let min_len = min(all_input.len(), all_decoded.len());
            let snr = calculate_snr(&all_input[0..min_len], &all_decoded[0..min_len]);
            println!("Signal-to-noise ratio (SNR): {:.2} dB", snr);
        }

        Ok(())
    }

    #[test]
    fn test_opus() -> anyhow::Result<()> {
        // Output Opus version information
        println!("Opus version: {}", crate::opus::OpusCodec::version());

        println!("\n=== Opus codec test - Basic encoding/decoding functionality ===");

        // Test different sample rates
        let sample_rates = [8000, 16000, 24000, 48000];
        let channels = 2;
        let bitrate = 256000; // 256 kbps

        for &rate in &sample_rates {
            test_basic_encoding_decoding(rate, channels, bitrate)?;
        }

        Ok(())
    }
}
