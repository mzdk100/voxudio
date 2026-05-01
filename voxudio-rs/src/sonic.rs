mod raw;

use crate::OperationError;

/// Sonic 音频样本类型特征
///
/// 定义了可用于 Sonic 变速处理的音频样本类型必须实现的方法。
/// 这个特征允许 SonicStream 与不同的音频样本格式（如 i16 和 f32）一起工作。
pub trait SonicSample: Clone {
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
}

impl SonicSample for f32 {
    fn is_f32() -> bool {
        true
    }

    fn zero() -> Self {
        0.0
    }
}

impl SonicSample for i16 {
    fn is_i16() -> bool {
        true
    }

    fn zero() -> Self {
        0
    }
}

/// Sonic 音频变速不变调处理器
///
/// 封装了 [Sonic](https://github.com/waywardgeek/sonic) C 库，提供高质量的音频变速不变调（time-stretch）处理。
/// 与简单的采样率调整不同，Sonic 通过 WSOLA/PICOLA 算法实现变速时保持音高不变，
/// 特别适合语音加速/减速场景。
///
/// # 主要功能
/// - 变速不变调（speed）：调整播放速度而不改变音高
/// - 变调（pitch）：独立调整音高
/// - 变速变调（rate）：同时调整速度和音高（类似于留声机效果）
/// - 音量调节（volume）
///
/// # 示例
/// ```no_run
/// use voxudio::SonicStream;
///
/// fn main() -> anyhow::Result<()> {
///     // 创建流式处理器
///     let mut stream = SonicStream::new(16000, 1)?;
///     // 设置1.5倍速（音高不变）
///     stream.set_speed(1.5);
///     // 写入 f32 音频数据
///     stream.write::<f32>(&[0.1, 0.2, 0.3])?;
///     // 刷新缓冲区
///     stream.flush()?;
///     // 读取处理后的数据
///     let output = stream.read::<f32>(1024);
///
///     Ok(())
/// }
/// ```
pub struct SonicStream {
    stream: *mut raw::sonicStreamStruct,
    #[allow(dead_code)]
    sample_rate: usize,
    channels: usize,
}

impl SonicStream {
    /// 创建新的 Sonic 流处理器
    ///
    /// # 参数
    /// - `sample_rate`: 采样率（1000-500000 Hz）
    /// - `channels`: 声道数（1-32）
    ///
    /// # 返回
    /// 成功返回 `SonicStream` 实例，失败返回错误
    ///
    /// # 示例
    /// ```
    /// use voxudio::SonicStream;
    /// fn main() -> anyhow::Result<()> {
    ///     let stream = SonicStream::new(16000, 1)?;
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn new(sample_rate: usize, channels: usize) -> Result<Self, OperationError> {
        let stream = unsafe { raw::sonicCreateStream(sample_rate as _, channels as _) };
        if stream.is_null() {
            return Err(OperationError::Sonic(
                "Failed to create sonic stream: out of memory".to_owned(),
            ));
        }

        Ok(Self {
            stream,
            sample_rate,
            channels,
        })
    }

    /// 向流中写入音频数据
    ///
    /// 对于交错排列的多声道数据，直接传入整个数组，`num_samples` 自动按每声道计算。
    ///
    /// # 参数
    /// - `samples`: 音频样本数组（交错排列），类型必须实现 [`SonicSample`] 特征
    ///
    /// # 返回
    /// 成功返回 `Ok(())`，失败返回错误
    ///
    /// # 示例
    /// ```no_run
    /// use voxudio::SonicStream;
    /// fn main() -> anyhow::Result<()> {
    ///     let mut stream = SonicStream::new(16000, 1)?;
    ///     // 写入 f32 数据
    ///     stream.write::<f32>(&[0.1, 0.2, 0.3])?;
    ///     // 写入 i16 数据
    ///     stream.write::<i16>(&[100, 200, 300])?;
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn write<S>(&mut self, samples: &[S]) -> Result<(), OperationError>
    where
        S: SonicSample,
    {
        let num_samples = samples.len() / self.channels;
        let ret = unsafe {
            if S::is_f32() {
                raw::sonicWriteFloatToStream(self.stream, samples.as_ptr() as _, num_samples as _)
            } else {
                raw::sonicWriteShortToStream(self.stream, samples.as_ptr() as _, num_samples as _)
            }
        };
        if ret == 0 {
            return Err(OperationError::Sonic(
                "Failed to write samples to sonic stream".to_owned(),
            ));
        }

        Ok(())
    }

    /// 向流中写入音频数据（指定每声道样本数）
    ///
    /// # 参数
    /// - `samples`: 音频样本数组
    /// - `num_samples`: 每声道的样本数
    pub fn write_with_num<S>(
        &mut self,
        samples: &[S],
        num_samples: usize,
    ) -> Result<(), OperationError>
    where
        S: SonicSample,
    {
        let ret = unsafe {
            if S::is_f32() {
                raw::sonicWriteFloatToStream(self.stream, samples.as_ptr() as _, num_samples as _)
            } else {
                raw::sonicWriteShortToStream(self.stream, samples.as_ptr() as _, num_samples as _)
            }
        };
        if ret == 0 {
            return Err(OperationError::Sonic(
                "Failed to write samples to sonic stream".to_owned(),
            ));
        }
        Ok(())
    }

    /// 从流中读取处理后的音频数据
    ///
    /// # 参数
    /// - `max_samples`: 最多读取的每声道样本数
    ///
    /// # 返回
    /// 读取到的样本数组（可能为空，不表示错误）
    ///
    /// # 示例
    /// ```no_run
    /// use voxudio::SonicStream;
    /// fn main() -> anyhow::Result<()> {
    ///     let mut stream = SonicStream::new(16000, 1)?;
    ///     // 读取 f32 数据
    ///     let output_f32 = stream.read::<f32>(1024);
    ///     // 读取 i16 数据
    ///     let output_i16 = stream.read::<i16>(1024);
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn read<S>(&mut self, max_samples: usize) -> Vec<S>
    where
        S: SonicSample,
    {
        let total_samples = max_samples * self.channels;
        let mut buf = vec![S::zero(); total_samples];
        let read = unsafe {
            if S::is_f32() {
                raw::sonicReadFloatFromStream(self.stream, buf.as_mut_ptr() as _, max_samples as _)
            } else {
                raw::sonicReadShortFromStream(self.stream, buf.as_mut_ptr() as _, max_samples as _)
            }
        };
        buf.truncate(read as usize * self.channels);

        buf
    }

    /// 刷新流，强制生成所有剩余输出
    ///
    /// 在输入完所有数据后调用，确保所有样本都被处理。
    /// 在语音中间刷新可能引入失真。
    pub fn flush(&mut self) -> Result<(), OperationError> {
        let ret = unsafe { raw::sonicFlushStream(self.stream) };
        if ret == 0 {
            return Err(OperationError::Sonic(
                "Failed to flush sonic stream".to_owned(),
            ));
        }

        Ok(())
    }

    /// 获取输出缓冲区中可用的样本数（每声道）
    pub fn get_samples_available(&self) -> usize {
        unsafe { raw::sonicSamplesAvailable(self.stream) as usize }
    }

    /// 获取当前速度因子
    pub fn get_speed(&self) -> f32 {
        unsafe { raw::sonicGetSpeed(self.stream) }
    }

    /// 设置速度因子（变速不变调）
    ///
    /// - `speed > 1.0`: 加速，音高不变
    /// - `speed < 1.0`: 减速，音高不变
    /// - `speed = 1.0`: 原始速度
    ///
    /// 有效范围：0.05 - 20.0
    pub fn set_speed(&mut self, speed: f32) {
        unsafe { raw::sonicSetSpeed(self.stream, speed) }
    }

    /// 获取当前音高因子
    pub fn get_pitch(&self) -> f32 {
        unsafe { raw::sonicGetPitch(self.stream) }
    }

    /// 设置音高因子
    ///
    /// - `pitch > 1.0`: 音高升高
    /// - `pitch < 1.0`: 音高降低
    ///
    /// 有效范围：0.05 - 20.0
    pub fn set_pitch(&mut self, pitch: f32) {
        unsafe { raw::sonicSetPitch(self.stream, pitch) }
    }

    /// 获取当前速率因子
    pub fn get_rate(&self) -> f32 {
        unsafe { raw::sonicGetRate(self.stream) }
    }

    /// 设置速率因子（变速变调，类似留声机效果）
    ///
    /// 同时改变速度和音高，rate 改变播放速率，
    /// 而 pitch 在 rate 基础上进一步调整音高。
    /// 最终效果：speed 控制变速不变调，rate*pitch 控制采样率转换。
    ///
    /// 有效范围：0.05 - 20.0
    pub fn set_rate(&mut self, rate: f32) {
        unsafe { raw::sonicSetRate(self.stream, rate) }
    }

    /// 获取当前音量因子
    pub fn get_volume(&self) -> f32 {
        unsafe { raw::sonicGetVolume(self.stream) }
    }

    /// 设置音量因子
    ///
    /// 有效范围：0.01 - 100.0，1.0 为原始音量
    pub fn set_volume(&mut self, volume: f32) {
        unsafe { raw::sonicSetVolume(self.stream, volume) }
    }

    /// 获取质量设置
    pub fn get_quality(&self) -> i32 {
        unsafe { raw::sonicGetQuality(self.stream) }
    }

    /// 设置质量
    ///
    /// - `0`: 默认，速度更快，质量几乎与 1 相同
    /// - `1`: 高质量，更慢
    pub fn set_quality(&mut self, quality: i32) {
        unsafe { raw::sonicSetQuality(self.stream, quality) }
    }

    /// 获取采样率
    pub fn get_sample_rate(&self) -> usize {
        unsafe { raw::sonicGetSampleRate(self.stream) as usize }
    }

    /// 设置采样率（会丢弃未读取的缓冲样本）
    pub fn set_sample_rate(&mut self, sample_rate: usize) {
        unsafe { raw::sonicSetSampleRate(self.stream, sample_rate as _) }
    }

    /// 获取声道数
    pub fn get_channels(&self) -> usize {
        unsafe { raw::sonicGetNumChannels(self.stream) as usize }
    }

    /// 设置声道数（会丢弃未读取的缓冲样本）
    pub fn set_channels(&mut self, channels: usize) {
        unsafe { raw::sonicSetNumChannels(self.stream, channels as _) }
    }

    /// 一次性变速不变调处理
    ///
    /// 便捷方法，使用流式 API 内部处理整段音频数据，适合不需要流式处理的场景。
    ///
    /// # 参数
    /// - `samples`: 音频样本数组（交错排列），类型必须实现 [`SonicSample`] 特征
    /// - `sample_rate`: 采样率
    /// - `channels`: 声道数
    /// - `speed`: 速度因子（变速不变调，1.0 为原始速度）
    ///
    /// # 返回
    /// 处理后的样本数组
    ///
    /// # 示例
    /// ```no_run
    /// use voxudio::SonicStream;
    /// fn main() -> anyhow::Result<()> {
    ///     let mut stream = SonicStream::new(16000, 1)?;
    ///     stream.set_speed(1.5);
    ///     let input = vec![0.0f32; 1024];
    ///     let output = stream.change_speed(&input)?;
    ///
    ///     Ok(())
    /// }
    /// ```
    pub fn change_speed<S>(&mut self, samples: &[S]) -> Result<Vec<S>, OperationError>
    where
        S: SonicSample,
    {
        if samples.is_empty() {
            return Ok(Default::default());
        }
        if self.get_speed() == 1.0 {
            return Ok(samples.to_owned());
        }

        self.write::<S>(samples)?;
        self.flush()?;
        let available = self.get_samples_available();

        Ok(self.read::<S>(available))
    }
}

impl Drop for SonicStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                raw::sonicDestroyStream(self.stream);
            }
        }
    }
}
