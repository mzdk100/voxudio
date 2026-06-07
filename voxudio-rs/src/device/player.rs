use {
    crate::{GenericSample, OperationError, samples_to_f32},
    cpal::{
        BufferSize, Device, HostId, Stream, StreamConfig, SupportedStreamConfig, default_host,
        traits::{DeviceTrait, HostTrait, StreamTrait},
    },
    rodio::{ChannelCount, SampleRate, buffer::SamplesBuffer, source::UniformSourceIterator},
    std::{
        fmt::{Debug, Error as FmtError, Formatter, Result as FmtResult},
        io::Error as IoError,
        mem::replace,
    },
    tokio::sync::{
        mpsc::{Sender as MpscSender, channel as mpsc_channel},
        watch::{Sender as WatchSender, channel as watch_channel},
    },
};

/// 音频播放器结构体，用于管理和控制音频播放
///
/// # 字段说明
/// - `device`: 音频输出设备
/// - `host_id`: 音频主机ID
/// - `stream_config`: 音频流配置
/// - `supported_stream_config`: 支持的音频流配置
/// - `stream`: 音频流实例
/// - `sender`: 音频数据发送通道
///
/// # 示例
/// ```
/// use voxudio::AudioPlayer;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
/// let mut player = AudioPlayer::new()?;
/// player.play()?;
/// player.write::<44100, f32>(&[0f32; 88200], 2).await?;
///
/// Ok(())
/// }
/// ```
///
/// # 错误
/// 可能返回`OperationError`类型的错误，包括设备不可用、配置错误等情况
pub struct AudioPlayer {
    device: Device,
    host_id: HostId,
    stream_config: StreamConfig,
    supported_stream_config: SupportedStreamConfig,
    stream: Stream,
    write_sender: MpscSender<f32>,
    stop_sender: WatchSender<bool>,
}

impl AudioPlayer {
    fn create_stream(
        device: &Device,
        stream_config: &StreamConfig,
    ) -> Result<(MpscSender<f32>, WatchSender<bool>, Stream), OperationError> {
        let buffer_size = match stream_config.buffer_size {
            BufferSize::Default => 8192,
            BufferSize::Fixed(size) => size as _,
        };
        let (write_tx, mut write_rx) = mpsc_channel(buffer_size);
        let (stop_tx, stop_rx) = watch_channel(false);

        Ok((
            write_tx,
            stop_tx,
            device.build_output_stream(
                stream_config.clone(),
                move |buffer: &mut [f32], _| {
                    let buf = buffer.iter_mut();
                    if *stop_rx.borrow() {
                        buf.for_each(|i| *i = Default::default());
                        while let Ok(_) = write_rx.try_recv() {}
                    } else {
                        buf.for_each(|i| *i = write_rx.try_recv().unwrap_or_default());
                    }
                },
                |e| eprintln!("{}", e),
                None,
            )?,
        ))
    }

    fn update_stream(&mut self) -> Result<(), OperationError> {
        let (write_sender, stop_sender, stream) =
            Self::create_stream(&self.device, &self.stream_config)?;
        drop(replace(&mut self.stream, stream));
        drop(replace(&mut self.write_sender, write_sender));
        drop(replace(&mut self.stop_sender, stop_sender));

        Ok(())
    }

    /// 创建一个新的音频播放器实例
    ///
    /// # 返回值
    /// 返回`Result<Self, OperationError>`，成功时包含初始化的音频播放器实例
    ///
    /// # 错误
    /// 可能返回以下错误：
    /// - `OperationError::NoDevice`: 当没有默认音频输出设备时
    /// - `OperationError::StreamError`: 当创建音频流失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioPlayer;
    /// fn main() -> anyhow::Result<()> {
    /// let player = AudioPlayer::new()?;
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn new() -> Result<Self, OperationError> {
        let host = default_host();
        let device = host
            .default_output_device()
            .ok_or(OperationError::NoDevice(
                "No default audio output device.".to_owned(),
            ))?;
        let host_id = host.id();
        let supported_stream_config = device.default_output_config()?;
        let stream_config = supported_stream_config.config();
        let (write_sender, stop_sender, stream) = Self::create_stream(&device, &stream_config)?;

        Ok(Self {
            device,
            host_id,
            stream_config,
            supported_stream_config,
            stream,
            write_sender,
            stop_sender,
        })
    }

    /// 获取音频输出设备名称
    ///
    /// # 返回值
    /// 返回`Result<String, OperationError>`，成功时包含设备名称字符串
    ///
    /// # 错误
    /// 可能返回`OperationError::DeviceError`，当获取设备名称失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioPlayer;
    /// fn main() -> anyhow::Result<()> {
    /// let player = AudioPlayer::new()?;
    /// let name = player.get_name()?;
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn get_name(&self) -> Result<String, OperationError> {
        Ok(self.device.description()?.name().to_owned())
    }

    /// 获取支持的音频流通道数
    ///
    /// # 返回值
    /// 返回支持的音频通道数量(usize)
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioPlayer;
    /// fn main() -> anyhow::Result<()> {
    /// let player = AudioPlayer::new()?;
    /// let channels = player.get_supported_stream_channels();
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn get_supported_stream_channels(&self) -> usize {
        self.supported_stream_config.channels() as _
    }

    /// 获取支持的音频流采样率
    ///
    /// # 返回值
    /// 返回支持的音频采样率(usize)
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioPlayer;
    /// fn main() -> anyhow::Result<()> {
    /// let player = AudioPlayer::new()?;
    /// let sample_rate = player.get_supported_stream_sample_rate();
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn get_supported_stream_sample_rate(&self) -> usize {
        self.supported_stream_config.sample_rate() as _
    }

    /// 设置音频流通道数
    ///
    /// # 参数
    /// - `channels`: 要设置的通道数(usize)
    ///
    /// # 返回值
    /// 返回`Result<(), OperationError>`，成功时表示设置完成
    ///
    /// # 错误
    /// 可能返回`OperationError::StreamError`，当更新音频流失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioPlayer;
    /// fn main() -> anyhow::Result<()> {
    /// let mut player = AudioPlayer::new()?;
    /// player.set_stream_channels(2)?;
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn set_stream_channels(&mut self, channels: usize) -> Result<(), OperationError> {
        self.stream_config.channels = channels as _;
        self.update_stream()
    }

    /// 设置音频流采样率
    ///
    /// # 参数
    /// - `sample_rate`: 要设置的采样率(usize)
    ///
    /// # 返回值
    /// 返回`Result<(), OperationError>`，成功时表示设置完成
    ///
    /// # 错误
    /// 可能返回`OperationError::StreamError`，当更新音频流失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioPlayer;
    /// fn main() -> anyhow::Result<()> {
    /// let mut player = AudioPlayer::new()?;
    /// if let Ok(_) = player.set_stream_sample_rate(44100) {
    /// println!("Set sample rate successfully.");
    /// }
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn set_stream_sample_rate(&mut self, sample_rate: usize) -> Result<(), OperationError> {
        self.stream_config.sample_rate = sample_rate as _;
        self.update_stream()
    }

    /// 获取当前音频流的通道数
    ///
    /// # 返回值
    /// 返回当前音频流的通道数量(usize)
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioPlayer;
    /// fn main() -> anyhow::Result<()> {
    /// let player = AudioPlayer::new()?;
    /// let channels = player.get_stream_channels();
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn get_stream_channels(&self) -> usize {
        self.stream_config.channels as _
    }

    /// 获取当前音频流的采样率
    ///
    /// # 返回值
    /// 返回当前音频流的采样率(usize)
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioPlayer;
    /// fn main() -> anyhow::Result<()> {
    /// let player = AudioPlayer::new()?;
    /// let sample_rate = player.get_stream_sample_rate();
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn get_stream_sample_rate(&self) -> usize {
        self.stream_config.sample_rate as _
    }

    /// 开始播放或恢复暂停的音频
    ///
    /// # 返回值
    /// 返回`Result<(), OperationError>`，成功时表示播放已开始
    ///
    /// # 错误
    /// 可能返回`OperationError::StreamError`，当启动播放失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioPlayer;
    /// fn main() -> anyhow::Result<()> {
    /// let player = AudioPlayer::new()?;
    /// player.play()?;
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn play(&self) -> Result<(), OperationError> {
        self.stop_sender.send(false)?;

        Ok(self.stream.play()?)
    }

    /// 暂停播放音频(可使用`play()`恢复）
    ///
    /// # 返回值
    /// 返回`Result<(), OperationError>`，成功时表示播放已暂停
    ///
    /// # 错误
    /// 可能返回`OperationError::StreamError`，当暂停播放失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioPlayer;
    /// fn main() -> anyhow::Result<()> {
    /// let player = AudioPlayer::new()?;
    /// player.pause()?;
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn pause(&self) -> Result<(), OperationError> {
        Ok(self.stream.pause()?)
    }

    pub fn stop(&self) -> Result<(), OperationError> {
        self.stop_sender.send(true)?;
        self.stream.pause()?;

        Ok(())
    }

    /// 写入音频样本数据
    /// 当流通道已满，此方法会异步等待，直到播放器消耗一部分数据
    ///
    /// # 参数
    /// - `samples`: 音频样本数据切片，类型必须实现 [`GenericSample`] 特征
    /// - `channels`: 音频通道数
    ///
    /// # 返回值
    /// 返回`Result<(), OperationError>`，成功时表示数据已写入
    ///
    /// # 错误
    /// 可能返回以下错误：
    /// - `OperationError::StreamError`: 当写入数据失败时
    /// - `OperationError::ChannelError`: 当通道发送失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioPlayer;
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let mut player = AudioPlayer::new()?;
    ///     player.play()?;
    ///     player.write::<44100, f32>(&[0f32; 88200], 2).await?;
    ///     Ok(())
    /// }
    /// ```
    pub async fn write<const SR: usize, S>(
        &mut self,
        samples: &[S],
        channels: usize,
    ) -> Result<(), OperationError>
    where
        S: GenericSample,
    {
        if self.get_stream_channels() != channels || self.get_stream_sample_rate() != SR {
            let f32_samples = samples_to_f32(samples);
            let mut iter = UniformSourceIterator::new(
                SamplesBuffer::new(
                    ChannelCount::new(channels as _)
                        .ok_or(IoError::other("invalid channel count"))?,
                    SampleRate::new(SR as _).ok_or(IoError::other("invalid sample rate"))?,
                    f32_samples,
                ),
                ChannelCount::new(self.stream_config.channels)
                    .ok_or(IoError::other("invalid channel count"))?,
                SampleRate::new(self.stream_config.sample_rate)
                    .ok_or(IoError::other("invalid sample rate"))?,
            );
            while !*self.stop_sender.borrow() {
                let mut p = self.write_sender.reserve_many(256).await?;
                while let Some(w) = p.next()
                    && let Some(i) = iter.next()
                {
                    w.send(i);
                }
            }

            return Ok(());
        }

        for s in samples {
            self.write_sender.send(s.to_f32()).await?;
        }

        Ok(())
    }
}

impl Debug for AudioPlayer {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "AudioPlayer({}, {})",
            self.host_id.name(),
            self.get_name().map_err(|_| FmtError)?
        )
    }
}
