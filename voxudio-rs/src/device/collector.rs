use {
    crate::OperationError,
    cpal::{
        BufferSize, Device, HostId, Stream, StreamConfig, SupportedStreamConfig, default_host,
        traits::{DeviceTrait, HostTrait, StreamTrait},
    },
    rodio::{buffer::SamplesBuffer, source::UniformSourceIterator},
    std::{
        fmt::{Debug, Error as FmtError, Formatter, Result as FmtResult},
        io::{Error as IoError, ErrorKind},
        mem::replace,
    },
    tokio::{
        sync::mpsc::{Receiver, channel},
        time::{Duration, sleep},
    },
};

/// 音频采集器模块
///
/// 使用音频输入设备提供音频的采集功能，支持以下操作：
/// - 创建音频采集器实例
/// - 获取/设置音频流参数（通道数、采样率）
/// - 开始/暂停音频采集
/// - 读取音频数据
/// - 关闭采集器
///
/// # 示例
/// ```
/// use voxudio::AudioCollector;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
/// let Ok(mut collector) = AudioCollector::new() else {
/// return Ok(());
/// };
/// collector.collect()?;
/// let data = collector.read::<44100>(2).await?;
/// collector.close();
///
/// Ok(())
/// }
/// ```
pub struct AudioCollector {
    device: Device,
    host_id: HostId,
    receiver: Receiver<f32>,
    stream_config: StreamConfig,
    supported_stream_config: SupportedStreamConfig,
    stream: Stream,
}

impl AudioCollector {
    fn create_stream(
        device: &Device,
        stream_config: &StreamConfig,
    ) -> Result<(Receiver<f32>, Stream), OperationError> {
        let buffer_size = match stream_config.buffer_size {
            BufferSize::Default => 8192,
            BufferSize::Fixed(size) => size as _,
        };
        let (tx, rx) = channel(buffer_size);

        Ok((
            rx,
            device.build_input_stream(
                stream_config,
                move |buffer: &[f32], _| {
                    let iter = match tx.try_reserve_many(buffer.len()) {
                        Err(e) => {
                            eprintln!("AudioCollector can't send data: {}", e);
                            return;
                        }
                        Ok(p) => p,
                    };
                    let mut iter = iter.enumerate();
                    while let Some((i, permit)) = iter.next() {
                        permit.send(buffer[i]);
                    }
                },
                |e| eprintln!("{}", e),
                None,
            )?,
        ))
    }

    fn update_stream(&mut self) -> Result<(), OperationError> {
        let (receiver, stream) = Self::create_stream(&self.device, &self.stream_config)?;
        drop(replace(&mut self.stream, stream));
        drop(replace(&mut self.receiver, receiver));

        Ok(())
    }

    /// 创建新的音频采集器实例
    ///
    /// # 返回值
    /// 返回`Result<Self, OperationError>`，成功时包含初始化的音频采集器
    ///
    /// # 错误
    /// 可能返回以下错误：
    /// - `OperationError::NoDevice`: 当没有默认音频输入设备时
    /// - `OperationError::StreamError`: 当创建音频流失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioCollector;
    ///
    /// if let Ok(collector) = AudioCollector::new() {
    /// println!("{:?}", collector);
    /// }
    /// ```
    pub fn new() -> Result<Self, OperationError> {
        let host = default_host();
        let host_id = host.id();
        let device = host.default_input_device().ok_or(OperationError::NoDevice(
            "No default audio input device.".to_owned(),
        ))?;
        let supported_stream_config = device.default_input_config()?;
        let stream_config = supported_stream_config.config();
        let (receiver, stream) = Self::create_stream(&device, &stream_config)?;

        Ok(Self {
            device,
            host_id,
            receiver,
            stream_config,
            supported_stream_config,
            stream,
        })
    }

    /// 获取音频输入设备名称
    ///
    /// # 返回值
    /// 返回`Result<String, OperationError>`，成功时包含设备名称字符串
    ///
    /// # 错误
    /// 可能返回`OperationError::DeviceError`，当获取设备名称失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioCollector;
    /// fn main() -> anyhow::Result<()> {
    /// let Ok(collector) = AudioCollector::new() else {
    /// return Ok(());
    /// };
    /// let name = collector.get_name()?;
    /// Ok(())
    /// }
    /// ```
    pub fn get_name(&self) -> Result<String, OperationError> {
        Ok(self.device.name()?)
    }

    /// 获取支持的音频流通道数
    ///
    /// # 返回值
    /// 返回支持的音频通道数量(usize)
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioCollector;
    /// if let Ok(collector) = AudioCollector::new() {
    /// let channels = collector.get_supported_stream_channels();
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
    /// use voxudio::AudioCollector;
    /// fn main() -> anyhow::Result<()> {
    /// let Ok(collector) = AudioCollector::new() else {
    /// return Ok(());
    /// };
    /// let sample_rate = collector.get_supported_stream_sample_rate();
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn get_supported_stream_sample_rate(&self) -> usize {
        self.supported_stream_config.sample_rate().0 as _
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
    /// use voxudio::AudioCollector;
    /// fn main() -> anyhow::Result<()> {
    /// let Ok(mut collector) = AudioCollector::new() else {
    /// return Ok(());
    /// };
    /// collector.set_stream_channels(2)?;
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
    /// use voxudio::AudioCollector;
    /// fn main() -> anyhow::Result<()> {
    /// let Ok(mut collector) = AudioCollector::new() else {
    /// return Ok(());
    /// };
    /// collector.set_stream_sample_rate(32000)?;
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn set_stream_sample_rate(&mut self, sample_rate: usize) -> Result<(), OperationError> {
        self.stream_config.sample_rate.0 = sample_rate as _;
        self.update_stream()
    }

    /// 获取当前音频流的通道数
    ///
    /// # 返回值
    /// 返回当前音频流的通道数量(usize)
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioCollector;
    /// if let Ok(collector) = AudioCollector::new() {
    /// let channels = collector.get_stream_channels();
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
    /// use voxudio::AudioCollector;
    /// if let Ok(collector) = AudioCollector::new() {
    /// let sample_rate = collector.get_stream_sample_rate();
    /// }
    /// ```
    pub fn get_stream_sample_rate(&self) -> usize {
        self.stream_config.sample_rate.0 as _
    }

    /// 开始采集或恢复暂停的音频
    ///
    /// # 返回值
    /// 返回`Result<(), OperationError>`，成功时表示采集已开始
    ///
    /// # 错误
    /// 可能返回`OperationError::StreamError`，当启动采集失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioCollector;
    /// fn main() -> anyhow::Result<()> {
    /// let Ok(collector) = AudioCollector::new() else {
    /// return Ok(());
    /// };
    /// collector.collect()?;
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn collect(&self) -> Result<(), OperationError> {
        Ok(self.stream.play()?)
    }

    /// 暂停采集音频(可使用`collect()`恢复）
    ///
    /// # 返回值
    /// 返回`Result<(), OperationError>`，成功时表示采集已暂停
    ///
    /// # 错误
    /// 可能返回`OperationError::StreamError`，当暂停采集失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioCollector;
    /// fn main() -> anyhow::Result<()> {
    /// let Ok(collector) = AudioCollector::new() else {
    /// return Ok(());
    /// };
    /// collector.pause()?;
    ///
    /// Ok(())
    /// }
    /// ```
    pub fn pause(&self) -> Result<(), OperationError> {
        Ok(self.stream.pause()?)
    }

    /// 从音频流中读取数据
    ///
    /// # 参数
    /// - `channels`: 目标通道数(usize)
    ///
    /// # 返回值
    /// 返回`Result<Vec<f32>, OperationError>`，成功时包含音频数据向量
    ///
    /// # 错误
    /// 可能返回`OperationError::Io`，当读取数据失败时
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioCollector;
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    /// let Ok(mut collector) = AudioCollector::new() else {
    /// return Ok(());
    /// };
    /// let data = collector.read::<44100>(2).await?;
    ///
    /// Ok(())
    /// }
    /// ```
    pub async fn read<const SR: usize>(
        &mut self,
        channels: usize,
    ) -> Result<Vec<f32>, OperationError> {
        if self.receiver.is_empty() {
            sleep(Duration::from_millis(
                (self.receiver.max_capacity() * 1000
                    / self.get_stream_channels()
                    / self.get_stream_sample_rate())
                .min(25) as _,
            ))
            .await;
        }
        let capacity = self.receiver.max_capacity() - self.receiver.capacity();
        let mut buffer = Vec::with_capacity(capacity);
        let read = self.receiver.recv_many(&mut buffer, capacity).await;
        if capacity > 0 && read == 0 {
            return Err(OperationError::Io(IoError::new(
                ErrorKind::UnexpectedEof,
                "No more data.",
            )));
        }

        let res = if self.get_stream_channels() != channels || self.get_stream_sample_rate() != SR {
            let buffer = SamplesBuffer::new(
                self.get_stream_channels() as _,
                self.get_stream_sample_rate() as _,
                &buffer[..read],
            );
            UniformSourceIterator::new(buffer, channels as _, SR as _).collect()
        } else {
            buffer
        };

        Ok(res)
    }

    /// 关闭音频采集器
    ///
    /// # 说明
    /// 关闭接收器通道，停止接收音频数据
    ///
    /// # 示例
    /// ```
    /// use voxudio::AudioCollector;
    /// if let Ok(mut collector) = AudioCollector::new() {
    /// collector.close();
    /// }
    /// ```
    pub fn close(&mut self) {
        self.receiver.close()
    }
}

impl Debug for AudioCollector {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "AudioCollector({}, {})",
            self.host_id.name(),
            self.get_name().map_err(|_| FmtError)?
        )
    }
}
