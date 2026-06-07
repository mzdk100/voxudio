//noinspection SpellCheckingInspection
#[cfg(feature = "device")]
use cpal::Error as CpalError;
#[cfg(feature = "model")]
use ort::Error as OrtError;
#[cfg(feature = "knf")]
use std::ffi::NulError;
use {
    ndarray::ShapeError,
    rodio::decoder::DecoderError,
    std::{
        error::Error,
        fmt::{Display, Formatter, Result as FmtResult},
        io::Error as IoError,
        time::SystemTimeError,
    },
    tokio::sync::{
        mpsc::error::SendError as MpscSendError, watch::error::SendError as WatchSendError,
    },
};

//noinspection SpellCheckingInspection
/// 操作过程中可能出现的错误类型
#[derive(Debug)]
pub enum OperationError {
    /// 音频设备错误
    #[cfg(feature = "device")]
    Cpal(CpalError),
    /// 音频解码错误
    Decoder(DecoderError),
    /// 无效的输入数据
    InputInvalid(String),
    /// 输入数据过短
    InputTooShort,
    /// IO操作错误
    Io(IoError),
    NoDevice(String),
    #[cfg(feature = "knf")]
    Nul(NulError),
    /// Opus编解码错误
    Opus(String),
    /// ONNX运行时错误
    #[cfg(feature = "model")]
    Ort(String),
    Send(String),
    /// 数组形状错误
    Shape(ShapeError),
    /// Sonic 变速处理错误
    #[cfg(feature = "sonic")]
    Sonic(String),
    /// 系统时间错误
    SystemTime(SystemTimeError),
}

impl Clone for OperationError {
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "device")]
            Self::Cpal(e) => Self::Cpal(e.to_owned()),
            Self::Decoder(e) => Self::Decoder(e.to_owned()),

            Self::InputInvalid(s) => Self::InputInvalid(s.to_owned()),
            Self::InputTooShort => Self::InputTooShort,
            Self::Io(e) => Self::Io(IoError::new(e.kind(), e.to_string())),
            #[cfg(feature = "knf")]
            Self::Nul(e) => Self::Nul(e.to_owned()),
            Self::NoDevice(s) => Self::NoDevice(s.to_owned()),
            Self::Opus(s) => Self::Opus(s.to_owned()),
            #[cfg(feature = "model")]
            Self::Ort(e) => Self::Ort(e.to_owned()),

            Self::Send(msg) => Self::Send(msg.clone()),
            Self::Shape(e) => Self::Shape(e.clone()),
            #[cfg(feature = "sonic")]
            Self::Sonic(s) => Self::Sonic(s.to_owned()),
            Self::SystemTime(e) => Self::SystemTime(e.clone()),
        }
    }
}

impl Display for OperationError {
    //noinspection SpellCheckingInspection
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "OperationError: ")?;
        match self {
            #[cfg(feature = "device")]
            Self::Cpal(e) => Display::fmt(e, f),
            Self::Decoder(e) => Display::fmt(e, f),

            Self::InputInvalid(s) => write!(f, "InputInvalid: {}", s,),
            Self::InputTooShort => write!(f, "InputTooShort: Input audio chunk is too short"),
            Self::Io(e) => Display::fmt(e, f),
            #[cfg(feature = "knf")]
            Self::Nul(e) => Display::fmt(e, f),
            Self::NoDevice(s) => write!(f, "NoDeviceError: {}", s,),
            Self::Opus(s) => write!(f, "OpusError: {}", s,),
            #[cfg(feature = "model")]
            Self::Ort(e) => write!(f, "OrtError: {}", e),

            Self::Send(msg) => write!(f, "SendError: {}", msg),
            Self::Shape(e) => Display::fmt(e, f),
            #[cfg(feature = "sonic")]
            Self::Sonic(s) => write!(f, "SonicError: {}", s),
            Self::SystemTime(e) => Display::fmt(e, f),
        }
    }
}

impl Error for OperationError {}

impl From<ShapeError> for OperationError {
    fn from(value: ShapeError) -> Self {
        Self::Shape(value)
    }
}

#[cfg(feature = "model")]
impl<T> From<OrtError<T>> for OperationError {
    fn from(value: OrtError<T>) -> Self {
        Self::Ort(value.to_string())
    }
}

impl From<SystemTimeError> for OperationError {
    fn from(value: SystemTimeError) -> Self {
        Self::SystemTime(value)
    }
}

impl From<IoError> for OperationError {
    fn from(value: IoError) -> Self {
        Self::Io(value)
    }
}

impl From<DecoderError> for OperationError {
    fn from(value: DecoderError) -> Self {
        Self::Decoder(value)
    }
}

impl<T> From<MpscSendError<T>> for OperationError {
    fn from(value: MpscSendError<T>) -> Self {
        Self::Send(value.to_string())
    }
}

impl<T> From<WatchSendError<T>> for OperationError {
    fn from(value: WatchSendError<T>) -> Self {
        Self::Send(value.to_string())
    }
}

#[cfg(feature = "knf")]
impl From<NulError> for OperationError {
    fn from(value: NulError) -> Self {
        Self::Nul(value)
    }
}

#[cfg(feature = "device")]
impl From<CpalError> for OperationError {
    fn from(value: CpalError) -> Self {
        Self::Cpal(value)
    }
}
