#[cfg(feature = "device")]
use cpal::{
    BuildStreamError, DefaultStreamConfigError, DeviceNameError, PauseStreamError, PlayStreamError,
};
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
    tokio::sync::mpsc::error::SendError,
};

/// 操作过程中可能出现的错误类型
#[derive(Debug)]
pub enum OperationError {
    #[cfg(feature = "device")]
    BuildStream(BuildStreamError),
    /// 音频解码错误
    Decoder(DecoderError),
    #[cfg(feature = "device")]
    DefaultStreamConfig(DefaultStreamConfigError),
    #[cfg(feature = "device")]
    DeviceName(DeviceNameError),
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
    Ort(OrtError),
    #[cfg(feature = "device")]
    PauseStream(PauseStreamError),
    #[cfg(feature = "device")]
    PlayStream(PlayStreamError),
    Send(SendError<f32>),
    /// 数组形状错误
    Shape(ShapeError),
    /// 系统时间错误
    SystemTime(SystemTimeError),
}

impl Clone for OperationError {
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "device")]
            Self::BuildStream(e) => Self::BuildStream(match e {
                BuildStreamError::DeviceNotAvailable => BuildStreamError::DeviceNotAvailable,
                BuildStreamError::StreamConfigNotSupported => {
                    BuildStreamError::StreamConfigNotSupported
                }
                BuildStreamError::InvalidArgument => BuildStreamError::InvalidArgument,
                BuildStreamError::StreamIdOverflow => BuildStreamError::StreamIdOverflow,
                BuildStreamError::BackendSpecific { err } => BuildStreamError::BackendSpecific {
                    err: err.to_owned(),
                },
            }),
            Self::Decoder(e) => Self::Decoder(e.to_owned()),
            #[cfg(feature = "device")]
            Self::DefaultStreamConfig(e) => Self::DefaultStreamConfig(match e {
                DefaultStreamConfigError::DeviceNotAvailable => {
                    DefaultStreamConfigError::DeviceNotAvailable
                }
                DefaultStreamConfigError::StreamTypeNotSupported => {
                    DefaultStreamConfigError::StreamTypeNotSupported
                }
                DefaultStreamConfigError::BackendSpecific { err } => {
                    DefaultStreamConfigError::BackendSpecific {
                        err: err.to_owned(),
                    }
                }
            }),
            #[cfg(feature = "device")]
            Self::DeviceName(e) => Self::DeviceName(e.to_owned()),
            Self::InputInvalid(s) => Self::InputInvalid(s.to_owned()),
            Self::InputTooShort => Self::InputTooShort,
            Self::Io(e) => Self::Io(IoError::new(e.kind(), e.to_string())),
            #[cfg(feature = "knf")]
            Self::Nul(e) => Self::Nul(e.to_owned()),
            Self::NoDevice(s) => Self::NoDevice(s.to_owned()),
            Self::Opus(s) => Self::Opus(s.to_owned()),
            #[cfg(feature = "model")]
            Self::Ort(e) => Self::Ort(OrtError::new(e.message())),
            #[cfg(feature = "device")]
            Self::PauseStream(e) => Self::PauseStream(match e {
                PauseStreamError::DeviceNotAvailable => PauseStreamError::DeviceNotAvailable,
                PauseStreamError::BackendSpecific { err } => PauseStreamError::BackendSpecific {
                    err: err.to_owned(),
                },
            }),
            #[cfg(feature = "device")]
            Self::PlayStream(e) => Self::PlayStream(match e {
                PlayStreamError::DeviceNotAvailable => PlayStreamError::DeviceNotAvailable,
                PlayStreamError::BackendSpecific { err } => PlayStreamError::BackendSpecific {
                    err: err.to_owned(),
                },
            }),
            Self::Send(e) => Self::Send(*e),
            Self::Shape(e) => Self::Shape(e.clone()),
            Self::SystemTime(e) => Self::SystemTime(e.clone()),
        }
    }
}

impl Display for OperationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "OperationError: ")?;
        match self {
            #[cfg(feature = "device")]
            Self::BuildStream(e) => Display::fmt(e, f),
            Self::Decoder(e) => Display::fmt(e, f),
            #[cfg(feature = "device")]
            Self::DefaultStreamConfig(e) => Display::fmt(e, f),
            #[cfg(feature = "device")]
            Self::DeviceName(e) => Display::fmt(e, f),
            Self::InputInvalid(s) => write!(f, "InputInvalid: {}", s,),
            Self::InputTooShort => write!(f, "InputTooShort: Input audio chunk is too short"),
            Self::Io(e) => Display::fmt(e, f),
            #[cfg(feature = "knf")]
            Self::Nul(e) => Display::fmt(e, f),
            Self::NoDevice(s) => write!(f, "NoDeviceError: {}", s,),
            Self::Opus(s) => write!(f, "OpusError: {}", s,),
            #[cfg(feature = "model")]
            Self::Ort(e) => Display::fmt(e, f),
            #[cfg(feature = "device")]
            Self::PauseStream(e) => Display::fmt(e, f),
            #[cfg(feature = "device")]
            Self::PlayStream(e) => Display::fmt(e, f),
            Self::Send(e) => Display::fmt(e, f),
            Self::Shape(e) => Display::fmt(e, f),
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
impl From<OrtError> for OperationError {
    fn from(value: OrtError) -> Self {
        Self::Ort(value)
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

#[cfg(feature = "device")]
impl From<DeviceNameError> for OperationError {
    fn from(value: DeviceNameError) -> Self {
        Self::DeviceName(value)
    }
}

#[cfg(feature = "device")]
impl From<DefaultStreamConfigError> for OperationError {
    fn from(value: DefaultStreamConfigError) -> Self {
        Self::DefaultStreamConfig(value)
    }
}

#[cfg(feature = "device")]
impl From<BuildStreamError> for OperationError {
    fn from(value: BuildStreamError) -> Self {
        Self::BuildStream(value)
    }
}

#[cfg(feature = "device")]
impl From<PlayStreamError> for OperationError {
    fn from(value: PlayStreamError) -> Self {
        Self::PlayStream(value)
    }
}

#[cfg(feature = "device")]
impl From<PauseStreamError> for OperationError {
    fn from(value: PauseStreamError) -> Self {
        Self::PauseStream(value)
    }
}

impl From<SendError<f32>> for OperationError {
    fn from(value: SendError<f32>) -> Self {
        Self::Send(value)
    }
}

#[cfg(feature = "knf")]
impl From<NulError> for OperationError {
    fn from(value: NulError) -> Self {
        Self::Nul(value)
    }
}
