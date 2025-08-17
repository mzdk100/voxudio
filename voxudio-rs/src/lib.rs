//! # Voxudio
//!
//! `voxudio` is a real-time audio processing library with ONNX runtime support.
//! It provides a set of tools for audio device management, signal processing,
//! and machine learning model integration for audio applications.
//!
//! ## Features
//!
//! - Audio device enumeration and management
//! - Real-time audio processing capabilities
//! - ONNX model integration for audio machine learning tasks
//! - OPUS audio codec support (encoding/decoding)
//! - Online feature extraction (FBank, MFCC, Whisper FBank) based on kaldi-native-fbank
//! - Cross-platform support
//!
//! ## Example
//!
//! ### Speaker embedding extraction example
//! ```rust,no_run
//! use voxudio::*;
//! use anyhow::Result;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Initialize voice activity detector and speaker embedding extractor
//!     let mut vad = VoiceActivityDetector::new("checkpoint/voice_activity_detector.onnx")?;
//!     let mut see = SpeakerEmbeddingExtractor::new("checkpoint/speaker_embedding_extractor.onnx")?;
//!
//!     // Load audio file
//!     let (audio, channels) = load_audio::<22050, _>("../asset/test.wav", false).await?;
//!
//!     // Detect speech segments
//!     let vad_audio = vad.retain_speech_only::<22050>(&audio, channels).await?;
//!
//!     // Extract speaker embedding
//!     let embedding = see.extract(&vad_audio, channels).await?;
//!     println!("Extracted embedding: {:?}", embedding);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Online feature extraction example
//! ```rust,no_run
//! use voxudio::*;
//! use anyhow::Result;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Build an online FBank feature extractor (algorithm from kaldi-native-fbank)
//!     let extractor = OnlineFbankFeatureExtractor::fbank()?
//!         .with_energy_floor(1.0)
//!         .build()?;
//!     // Load audio file
//!     let (audio, channels) = load_audio::<16000, _>("../asset/test.wav", true).await?;
//!     // Extract FBank features
//!     let features = extractor.extract::<16000>(&audio);
//!     println!("FBank features: {:?}", features);
//!
//!     Ok(())
//! }
//! ```
//! See more from `examples`.
//!
//! ## License
//!
//! This project is licensed under the Apache License, Version 2.0.

#[cfg(feature = "device")]
mod device;
mod error;
#[cfg(feature = "knf")]
mod knf;
#[cfg(feature = "model")]
mod model;
#[cfg(feature = "opus")]
mod opus;
mod utils;

#[cfg(feature = "device")]
pub use device::*;
#[cfg(feature = "knf")]
pub use knf::*;
#[cfg(feature = "model")]
pub use model::*;
#[cfg(feature = "opus")]
pub use opus::*;
pub use {error::*, utils::*};
