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
//! - Cross-platform support
//!
//! ## Example
//!
//! ```rust,no_run
//! // Speaker embedding extraction example
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
//!     let (audio, channels) = load_audio::<22050, _>("asset/sample.wav", false).await?;
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
//! See more from `examples`.
//!
//! ## License
//!
//! This project is licensed under the Apache License, Version 2.0.

mod device;
mod error;
mod model;
mod utils;

pub use {device::*, error::*, model::*, utils::*};
