//! 音色嵌入提取示例程序
//!
//! 该程序演示了如何使用voxudio库进行说话人嵌入提取
//! 主要功能包括：
//! 1. 加载音频文件
//! 2. 使用VAD模型检测语音活动区域
//! 3. 提取说话人嵌入特征
//!
//! 依赖模型文件：
//! - voice_activity_detector.onnx
//! - speaker_embedding_extractor.onnx

use voxudio::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;
    let mut see = SpeakerEmbeddingExtractor::new("../checkpoint/speaker_embedding_extractor.onnx")?;

    let (audio, channels) = load_audio::<22050, _>("../asset/houge.mp3", false).await?;
    let vad_audio = vad.retain_speech_only::<22050>(&audio, channels).await?;
    let se = see.extract(&vad_audio, channels).await?;
    println!("{:?}", se);

    Ok(())
}
