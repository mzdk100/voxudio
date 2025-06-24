//! 语音活动检测(VAD)示例程序
//!
//! 本示例展示了如何使用`voxudio`库进行语音活动检测，包括：
//! 1. 从音频文件中检测语音段
//! 2. 仅保留语音部分（过滤非语音）
//! 3. 播放处理后的音频
//!
//! 主要功能：
//! - 加载ONNX格式的VAD模型
//! - 处理16kHz采样率的音频文件
//! - 输出检测到的语音时间段
//! - 生成仅包含语音的音频并播放

use voxudio::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut vad = VoiceActivityDetector::builder()
        .with_min_silence_duration(300)
        .build("../checkpoint/voice_activity_detector.onnx")?;
    let (audio, channels) = load_audio::<16000, _>("../asset/test4.wav", true).await?;

    let segments = vad.get_speech_segments::<16000>(&audio).await?;
    println!("Speech segments:");
    for (i, (start, end)) in segments.iter().enumerate() {
        println!("{}: {}..{}", i, start, end);
    }
    let speech_only = vad.retain_speech_only::<16000>(&audio, channels).await?;

    let mut ap = AudioPlayer::new()?;
    ap.play()?;
    ap.write::<16000>(&speech_only, channels).await?;

    Ok(())
}
