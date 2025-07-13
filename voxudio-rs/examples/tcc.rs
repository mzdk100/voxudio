//! 语音转换示例程序
//!
//! 该程序演示了如何使用voxudio库进行语音活动检测、说话人嵌入提取和音色转换
//! 主要流程:
//! 1. 加载源音频和目标音频
//! 2. 使用VAD保留语音部分
//! 3. 提取说话人嵌入特征
//! 4. 进行音色转换
//! 5. 播放转换后的音频

use voxudio::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;
    let mut see = SpeakerEmbeddingExtractor::new("../checkpoint/speaker_embedding_extractor.onnx")?;
    let mut tcc = ToneColorConverter::new("../checkpoint/tone_color_converter.onnx")?;

    let (src_audio, src_channels) = load_audio::<22050, _>("../asset/test6.wav", false).await?;
    let vad_audio = vad
        .retain_speech_only::<22050>(&src_audio, src_channels)
        .await?;
    let src_se = see.extract(&vad_audio, src_channels).await?;

    let (tgt_audio, tgt_channels) = load_audio::<22050, _>("../asset/bajie.mp3", false).await?;
    let vad_audio = vad
        .retain_speech_only::<22050>(&tgt_audio, tgt_channels)
        .await?;
    let tgt_se = see.extract(&vad_audio, tgt_channels).await?;

    let (out_audio, out_channels, cost) = tcc.convert(&src_audio, &src_se, &tgt_se).await?;
    println!("Convert cost: {:?}", cost);

    let mut ap = AudioPlayer::new()?;
    ap.play()?;
    ap.write::<22050>(&out_audio, out_channels).await?;

    Ok(())
}