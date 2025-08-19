//! 自动语音识别(ASR)示例程序
//!
//! 本示例展示了如何使用`voxudio`库进行语音识别，包括：
//! 1. 过滤音频文件中的非语音段
//! 2. 提取FBank特征
//! 3. 识别语音对应的文字
//!
//! 主要功能：
//! - 加载ONNX格式的ASR模型
//! - 处理16kHz采样率的音频文件
//! - 输出识别到的文字

use voxudio::*;

//noinspection MissingFeatures
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut vad = VoiceActivityDetector::builder()
        .with_min_silence_duration(300)
        .build("../checkpoint/voice_activity_detector.onnx")?;
    let offe = OnlineFbankFeatureExtractor::fbank()?
        .with_frame_opts(FrameExtractionOptions {
            // 抖动常数，用于在非语音部分信号存在硬零值时
            // 防止在对数梅尔滤波器组中出现较大的负值
            //
            // 在 k2 中，音频样本范围为 [-1..+1]，而在 kaldi 中范围为 [-32k..+32k]
            // 因此值 0.00003 等价于 kaldi 默认值 1.0
            dither: 0.,
            window_type: "hamming",
            ..Default::default()
        })?
        .with_mel_opts(MelBanksOptions {
            num_bins: AutomaticSpeechRecognizer::NUM_BINS,
            ..Default::default()
        })?
        .build()?;
    let mut asr = AutomaticSpeechRecognizer::new("../checkpoint/automatic_speech_recognizer.onnx")?;

    let (audio, channels) = load_audio::<16000, _>(r"../asset/test.wav", true).await?;
    let speech_only = vad.retain_speech_only::<16000>(&audio, channels).await?;
    let features = offe.extract::<16000>(&speech_only);
    let text = asr.recognize(&features).await?;
    println!("{}", text);

    Ok(())
}
