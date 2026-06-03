//! 自动语音识别(ASR)示例程序
//!
//! 本示例展示了如何使用`voxudio`库进行语音识别，包括：
//! 1. 过滤音频文件中的非语音段（Legacy ASR）
//! 2. 流式 ASR 识别（X-ASR-zh-en Zipformer2 transducer）
//!
//! 主要功能：
//! - 加载ONNX格式的ASR模型
//! - 处理16kHz采样率的音频文件
//! - 输出识别到的文字

use {
    futures_util::StreamExt,
    std::{
        io::{Write, stdout},
        path::Path,
        time::SystemTime,
    },
    voxudio::*,
};

//noinspection MissingFeatures
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ---- Legacy ASR ----
    let mut vad = VoiceActivityDetector::builder()
        .with_min_silence_duration(300)
        .build("../checkpoint/voice_activity_detector.onnx")?;
    let mut offe = OnlineFbankFeatureExtractor::fbank()?
        .with_frame_opts(FrameExtractionOptions {
            dither: 0.,
            window_type: "hamming",
            ..Default::default()
        })?
        .with_mel_opts(MelBanksOptions {
            num_bins: AutomaticSpeechRecognizerLegacy::NUM_BINS,
            ..Default::default()
        })?
        .build()?;
    let mut asr_legacy =
        AutomaticSpeechRecognizer::new_legacy("../checkpoint/automatic_speech_recognizer.onnx")?;

    let (audio, channels) = load_audio::<16000, f32, _>(r"../asset/test.wav", true).await?;
    let speech_only = vad.retain_speech_only::<16000>(&audio, channels).await?;
    let mut features = offe.extract::<16000>(&speech_only);
    let mut took = SystemTime::now();
    let text = asr_legacy.recognize(&features).await?;
    println!("[Legacy] {} ({:?})", text, took.elapsed()?);

    // ---- Streaming ASR (X-ASR-zh-en Zipformer2) ----
    // FBank 配置必须与模型训练时一致（匹配 sherpa-onnx / Python kaldi-native-fbank）
    offe = OnlineFbankFeatureExtractor::fbank()?
        .with_frame_opts(FrameExtractionOptions {
            samp_freq: 16000f32,
            frame_shift_ms: 10.0,
            frame_length_ms: 25.0,
            dither: 0.00003, // 使用默认值，与模型训练对齐
            preemph_coeff: 0.97,
            remove_dc_offset: true,
            window_type: "povey",
            snip_edges: false,
            ..Default::default()
        })?
        .with_mel_opts(MelBanksOptions {
            num_bins: 80,
            low_freq: 20.0,
            high_freq: -400.0,
            vtln_low: 100.0,
            vtln_high: -500.0,
            ..Default::default()
        })?
        .build()?;
    features = offe.extract::<16000>(&speech_only);

    // 支持不同Chunk的模型：160ms, 480ms, 960ms, 1920ms
    // git lfs install
    // git clone https://www.modelscope.ai/Gilgamesh-J/X-ASR-zh-en.git
    // 将`deployment/models`复制到`../checkpoint`中
    let model_configs = vec![
        ("160ms", "chunk-160ms-model"),
        ("480ms", "chunk-480ms-model"),
        ("960ms", "chunk-960ms-model"),
        ("1920ms", "chunk-1920ms-model"),
    ];

    for (latency, model_dir_name) in model_configs {
        let model_dir = Path::new(r"../checkpoint/x-asr-models").join(model_dir_name);

        println!("\n--- Testing {}-{} model ---", latency, model_dir_name);

        let mut asr = AutomaticSpeechRecognizer::with_config(
            model_dir.join(format!("encoder-{}.onnx", latency)),
            model_dir.join(format!("decoder-{}.onnx", latency)),
            model_dir.join(format!("joiner-{}.onnx", latency)),
        )?;

        took = SystemTime::now();
        let mut stream = asr.recognize(&features);
        let mut first_packet_at = None;
        let mut output = String::new();

        while let Some(token) = stream.next().await {
            let token_text = token?;
            if first_packet_at.is_none() {
                first_packet_at = Some(took.elapsed()?);
            }
            output.push_str(&token_text);
            print!("{}", token_text);
            stdout().flush()?;
        }

        println!(
            "\n[{}] {} (first_token: {:?}, total: {:?})",
            latency,
            output,
            first_packet_at,
            took.elapsed()?
        );
    }

    Ok(())
}
