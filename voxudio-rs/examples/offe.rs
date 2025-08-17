//! OnlineFBank提取示例程序
//!
//! 该程序演示了如何使用voxudio库提取Online FBank特征（算法来自于kaldi-native-fbank）
//! 主要功能包括：
//! 1. 加载音频文件
//! 2. 提取FBank

use voxudio::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (audio, _) = load_audio::<16000, _>(
        "F:/python/sherpa-onnx/sherpa-onnx-zipformer-ctc-small-zh-int8-2025-07-16/test_wavs/0.wav",
        true,
    )
    .await?;
    let audio = audio.iter().map(|i| i * 32768f32).collect::<Vec<_>>();

    let offe = OnlineFbankFeatureExtractor::new()?;
    let features = offe.extract::<16000>(&audio);
    println!("{:?}", features);

    Ok(())
}
