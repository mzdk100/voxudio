use futures_util::StreamExt;
use std::path::Path;
use voxudio::*;

#[tokio::test]
async fn test_asr() {
    // 测试 1920ms 模型
    let model_dir = Path::new(r"F:\python\X-ASR-zh-en\deployment\models\chunk-1920ms-model");

    // 跳过测试如果没有模型文件
    if !model_dir.exists() {
        eprintln!("Skipping test: model directory not found");
        return;
    }

    let mut asr = AutomaticSpeechRecognizer::with_config(
        model_dir.join("encoder-1920ms.onnx"),
        model_dir.join("decoder-1920ms.onnx"),
        model_dir.join("joiner-1920ms.onnx"),
    )
    .expect("Failed to create ASR recognizer");

    // 测试空特征
    let features = vec![0.0; AutomaticSpeechRecognizer::NUM_BINS as usize * 10];
    let mut stream = asr.recognize(&features);

    // 空特征应该不会产生任何 token
    let mut count = 0;
    while let Some(result) = stream.next().await {
        if result.is_ok() {
            count += 1;
        }
    }

    assert_eq!(count, 0, "Empty features should not produce tokens");
}
