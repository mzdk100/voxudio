use voxudio::*;

#[tokio::test]
async fn test_asr() {
    tokio::spawn(async move {
        let mut asr =
            AutomaticSpeechRecognizer::new("../checkpoint/automatic_speech_recognizer.onnx")?;
        let features = vec![0.0; AutomaticSpeechRecognizer::NUM_BINS as usize * 10];
        let result = asr.recognize(&features).await?;
        assert!(result.is_empty());

        Ok::<_, anyhow::Error>(())
    });
}
