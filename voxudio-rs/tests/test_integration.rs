use {
    std::{f32::consts::PI, time::Duration},
    tokio::time::sleep,
    voxudio::*,
};

// This test will test the complete audio processing pipeline:
// 1. Create a simulated audio signal
// 2. Detect speech activity with VAD
// 3. Extract speaker features
// 4. Perform voice conversion with another speaker's features
#[tokio::test]
async fn test_complete_audio_pipeline() -> anyhow::Result<()> {
    // Create simulated audio signal
    let sample_rate = 22050;
    let duration_secs = 3.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;

    // Generate source audio signal
    let source_audio = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let f0 = 120.0; // Fundamental frequency
            let sample = 0.5 * (2.0 * PI * f0 * t).sin()
                + 0.3 * (2.0 * PI * f0 * 2.0 * t).sin()
                + 0.2 * (2.0 * PI * f0 * 3.0 * t).sin();
            let envelope = 0.5 + 0.5 * (2.0 * PI * 0.5 * t).sin();
            sample * envelope * 0.5
        })
        .collect::<Vec<_>>();

    // Generate target audio signal
    let target_audio = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let f0 = 220.0; // Higher fundamental frequency
            let sample = 0.5 * (2.0 * PI * f0 * t).sin()
                + 0.3 * (2.0 * PI * f0 * 2.0 * t).sin()
                + 0.2 * (2.0 * PI * f0 * 3.0 * t).sin();
            let envelope = 0.5 + 0.5 * (2.0 * PI * 0.5 * t).sin();
            sample * envelope * 0.5
        })
        .collect::<Vec<_>>();

    // Step 1: Detect speech activity with VAD
    println!("Step 1: Running voice activity detection");
    let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;
    let source_audio_16k = resample::<22050, 16000>(&source_audio, 1, 1);
    let is_speech = vad.detect::<16000>(&source_audio_16k[..512]).await?;
    println!("VAD detection result: is_speech = {}", is_speech);

    // Step 2: Extract source speaker features
    println!("Step 2: Extracting speaker embeddings from source");
    let mut see = SpeakerEmbeddingExtractor::new("../checkpoint/speaker_embedding_extractor.onnx")?;
    let source_features = see.extract(&source_audio, 1).await?;
    println!("Source feature dimensions: {}", source_features.len());

    // Step 3: Extract target speaker features
    println!("Step 3: Extracting speaker embeddings from target");
    let target_features = see.extract(&target_audio, 1).await?;
    println!("Target feature dimensions: {}", target_features.len());

    // Step 4: Perform voice conversion with target features
    println!("Step 4: Running voice conversion model with target embeddings");
    let mut tcc = ToneColorConverter::new("../checkpoint/tone_color_converter.onnx")?;
    let converted_audio = tcc
        .convert(&source_audio, &source_features, &target_features)
        .await?;
    println!("Converted audio length: {}", converted_audio.0.len());

    println!("Integration test completed");
    Ok(())
}

// This test will test the real-time audio processing pipeline:
// 1. Start audio collector
// 2. Read audio segment
// 3. Detect speech activity with VAD
// 4. If speech detected, extract speaker features
// 5. Stop audio collector
#[tokio::test]
async fn test_realtime_audio_processing() -> anyhow::Result<()> {
    // Create audio collector
    let Ok(mut collector) = AudioCollector::new() else {
        println!("Failed to create AudioCollector: skipping test");
        return Ok(());
    };

    // Create VAD
    let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;

    // Create SEE
    let mut see = SpeakerEmbeddingExtractor::new("../checkpoint/speaker_embedding_extractor.onnx")?;

    // Start audio collector
    println!("Starting audio collector...");
    collector.collect()?;

    // Wait briefly to allow collector to gather some data
    println!("Waiting for audio data collection...");
    sleep(Duration::from_millis(500)).await;

    // Read audio data
    let audio_data = collector.read::<44100>(2).await?;
    println!("Read {} audio samples", audio_data.len());

    // Stop audio collector
    println!("Stopping audio collector...");
    collector.pause()?;

    // Detect speech activity with VAD
    println!("Detecting speech activity with VAD...");
    let is_speech = vad.detect::<48000>(&audio_data).await?;
    println!("VAD detection result: is_speech = {}", is_speech);

    // If speech detected, extract speaker features
    if is_speech > 0.5 {
        println!("Speech detected, extracting speaker features...");
        let features = see.extract(&audio_data, 1).await?;
        println!(
            "Successfully extracted features, dimensions: {}",
            features.len()
        );
    } else {
        println!("No speech detected, skipping feature extraction");
    }

    Ok(())
}
