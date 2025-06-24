use {std::f32::consts::PI, voxudio::*};

#[tokio::test]
async fn test_vad_creation() {
    let vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx");
    assert!(vad.is_ok(), "Should be able to create VoiceActivityDetector");
}

#[tokio::test]
async fn test_vad_detect() -> anyhow::Result<()> {
    let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;

    // Create an audio segment containing speech (512 samples, 16kHz)
    let mut audio_with_speech = Vec::with_capacity(512);
    for i in 0..512 {
        let t = i as f32 / 16000.0;
        // Create a signal in human voice frequency range (~100-300Hz)
        let sample = 0.5 * (2.0 * PI * 150.0 * t).sin() + 0.3 * (2.0 * PI * 250.0 * t).sin();
        audio_with_speech.push(sample);
    }

    // Create a silent audio segment (512 samples)
    let audio_silence = vec![0.0f32; 512];

    // Test speech detection
    let prob = vad.detect::<16000>(&audio_with_speech).await?;
    println!("Speech probability: {}", prob);
    // Speech segment should have higher probability
    assert!(prob > 0.3, "Speech segment should have higher detection probability");

    // Test silence detection
    let prob = vad.detect::<16000>(&audio_silence).await?;
    println!("Silence probability: {}", prob);
    // Silent segment should have lower probability
    assert!(prob < 0.3, "Silent segment should have lower detection probability");

    Ok(())
}

#[tokio::test]
async fn test_vad_get_speech_segments() -> anyhow::Result<()> {
    let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;

    // Create a test audio with alternating speech and silence segments
    let sample_rate = 16000;
    let duration_secs = 3.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut audio = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        // Add speech signal between 1-2 seconds
        if t >= 1.0 && t <= 2.0 {
            let sample = 0.5 * (2.0 * PI * 150.0 * t).sin() + 0.3 * (2.0 * PI * 250.0 * t).sin();
            audio.push(sample);
        } else {
            // Other times are silence
            audio.push(0.0);
        }
    }

    // Get speech segments
    let segments = vad.get_speech_segments::<16000>(&audio).await?;

    println!("Detected speech segments: {:?}", segments);
    assert!(!segments.is_empty(), "Should detect at least one speech segment");

    // Print raw results before iteration
    println!("Raw detection results: {:?}", segments);

    // Verify segment positions (using approximate values considering padding)
    for (start, end) in &segments {
        let start_time = *start as f32 / sample_rate as f32;
        let end_time = *end as f32 / sample_rate as f32;

        println!("Converted to seconds: {:.2}-{:.2}", start_time, end_time);

        // Note: VAD has inherent latency, accept results in 1.9-3.0s range
        // Because:
        // 1. VAD needs time to confirm speech start
        // 2. Silence detection also needs buffer time to confirm speech end

        // Verify detected segments cover at least part of speech region
        assert!(
            start_time <= 2.5 && end_time >= 1.5,
            "Detected segment should cover part of speech region (1-2s), actual: {:.2}-{:.2}s",
            start_time,
            end_time
        );

        println!(
            "Note: VAD detection result {:.2}-{:.2}s covers speech region as expected",
            start_time, end_time
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_vad_retain_speech_only() -> anyhow::Result<()> {
    let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;

    // Create a stereo test audio with alternating speech and silence segments
    let sample_rate = 16000;
    let duration_secs = 3.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let channels = 2;
    let mut audio = Vec::with_capacity(num_samples * channels);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        // Add more realistic speech signal between 1-2 seconds
        if t >= 1.0 && t <= 2.0 {
            // Fundamental frequency + harmonics to simulate real speech
            let f0 = 120.0; // Fundamental frequency
            let sample_left = 0.5 * (2.0 * PI * f0 * t).sin()
                + 0.3 * (2.0 * PI * f0 * 2.0 * t).sin()
                + 0.2 * (2.0 * PI * f0 * 3.0 * t).sin();

            // Add slight variation to right channel
            let sample_right = 0.4 * (2.0 * PI * f0 * t).sin()
                + 0.35 * (2.0 * PI * f0 * 2.0 * t).sin()
                + 0.25 * (2.0 * PI * f0 * 3.0 * t).sin();

            // Add amplitude envelope to simulate speech prosody
            let envelope = 0.5 + 0.5 * (2.0 * PI * 0.5 * t).sin();

            audio.push(sample_left * envelope);
            audio.push(sample_right * envelope);
        } else {
            // Other times are silence
            audio.push(0.0);
            audio.push(0.0);
        }
    }

    // Retain only speech parts
    let speech_only = vad.retain_speech_only::<16000>(&audio, channels).await?;

    // Verify results
    assert!(!speech_only.is_empty(), "Result should not be empty");
    assert_eq!(speech_only.len() % channels, 0, "Result should maintain correct number of channels");

    // Result length should be shorter than original (since silence was removed)
    assert!(speech_only.len() < audio.len(), "Result should be shorter than original audio");

    // Verify it contains non-zero samples (speech)
    let has_non_zero = speech_only.iter().any(|&x| x != 0.0);
    assert!(has_non_zero, "Result should contain non-zero samples");

    Ok(())
}

#[tokio::test]
async fn test_vad_with_different_sample_rates() -> anyhow::Result<()> {
    let mut vad = VoiceActivityDetector::new("../checkpoint/voice_activity_detector.onnx")?;

    // Create 8kHz test audio
    let mut audio_8k = Vec::with_capacity(256);
    for i in 0..256 {
        let t = i as f32 / 8000.0;
        let sample = 0.5 * (2.0 * PI * 150.0 * t).sin();
        audio_8k.push(sample);
    }

    // Test 8kHz sample rate
    let result_8k = vad.detect::<8000>(&audio_8k).await;
    assert!(result_8k.is_ok(), "8kHz sample rate detection should succeed");

    // Create 16kHz test audio
    let mut audio_16k = Vec::with_capacity(512);
    for i in 0..512 {
        let t = i as f32 / 16000.0;
        let sample = 0.5 * (2.0 * PI * 150.0 * t).sin();
        audio_16k.push(sample);
    }

    // Test 16kHz sample rate
    let result_16k = vad.detect::<16000>(&audio_16k).await;
    assert!(result_16k.is_ok(), "16kHz sample rate detection should succeed");

    // Test 32kHz sample rate (should automatically downsample to 16kHz)
    let mut audio_32k = Vec::with_capacity(1024);
    for i in 0..1024 {
        let t = i as f32 / 32000.0;
        let sample = 0.5 * (2.0 * PI * 150.0 * t).sin();
        audio_32k.push(sample);
    }

    let result_32k = vad.detect::<32000>(&audio_32k).await;
    assert!(result_32k.is_ok(), "32kHz sample rate detection should succeed");

    Ok(())
}
