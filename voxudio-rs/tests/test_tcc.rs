use {std::f32::consts::PI, voxudio::*};

#[tokio::test]
async fn test_tcc_creation() {
    let tcc = ToneColorConverter::new("../checkpoint/tone_color_converter.onnx");
    assert!(tcc.is_ok(), "Should be able to create ToneColorConverter");
}

#[tokio::test]
async fn test_tcc_process() -> anyhow::Result<()> {
    let mut tcc = ToneColorConverter::new("../checkpoint/tone_color_converter.onnx")?;

    // Create a simulated speech signal
    let sample_rate = 16000; // 16kHz
    let duration_secs = 3.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;

    let mut source_audio = Vec::with_capacity(num_samples);

    // Create a signal with multiple frequencies to simulate speech
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        // Fundamental frequency + harmonics to simulate speech characteristics
        let f0 = 120.0; // Fundamental frequency, similar to human voice
        let sample = 0.5 * (2.0 * PI * f0 * t).sin()
            + 0.3 * (2.0 * PI * f0 * 2.0 * t).sin()
            + 0.2 * (2.0 * PI * f0 * 3.0 * t).sin();

        // Add amplitude variation to simulate speech prosody
        let envelope = 0.5 + 0.5 * (2.0 * PI * 0.5 * t).sin();

        source_audio.push(sample * envelope * 0.5);
    }

    // Create source and target speaker feature vectors (256-dim)
    let source_features = [[0.0f32; 256]];
    let target_features = [[0.0f32; 256]];

    // Process audio data
    let converted_audio = tcc
        .convert(&source_audio, &source_features, &target_features)
        .await?;

    // Verify converted audio
    // Converted audio length should be similar to source
    assert!(
        (converted_audio.0.len() as i32 - source_audio.len() as i32).abs() < 1000,
        "Converted audio length should be close to source length"
    );

    // Converted audio shouldn't be all zeros
    let sum: f32 = converted_audio.0.iter().map(|&x| x.abs()).sum();
    assert!(sum > 0.0, "Converted audio shouldn't be all zeros");

    Ok(())
}

#[tokio::test]
async fn test_tcc_with_real_features() -> anyhow::Result<()> {
    // This test uses real feature extractor to generate target features
    let mut tcc = ToneColorConverter::new("../checkpoint/tone_color_converter.onnx")?;
    let mut see = SpeakerEmbeddingExtractor::new("../checkpoint/speaker_embedding_extractor.onnx")?;

    // Create two different simulated speech signals
    let sample_rate = 16000;
    let duration_secs = 3.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;

    // Source audio - lower fundamental frequency (male voice)
    let mut source_audio = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let f0 = 120.0; // Lower fundamental frequency
        let sample = 0.5 * (2.0 * PI * f0 * t).sin()
            + 0.3 * (2.0 * PI * f0 * 2.0 * t).sin()
            + 0.2 * (2.0 * PI * f0 * 3.0 * t).sin();
        let envelope = 0.5 + 0.5 * (2.0 * PI * 0.5 * t).sin();
        source_audio.push(sample * envelope * 0.5);
    }

    // Target audio - higher fundamental frequency (female voice)
    let mut target_audio = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let f0 = 220.0; // 较高的基频
        let sample = 0.5 * (2.0 * PI * f0 * t).sin()
            + 0.3 * (2.0 * PI * f0 * 2.0 * t).sin()
            + 0.2 * (2.0 * PI * f0 * 3.0 * t).sin();
        let envelope = 0.5 + 0.5 * (2.0 * PI * 0.5 * t).sin();
        target_audio.push(sample * envelope * 0.5);
    }

    // Extract features from source and target audio
    let source_features = see.extract(&source_audio, 1).await?;
    let target_features = see.extract(&target_audio, 1).await?;

    // Perform voice conversion using extracted features
    let converted_audio = tcc
        .convert(&source_audio, &source_features, &target_features)
        .await?;

    // 转换后的音频长度应该与源音频相同或接近
    assert!(
        (converted_audio.0.len() as i32 - source_audio.len() as i32).abs() < 1000,
        "Converted audio length should be close to source length"
    );

    // 转换后的音频不应该全是零
    let sum: f32 = converted_audio.0.iter().map(|&x| x.abs()).sum();
    assert!(sum > 0.0, "Converted audio shouldn't be all zeros");

    Ok(())
}

#[tokio::test]
async fn test_tcc_empty_input() -> anyhow::Result<()> {
    let mut tcc = ToneColorConverter::new("../checkpoint/tone_color_converter.onnx")?;

    // Test empty input
    let empty_audio = Vec::<f32>::new();
    let source_features = vec![0.0f32; 256];
    let target_features = vec![0.0f32; 256];

    let source_features_array: Vec<[f32; 256]> = source_features
        .chunks(256)
        .map(|chunk| {
            let mut arr = [0.0; 256];
            arr.copy_from_slice(chunk);
            arr
        })
        .collect();

    let target_features_array: Vec<[f32; 256]> = target_features
        .chunks(256)
        .map(|chunk| {
            let mut arr = [0.0; 256];
            arr.copy_from_slice(chunk);
            arr
        })
        .collect();

    let result = tcc
        .convert(&empty_audio, &source_features_array, &target_features_array)
        .await;

    // 空输入应该返回错误
    assert!(result.is_err(), "Processing empty input should return error");

    Ok(())
}

#[tokio::test]
async fn test_tcc_feature_influence() -> anyhow::Result<()> {
    let mut tcc = ToneColorConverter::new("../checkpoint/tone_color_converter.onnx")?;

    // Create a simple audio signal
    let sample_rate = 16000;
    let duration_secs = 3.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;

    let mut audio_data = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let f0 = 150.0;
        let sample = 0.5 * (2.0 * PI * f0 * t).sin();
        audio_data.push(sample);
    }

    // Create two different feature vectors
    let mut features1 = [[0.0f32; 256]];
    let mut features2 = [[0.0f32; 256]];

    // Feature vector 1: all set to 0.1
    for i in 0..256 {
        features1[0][i] = 0.1;
    }

    // Feature vector 2: all set to -0.1
    for i in 0..256 {
        features2[0][i] = -0.1;
    }

    // 使用两个不同的特征向量进行转换
    let converted1 = tcc.convert(&audio_data, &features1, &features1).await?;
    let converted2 = tcc.convert(&audio_data, &features2, &features2).await?;

    // 验证两个不同特征向量产生的结果应该不同
    // 计算两个转换结果的差异
    let mut diff_sum = 0.0;
    let min_len = converted1.0.len().min(converted2.0.len());

    for i in 0..min_len {
        diff_sum += (converted1.0[i] - converted2.0[i]).abs();
    }

    // 不同的特征向量应该产生不同的转换结果
    assert!(diff_sum > 0.0, "Different feature vectors should produce different conversion results");

    Ok(())
}
