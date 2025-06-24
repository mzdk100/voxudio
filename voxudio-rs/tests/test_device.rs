use {std::time::Duration, tokio::time::sleep, voxudio::*};

async fn test_audio_player_creation() -> anyhow::Result<()> {
    let player = AudioPlayer::new()?;
    println!("Successfully created AudioPlayer: {:?}", player);
    player.get_name()?;

    Ok(())
}

async fn test_audio_player_play() -> anyhow::Result<()> {
    let mut player = AudioPlayer::new()?;
    // Create short test audio
    let sample_rate = 48000;
    let duration_secs = 0.1;
    let frequency = 440.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;

    let mut test_audio = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
        test_audio.push(sample);
    }

    // Play audio
    println!("Attempting to play audio...");
    player.play()?;
    player.write::<48000>(&test_audio, 1).await?;

    Ok(())
}

async fn test_audio_collector_creation() -> anyhow::Result<()> {
    let Ok(collector) = AudioCollector::new() else {
        println!("Testing audio collector creation - requires audio input device");
        return Ok(());
    };
    println!("Successfully created AudioCollector: {:?}", collector);
    assert!(collector.get_name().is_ok(), "Should be able to get device name");

    Ok(())
}

async fn test_audio_collector_start_stop() -> anyhow::Result<()> {
    // Create audio collector
    let Ok(collector) = AudioCollector::new() else {
        println!("Testing audio collector start/stop - requires audio input device");
        return Ok(());
    };

    // Start collection
    println!("Attempting to start audio collection...");
    collector.collect()?;
    println!("Audio collection started, waiting 100ms...");
    sleep(Duration::from_millis(100)).await;

    // Stop collection
    println!("Attempting to stop audio collection...");
    collector.pause()?;

    Ok(())
}

async fn test_audio_collector_read() -> anyhow::Result<()> {
    // Create audio collector
    let Ok(mut collector) = AudioCollector::new() else {
        println!("Testing audio collector read - requires audio input device");
        return Ok(());
    };

    // Start collection
    println!("Attempting to start audio collection...");
    collector.collect()?;
    println!("Audio collection started, waiting 100ms...");
    sleep(Duration::from_millis(100)).await;

    // Read data - add more error handling and logging
    println!("Attempting to read audio data...");
    let data = collector.read::<48000>(2).await?;

    // Ensure to stop collection immediately after reading, regardless of success
    println!("Attempting to stop audio collection...");
    collector.pause()?;

    // Process read results

    println!("Successfully read {} audio samples", data.len());
    assert!(!data.is_empty(), "Should be able to read audio data");

    // Stop collection
    println!("Attempting to stop audio collection...");
    collector.pause()?;

    Ok(())
}

#[tokio::test]
async fn test_all() -> anyhow::Result<()> {
    test_audio_player_creation().await?;
    test_audio_player_play().await?;
    test_audio_collector_creation().await?;
    test_audio_collector_start_stop().await?;
    test_audio_collector_read().await?;

    Ok(())
}
