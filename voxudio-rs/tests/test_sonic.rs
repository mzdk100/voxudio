use {std::f32::consts::PI, voxudio::*};

// 生成正弦波音频数据（f32, -1.0 ~ 1.0）
fn generate_sine_wave(
    freq: f32,
    duration_ms: usize,
    sample_rate: usize,
    channels: usize,
) -> Vec<f32> {
    let num_samples = sample_rate * duration_ms / 1000;
    let mut samples = Vec::with_capacity(num_samples * channels);
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let value = 0.5 * (2.0 * PI * freq * t).sin();
        for _ in 0..channels {
            samples.push(value);
        }
    }
    samples
}

#[test]
fn test_sonic_stream_create_destroy() -> anyhow::Result<()> {
    let stream = SonicStream::new(16000, 1)?;
    assert_eq!(stream.get_sample_rate(), 16000);
    assert_eq!(stream.get_channels(), 1);
    assert!((stream.get_speed() - 1.0).abs() < f32::EPSILON);
    assert!((stream.get_pitch() - 1.0).abs() < f32::EPSILON);
    assert!((stream.get_volume() - 1.0).abs() < f32::EPSILON);
    Ok(())
}

#[test]
fn test_sonic_stream_setters() -> anyhow::Result<()> {
    let mut stream = SonicStream::new(16000, 1)?;

    stream.set_speed(2.0);
    assert!((stream.get_speed() - 2.0).abs() < 0.01);

    stream.set_pitch(1.5);
    assert!((stream.get_pitch() - 1.5).abs() < 0.01);

    stream.set_rate(0.8);
    assert!((stream.get_rate() - 0.8).abs() < 0.01);

    stream.set_volume(0.5);
    assert!((stream.get_volume() - 0.5).abs() < 0.01);

    stream.set_quality(1);
    assert_eq!(stream.get_quality(), 1);

    Ok(())
}

#[test]
fn test_sonic_change_speed_float() -> anyhow::Result<()> {
    let mut stream = SonicStream::new(16000, 1)?;
    let input = generate_sine_wave(440.0, 1000, 16000, 1);

    // 2倍速：输出应约为输入的1/2长度
    stream.set_speed(2.0);
    let output = stream.change_speed(&input)?;
    let expected_len = input.len() / 2;
    assert!(
        (output.len() as f32 - expected_len as f32).abs() / (expected_len as f32) < 0.2,
        "Expected ~{} samples, got {}",
        expected_len,
        output.len()
    );

    // 0.5倍速：输出应约为输入的2倍长度
    stream.set_speed(0.5);
    let output = stream.change_speed(&input)?;
    let expected_len = input.len() * 2;
    assert!(
        (output.len() as f32 - expected_len as f32).abs() / (expected_len as f32) < 0.2,
        "Expected ~{} samples, got {}",
        expected_len,
        output.len()
    );

    Ok(())
}

#[test]
fn test_sonic_change_speed_short() -> anyhow::Result<()> {
    let mut stream = SonicStream::new(16000, 1)?;
    let input: Vec<i16> = (0..16000)
        .map(|i| ((i as f32 / 16000.0 * 2.0 * PI * 440.0).sin() * 16000.0) as i16)
        .collect();

    stream.set_speed(2.0);
    let output = stream.change_speed(&input)?;
    let expected_len = input.len() / 2;
    assert!(
        (output.len() as f32 - expected_len as f32).abs() / (expected_len as f32) < 0.3,
        "Expected ~{} samples, got {}",
        expected_len,
        output.len()
    );

    Ok(())
}

#[test]
fn test_sonic_streaming() -> anyhow::Result<()> {
    let sample_rate = 16000;
    let channels = 1;
    let input = generate_sine_wave(440.0, 500, sample_rate, channels);

    let mut stream = SonicStream::new(sample_rate, channels)?;
    stream.set_speed(1.5);

    // 分段写入
    let chunk_size = 1600; // 100ms
    let mut total_output = Vec::new();
    for chunk in input.chunks(chunk_size) {
        stream.write::<f32>(chunk)?;
        let available = stream.get_samples_available();
        if available > 0 {
            total_output.extend(stream.read::<f32>(available));
        }
    }
    stream.flush()?;
    let available = stream.get_samples_available();
    if available > 0 {
        total_output.extend(stream.read::<f32>(available));
    }

    // 流式处理后的总输出长度应与非流式相近
    stream.set_speed(1.5);
    let batch_output = stream.change_speed(&input)?;
    let len_diff = (total_output.len() as f32 - batch_output.len() as f32).abs();
    assert!(
        len_diff / (batch_output.len() as f32) < 0.2,
        "Streaming len={}, batch len={}",
        total_output.len(),
        batch_output.len()
    );

    Ok(())
}

#[test]
fn test_sonic_stereo() -> anyhow::Result<()> {
    let mut stream = SonicStream::new(16000, 2)?;
    let input = generate_sine_wave(440.0, 500, 16000, 2);

    stream.set_speed(2.0);
    let output = stream.change_speed(&input)?;
    let expected_len = input.len() / 2;
    assert!(
        (output.len() as f32 - expected_len as f32).abs() / (expected_len as f32) < 0.3,
        "Expected ~{} samples, got {}",
        expected_len,
        output.len()
    );

    Ok(())
}

#[test]
fn test_sonic_speed_identity() -> anyhow::Result<()> {
    let mut stream = SonicStream::new(16000, 1)?;
    let input = generate_sine_wave(440.0, 100, 16000, 1);

    // speed=1.0 应返回原始数据
    stream.set_speed(1.0);
    let output = stream.change_speed(&input)?;
    assert_eq!(output, input);

    Ok(())
}

#[test]
fn test_sonic_empty_input() -> anyhow::Result<()> {
    let mut stream = SonicStream::new(16000, 1)?;
    stream.set_speed(1.0);
    let output = stream.change_speed::<f32>(&[])?;
    assert!(output.is_empty());

    Ok(())
}

#[test]
fn test_sonic_pitch_and_rate() -> anyhow::Result<()> {
    let sample_rate = 16000;
    let channels = 1;
    let input = generate_sine_wave(440.0, 500, sample_rate, channels);

    // 测试 pitch 调整（变调不变速）
    let mut stream = SonicStream::new(sample_rate, channels)?;
    stream.set_pitch(2.0);
    stream.write::<f32>(&input)?;
    stream.flush()?;
    let output = stream.read::<f32>(stream.get_samples_available());
    assert!(!output.is_empty(), "Pitch output should not be empty");

    // 测试 rate 调整（变速变调）
    let mut stream = SonicStream::new(sample_rate, channels)?;
    stream.set_rate(2.0);
    stream.write::<f32>(&input)?;
    stream.flush()?;
    let output = stream.read::<f32>(stream.get_samples_available());
    assert!(!output.is_empty(), "Rate output should not be empty");

    Ok(())
}
