use {
    std::{cmp::min, f32::consts::PI},
    voxudio::*,
};

// 计算两个音频信号之间的信噪比（SNR）
fn calculate_snr(original: &[i16], decoded: &[i16]) -> f32 {
    let len = min(original.len(), decoded.len());
    let mut signal_power = 0.0;
    let mut noise_power = 0.0;

    for i in 0..len {
        let signal = original[i] as f32;
        let noise = signal - decoded[i] as f32;

        signal_power += signal * signal;
        noise_power += noise * noise;
    }

    if noise_power > 0.0 && signal_power > 0.0 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f32::INFINITY // 如果没有噪声，SNR为无穷大
    }
}

// 生成一个更复杂的音频信号（包含多个频率）
fn generate_complex_audio(duration_ms: usize, sample_rate: usize) -> Vec<i16> {
    let num_samples = sample_rate * duration_ms / 1000;
    let mut samples = Vec::with_capacity(num_samples * 2); // 立体声

    // 根据采样率调整频率，确保不会超过奈奎斯特频率
    // 对于8kHz采样率，最高频率应低于4kHz
    let max_freq = (sample_rate as f32 / 2.0) * 0.8; // 留出20%的余量

    // 根据采样率选择合适的频率
    let frequencies = [261.63, 329.63, 392.00, 523.25, 659.25]; // C4, E4, G4, C5, E5

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        // 组合多个频率
        let mut value = 0.0;
        for (j, &freq) in frequencies.iter().enumerate() {
            // 确保频率不超过最大允许频率
            if freq < max_freq {
                // 每个频率的振幅逐渐减小
                let amplitude = 0.1 / (j + 1) as f32; // 进一步降低振幅以避免溢出
                value += amplitude * f32::sin(2.0 * PI * freq * t);
            }
        }

        // 添加一些确定性"噪声"，但确保噪声频率不超过最大允许频率
        let noise_freq = max_freq * 0.5; // 噪声频率为最大允许频率的一半
        let noise = f32::sin(t * noise_freq) * f32::cos(t * (noise_freq * 0.7) * 0.005);
        value += noise;

        // 确保值在[-0.8, 0.8]范围内，进一步避免溢出
        value = value.max(-0.8).min(0.8);

        // 归一化并转换为i16，使用安全的转换方式
        let sample = (value * 30000.0) as i16; // 使用30000而不是32767，留出更多余量

        // 左右声道略有不同
        samples.push(sample); // 左声道
        samples.push((sample as f32 * 0.95) as i16); // 右声道（稍微安静一点）
    }

    samples
}

// 测试基本的 编解码功能
fn test_basic_encoding_decoding(
    sample_rate: usize,
    channels: usize,
    bitrate: usize,
) -> Result<(), OperationError> {
    println!("\n--- Testing basic encode/decode ({} Hz) ---", sample_rate);

    let frame_size = sample_rate * 20 / 1000;

    let input_audio = generate_complex_audio(500, sample_rate);

    let mut encoder = OpusCodec::new_encoder(sample_rate, channels, OpusApplication::Audio)?;
    encoder.set_bitrate(bitrate)?;
    encoder.set_complexity(10)?;
    encoder.set_bandwidth(OpusCodec::get_max_bandwidth_for_sample_rate(sample_rate))?;

    let decoder = OpusDecoder::new(sample_rate, channels)?;

    println!("Encoding and decoding multiple frames...");
    let frame_count = 5;
    let mut all_input = Vec::new();
    let mut all_decoded = Vec::new();

    for i in 0..frame_count {
        let start = i * frame_size * channels;
        let end = start + frame_size * channels;
        let end = end.min(input_audio.len());

        if start >= input_audio.len() {
            break;
        }

        let frame_input = &input_audio[start..end];
        all_input.extend_from_slice(frame_input);

        let encoded = encoder.encode(frame_input, frame_size, OpusCodec::MAX_PACKET_SIZE)?;
        println!("  Frame {}: encoded {} bytes", i + 1, encoded.len());

        let decoded = decoder.decode::<i16>(Some(&encoded), frame_size)?;
        println!("  Frame {}: decoded {} samples", i + 1, decoded.len());
        all_decoded.extend(decoded);
    }

    println!("Frame size: {} samples (20 ms)", frame_size);
    println!("Total input samples: {}", all_input.len());
    println!("Total decoded samples: {}", all_decoded.len());

    if !all_decoded.is_empty() {
        let min_len = min(all_input.len(), all_decoded.len());
        let snr = calculate_snr(&all_input[0..min_len], &all_decoded[0..min_len]);
        println!("Signal-to-noise ratio (SNR): {:.2} dB", snr);
    }

    Ok(())
}

#[test]
fn test_opus() -> anyhow::Result<()> {
    // Output Opus version information
    println!("Opus version: {}", crate::opus::OpusCodec::version());

    println!("\n=== Opus codec test - Basic encoding/decoding functionality ===");

    // Test different sample rates
    let sample_rates = [8000, 16000, 24000, 48000];
    let channels = 2;
    let bitrate = 256000; // 256 kbps

    for &rate in &sample_rates {
        test_basic_encoding_decoding(rate, channels, bitrate)?;
    }

    Ok(())
}
