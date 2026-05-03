//! Sonic 变速不变调处理示例
//!
//! 演示如何使用 SonicStream 进行音频变速处理。
//!
//! 运行方式：cargo run --features sonic --example sonic

use voxudio::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut stream = SonicStream::new(32000, 1)?;
    let (input, _) = load_audio::<32000, f32, _>("../asset/test3.wav", true).await?;
    let duration_ms = input.len() * 1000 / (stream.get_sample_rate() * stream.get_channels());
    println!(
        "输入样本数: {} ({}ms @ {}Hz)",
        input.len(),
        duration_ms,
        stream.get_sample_rate()
    );

    // === 使用 SonicStream::change_speed 便捷方法 ===
    println!("\n=== 使用 SonicStream::change_speed 便捷方法 ===");
    let mut player = AudioPlayer::new()?;
    player.play()?;

    for &speed in &[0.5, 0.75, 1.0, 1.5, 2.0] {
        stream.set_speed(speed);
        let output = stream.change_speed(&input)?;
        let output_ms = output.len() * 1000 / (stream.get_sample_rate() * stream.get_channels());
        println!(
            "  速度 {:.2}x: 输出 {} 样本 (~{}ms)",
            speed,
            output.len(),
            output_ms
        );
        player.write::<32000, f32>(&output, 1).await?;
    }

    // === 使用流式 API ===
    println!("\n=== 使用流式 API ===");
    stream.set_speed(1.5);
    println!("当前速度: {:.2}x", stream.get_speed());

    // 分段写入
    let chunk_size = 1600;
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
    println!("流式处理 1.5x: 输出 {} 样本", total_output.len());

    // === Pitch 和 Rate ===
    println!("\n=== Pitch 和 Rate ===");

    // Pitch 改变音高但不改变时长
    let mut stream = SonicStream::new(16000, 1)?;
    stream.set_pitch(2.0);
    stream.write::<f32>(&input)?;
    stream.flush()?;
    let output = stream.read::<f32>(stream.get_samples_available());
    println!(
        "Pitch 2.0x: 输出 {} 样本（长度不变，音高升高）",
        output.len()
    );

    // Rate 同时改变速度和音高
    let mut stream = SonicStream::new(16000, 1)?;
    stream.set_rate(2.0);
    stream.write::<f32>(&input)?;
    stream.flush()?;
    let output = stream.read::<f32>(stream.get_samples_available());
    println!("Rate 2.0x: 输出 {} 样本（变速变调）", output.len());

    Ok(())
}
