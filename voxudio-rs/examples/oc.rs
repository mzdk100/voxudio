//! OPUS编解码器示例程序
//!
//! 该程序演示了如何使用voxudio库编解码OPUS音频

use voxudio::*;

fn main() -> anyhow::Result<()> {
    let encoder = OpusCodec::new_encoder(24000, 2, OpusApplication::Audio)?;
    let data = [1.0; 240];
    let encoded = dbg!(encoder.encode(&data, 120, OpusCodec::MAX_PACKET_SIZE)?);
    let decoder = OpusCodec::new_decoder(24000, 2)?;
    let reconstructed = decoder.decode::<f32>(Some(&encoded), 120)?;
    dbg!(reconstructed);

    Ok(())
}
