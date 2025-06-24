//! 音频播放器示例程序
//!
//! 该程序演示了如何使用voxudio库加载和播放音频文件
//! 主要功能包括:
//! - 创建音频播放器实例
//! - 加载WAV格式音频文件
//! - 播放和暂停音频

use voxudio::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut ap = AudioPlayer::new()?;
    ap.play()?;
    let (audio, channels) = load_audio::<32000, _>("../asset/test2.wav", false).await?;
    ap.write::<32000>(&audio, channels).await?;
    ap.pause()?;

    Ok(())
}