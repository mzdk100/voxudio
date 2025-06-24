//! 音频采集示例程序
//!
//! 本程序演示了如何使用voxudio库进行音频采集和实时播放。
//! 主要功能包括：
//! - 初始化音频采集器(AudioCollector)采集音频数据
//! - 初始化音频播放器(AudioPlayer)播放音频
//! - 在主循环中实时读取采集的音频数据并写入播放器
//!
//! 注意：本示例使用固定16000Hz采样率和单声道(1)配置

use voxudio::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut ac = AudioCollector::new()?;
    ac.collect()?;
    let mut ap = AudioPlayer::new()?;
    ap.play()?;
    loop {
        let audio = ac.read::<16000>(1).await?;
        ap.write::<16000>(&audio, 1).await?;
    }
}
