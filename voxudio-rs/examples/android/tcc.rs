//! 音色转换安卓示例
//! 运行 `.\run.bat` 或 `./run.sh`

use {
    std::{error::Error, path::Path},
    tokio::runtime::Builder,
    voxudio::*,
};

#[mobile_entry_point::mobile_entry_point]
fn main() {
    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(run()).unwrap();
}

async fn run() -> Result<(), Box<dyn Error>> {
    let assets_dir = Path::new("/data/local/tmp");
    let mut vad = VoiceActivityDetector::new(assets_dir.join("voice_activity_detector.onnx"))?;
    let mut see =
        SpeakerEmbeddingExtractor::new(assets_dir.join("speaker_embedding_extractor.onnx"))?;
    let mut tcc = ToneColorConverter::new(assets_dir.join("tone_color_converter.onnx"))?;

    let (src_audio, src_channels) =
        load_audio::<22050, _>(assets_dir.join("test6.wav"), false).await?;
    let vad_audio = vad
        .retain_speech_only::<22050>(&src_audio, src_channels)
        .await?;
    let src_se = see.extract(&vad_audio, src_channels).await?;

    let (tgt_audio, tgt_channels) =
        load_audio::<22050, _>(assets_dir.join("bajie.mp3"), false).await?;
    let vad_audio = vad
        .retain_speech_only::<22050>(&tgt_audio, tgt_channels)
        .await?;
    let tgt_se = see.extract(&vad_audio, tgt_channels).await?;

    let (out_audio, out_channels, cost) = tcc.convert(&src_audio, &src_se, &tgt_se).await?;
    println!("Convert cost: {:?}", cost);

    let mut ap = AudioPlayer::new()?;
    ap.play()?;
    ap.write::<22050>(&out_audio, out_channels).await?;

    Ok(())
}
