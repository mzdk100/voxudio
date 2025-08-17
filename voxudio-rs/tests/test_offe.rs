use voxudio::*;

//noinspection SpellCheckingInspection
#[test]
fn test_offe() -> anyhow::Result<()> {
    let input = (0..1600)
        .map(|i| (i * i - i / 2) as f32 / 32767.)
        .collect::<Vec<_>>();

    let fbank = OnlineFbankFeatureExtractor::fbank()?
        .with_frame_opts(Default::default())?
        .with_mel_opts(MelBanksOptions {
            num_bins: 10,
            ..Default::default()
        })?
        .build()?;
    let features = fbank.extract::<16000>(&input);
    assert_eq!(features.len(), 80);

    let mfcc = OnlineFbankFeatureExtractor::mfcc()?.build()?;
    let features = mfcc.extract::<16000>(&input);
    assert_eq!(features.len(), 104);

    let whisper_fbank = OnlineFbankFeatureExtractor::whisper_fbank()?.build()?;
    let features = whisper_fbank.extract::<16000>(&input);
    assert_eq!(features.len(), 800);

    Ok(())
}
