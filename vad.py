import torchaudio
from model import VoiceActivityDetector


def read_audio(path: str, sampling_rate: int = 16000):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav = transform(wav)
        sr = sampling_rate
    assert sr == sampling_rate
    return wav.squeeze(0)


wav = read_audio("bajie.mp3")
vad = VoiceActivityDetector("checkpoint/voice_activity_detector.onnx")
segments = vad.get_speech_timestamps(wav)
print(segments)
