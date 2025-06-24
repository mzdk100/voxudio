import os
import soundfile
import time
from onnxexp import export_tcc
from torch import (
    cuda,
    load,
    no_grad,
)
from model import SpeakerEmbeddingExtractor, ToneColorConverter, VoiceActivityDetector


FILTER_LENGTH = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
SAMPLING_RATE = 22050
GIN_CHANNELS = 256

asset_dir = 'asset'
model_dir = "checkpoint"
device = "cuda:0" if cuda.is_available() else "cpu"

tcc = ToneColorConverter(
    FILTER_LENGTH, GIN_CHANNELS, HOP_LENGTH, SAMPLING_RATE, WIN_LENGTH
).to(device)
tcc.load_state_dict(load(os.path.join(model_dir, "tone_color_converter.pt")))
see = SpeakerEmbeddingExtractor(
    FILTER_LENGTH, GIN_CHANNELS, HOP_LENGTH, SAMPLING_RATE, WIN_LENGTH
).to(device)
see.load_state_dict(load(os.path.join(model_dir, "speaker_embedding_extractor.pt")))
vad = VoiceActivityDetector(os.path.join(model_dir, "voice_activity_detector.onnx"))

tgt_se, _ = see.get_se(os.path.join(asset_dir, 'houge.mp3'), vad, device)
src_se, src_audio = see.get_se(os.path.join(asset_dir, 'test7.wav'), vad, device)

with no_grad():
    t = time.time()
    audio = tcc(src_audio, src_se, tgt_se).cpu().numpy()
    print(time.time() - t)
    soundfile.write("out.wav", audio, SAMPLING_RATE)
    export_tcc(
        tcc,
        os.path.join(model_dir, "tone_color_converter.onnx"),
        src_audio,
        src_se,
        tgt_se,
    )
