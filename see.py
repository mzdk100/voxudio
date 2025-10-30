import os
from torch import load
from model import SpeakerEmbeddingExtractor, VoiceActivityDetector
from onnxexp import export_see

asset_dir = 'asset'
model_dir = "checkpoint"
device = "cuda"
vad = VoiceActivityDetector(os.path.join(model_dir, "voice_activity_detector.onnx"))
see = SpeakerEmbeddingExtractor(1024, 256, 256, 22050, 1024).to(device)
see.load_state_dict(load(os.path.join(model_dir, "speaker_embedding_extractor.pt")))
se, audio = see.get_se(os.path.join(asset_dir, "bajie.mp3"), vad, device)
print(se.shape, audio.shape)
export_see(see, os.path.join(model_dir, "speaker_embedding_extractor.onnx"), audio)
