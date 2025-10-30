from librosa import load
from torch import nn, tensor, float32, zeros, cat, stack
from utils import spectrogram


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0, layernorm=True):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            nn.utils.weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)
        if layernorm:
            self.layernorm = nn.LayerNorm(self.spec_channels)
        else:
            self.layernorm = None

    def forward(self, inputs, mask=None):
        N = inputs.size(0)

        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        if self.layernorm is not None:
            out = self.layernorm(out)

        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = nn.functional.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class SpeakerEmbeddingExtractor(nn.Module):
    def __init__(
        self, filter_length, gin_channels, hop_length, sampling_rate, win_length
    ):
        super().__init__()
        self.filter_length, self.hop_length, self.sampling_rate, self.win_length = (
            filter_length,
            hop_length,
            sampling_rate,
            win_length,
        )
        self.ref_enc = ReferenceEncoder(filter_length // 2 + 1, gin_channels)

    def forward_inner(self, audio):
        y = spectrogram(
            audio,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        ).to(audio.device)
        return self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)

    def forward(self, audio):
        interval = self.sampling_rate * 10
        gs = []
        for i in range(0, audio.shape[0], interval):
            ref_audio = audio[i : i + interval, ...]
            gs.append(self.forward_inner(ref_audio.T.unsqueeze(0)))
        return stack(gs).mean(0).squeeze(2)

    def get_se(self, audio_path, vad, device):
        SAMPLE_RATE = 16000
        audio_vad, sr = load(audio_path, sr=SAMPLE_RATE)
        segments = vad.get_speech_timestamps(
            tensor(audio_vad, dtype=float32), sampling_rate=SAMPLE_RATE
        )
        segments = [
            (
                int(seg["start"] / SAMPLE_RATE * self.sampling_rate),
                int(seg["end"] / SAMPLE_RATE * self.sampling_rate),
            )
            for seg in segments
        ]
        audio_ref, sr = load(audio_path, sr=self.sampling_rate, mono=False)
        audio_ref = tensor(audio_ref.T, dtype=float32, device=device)
        if audio_ref.ndim < 2:
            audio_ref = audio_ref.unsqueeze(1)
        audio_active = zeros(
            (0, audio_ref.shape[1]),
            dtype=float32,
            device=device,
        )
        for start_time, end_time in segments:
            audio_active = cat(
                (
                    audio_active,
                    audio_ref[start_time:end_time, ...],
                ),
                dim=0,
            )
        return self(audio_active), audio_ref
