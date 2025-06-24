import numpy as np
import onnxruntime
from torch import no_grad, zeros, cat, from_numpy, nn


class VoiceActivityDetector:
    def __init__(self, path, force_onnx_cpu=False):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        if (
            force_onnx_cpu
            and "CPUExecutionProvider" in onnxruntime.get_available_providers()
        ):
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts
            )
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()
        self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(
                f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)"
            )
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        self._state = zeros((2, batch_size, 128)).float()
        self._context = zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = zeros(batch_size, context_size)

        x = cat([self._context, x], dim=1)
        if sr in [8000, 16000]:
            ort_inputs = {
                "input": x.numpy(),
                "state": self._state.numpy(),
                "sr": np.array(sr, dtype="int64"),
            }
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs
            self._state = from_numpy(state)
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        out = from_numpy(out)
        return out

    @no_grad()
    def get_speech_timestamps(
        self,
        audio,
        threshold=0.5,
        sampling_rate=16000,
        min_speech_duration_ms=100,
        max_speech_duration_s=float("inf"),
        min_silence_duration_ms=1000,
        speech_pad_ms=30,
    ):
        """
        This method is used for splitting long audios into speech chunks using silero VAD

        Parameters
        ----------
        audio: torch.Tensor, one dimensional
            One dimensional float torch.Tensor, other types are casted to torch if possible

        threshold: float (default - 0.5)
            Speech threshold. VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates

        min_speech_duration_ms: int (default - 100 milliseconds)
            Final speech chunks shorter min_speech_duration_ms are thrown out

        max_speech_duration_s: int (default -  inf)
            Maximum duration of speech chunks in seconds
            Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence that lasts more than 100ms (if any), to prevent agressive cutting.
            Otherwise, they will be split aggressively just before max_speech_duration_s.

        min_silence_duration_ms: int (default - 1000 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        if len(audio.shape) > 1:
            for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
                audio = audio.squeeze(0)
            if len(audio.shape) > 1:
                raise ValueError(
                    "More than one dimension in audio. Are you trying to process audio with 2 channels?"
                )

        window_size_samples = 512 if sampling_rate == 16000 else 256
        self.reset_states()
        min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
        speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        max_speech_samples = (
            sampling_rate * max_speech_duration_s
            - window_size_samples
            - 2 * speech_pad_samples
        )
        min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

        audio_length_samples = len(audio)

        speech_probs = []
        for current_start_sample in range(0, audio_length_samples, window_size_samples):
            chunk = audio[
                current_start_sample : current_start_sample + window_size_samples
            ]
            if len(chunk) < window_size_samples:
                chunk = nn.functional.pad(
                    chunk, (0, int(window_size_samples - len(chunk)))
                )
            speech_prob = self(chunk, sampling_rate).item()
            speech_probs.append(speech_prob)
        triggered = False
        speeches = []
        current_speech = {}
        neg_threshold = max(threshold - 0.15, 0.01)
        temp_end = 0  # to save potential segment end (and tolerate some silence)
        prev_end = next_start = (
            0  # to save potential segment limits in case of maximum segment size reached
        )

        for i, speech_prob in enumerate(speech_probs):
            if (speech_prob >= threshold) and temp_end:
                temp_end = 0
                if next_start < prev_end:
                    next_start = window_size_samples * i

            if (speech_prob >= threshold) and not triggered:
                triggered = True
                current_speech["start"] = window_size_samples * i
                continue

            if (
                triggered
                and (window_size_samples * i) - current_speech["start"]
                > max_speech_samples
            ):
                if prev_end:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if (
                        next_start < prev_end
                    ):  # previously reached silence (< neg_thres) and is still not speech (< thres)
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = next_start = temp_end = 0
                else:
                    current_speech["end"] = window_size_samples * i
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = window_size_samples * i
                if (
                    ((window_size_samples * i) - temp_end)
                    > min_silence_samples_at_max_speech
                ):  # condition to avoid cutting in very short silence
                    prev_end = temp_end
                if (window_size_samples * i) - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech["end"] = temp_end
                    if (
                        current_speech["end"] - current_speech["start"]
                    ) > min_speech_samples:
                        speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

        if (
            current_speech
            and (audio_length_samples - current_speech["start"]) > min_speech_samples
        ):
            current_speech["end"] = audio_length_samples
            speeches.append(current_speech)

        for i, speech in enumerate(speeches):
            if i == 0:
                speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]["start"] - speech["end"]
                if silence_duration < 2 * speech_pad_samples:
                    speech["end"] += int(silence_duration // 2)
                    speeches[i + 1]["start"] = int(
                        max(0, speeches[i + 1]["start"] - silence_duration // 2)
                    )
                else:
                    speech["end"] = int(
                        min(audio_length_samples, speech["end"] + speech_pad_samples)
                    )
                    speeches[i + 1]["start"] = int(
                        max(0, speeches[i + 1]["start"] - speech_pad_samples)
                    )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )

        return speeches
