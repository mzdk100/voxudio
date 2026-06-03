"""
Pure ONNX Runtime inference for X-ASR-zh-en streaming ASR model.

Replaces sherpa-onnx dependency with direct onnxruntime + kaldi-native-fbank.
Supports all 4 chunk variants: 160ms, 480ms, 960ms, 1920ms.

Usage:
    # File recognition
    python x_asr_onnx_infer.py --wav audio.wav --chunk 960ms

    # As a library
    from x_asr_onnx_infer import XAsrOnnxRecognizer
    rec = XAsrOnnxRecognizer(encoder_path, decoder_path, joiner_path, tokens_path)
    text = rec.recognize_file("audio.wav")
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import kaldi_native_fbank as knf


# ---------------------------------------------------------------------------
# Token table
# ---------------------------------------------------------------------------

class TokenTable:
    """Load tokens.txt (SentencePiece-style: '<token> <id>' per line)."""

    def __init__(self, path: str):
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                token_text, token_id = parts[0], int(parts[1])
                self.id_to_token[token_id] = token_text
                self.token_to_id[token_text] = token_id

        self.blank_id = self.token_to_id.get("<blk>", 0)
        self.sos_eos_id = self.token_to_id.get("<sos/eos>", 1)
        self.unk_id = self.token_to_id.get("<unk>", 4015)
        self.vocab_size = max(self.id_to_token.keys()) + 1

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Convert token IDs to text string, replacing ▁ with space."""
        tokens = [self.id_to_token.get(tid, "") for tid in token_ids]
        text = "".join(tokens)
        text = text.replace("▁", " ").strip()
        return text


# ---------------------------------------------------------------------------
# Feature extraction (kaldi-native-fbank, matching sherpa-onnx config)
# ---------------------------------------------------------------------------

def _make_fbank_opts(sample_rate: int, feature_dim: int) -> knf.FbankOptions:
    """Create FbankOptions matching sherpa-onnx defaults."""
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.dither = 0.0
    opts.frame_opts.preemph_coeff = 0.97
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.snip_edges = False
    opts.mel_opts.num_bins = feature_dim
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.high_freq = -400.0  # Nyquist - 400
    opts.mel_opts.vtln_low = 100.0
    opts.mel_opts.vtln_high = -500.0
    return opts


class FbankExtractor:
    """Online Fbank feature extractor using kaldi-native-fbank."""

    def __init__(self, sample_rate: int = 16000, feature_dim: int = 80):
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self._opts = _make_fbank_opts(sample_rate, feature_dim)
        self.fbank = knf.OnlineFbank(self._opts)

    def accept_waveform(self, samples: np.ndarray):
        """Feed audio samples (float32, [-1, 1] range)."""
        self.fbank.accept_waveform(self.sample_rate, samples)

    @property
    def num_frames_ready(self) -> int:
        return self.fbank.num_frames_ready

    def get_frames(self, start: int, n: int) -> np.ndarray:
        """Get n frames starting from index start. Returns (n, feature_dim)."""
        features = np.empty((n, self.feature_dim), dtype=np.float32)
        for i in range(n):
            features[i, :] = self.fbank.get_frame(start + i)
        return features

    def reset(self):
        self.fbank = knf.OnlineFbank(self._opts)


# ---------------------------------------------------------------------------
# Encoder cache management
# ---------------------------------------------------------------------------

def _parse_encoder_metadata(model_path: str) -> dict:
    """Parse encoder ONNX metadata to get architecture config."""
    import onnx
    model = onnx.load(model_path)
    return {prop.key: prop.value for prop in model.metadata_props}


def _build_cache_shapes(meta: dict) -> list[tuple[str, tuple[int, ...], np.dtype]]:
    """Build list of (name, shape, dtype) for all per-layer encoder cache inputs.

    Order matches the encoder ONNX input order (19 layers * 6 tensors each).
    """
    layers_per_stack = [int(x) for x in meta["num_encoder_layers"].split(",")]
    encoder_dims = [int(x) for x in meta["encoder_dims"].split(",")]
    query_head_dims = [int(x) for x in meta["query_head_dims"].split(",")]
    value_head_dims = [int(x) for x in meta["value_head_dims"].split(",")]
    num_heads = [int(x) for x in meta["num_heads"].split(",")]
    cnn_kernels = [int(x) for x in meta["cnn_module_kernels"].split(",")]
    left_ctx = [int(x) for x in meta["left_context_len"].split(",")]

    entries = []
    layer_idx = 0
    for stack, n_layers in enumerate(layers_per_stack):
        enc_dim = encoder_dims[stack]
        key_dim = query_head_dims[stack] * num_heads[stack]
        val_dim = value_head_dims[stack] * num_heads[stack]
        ctx = left_ctx[stack]
        cnn_half = cnn_kernels[stack] // 2
        nonlin_dim = 3 * enc_dim // 4

        for _ in range(n_layers):
            entries.append((f"cached_key_{layer_idx}",        (ctx, 1, key_dim),      np.float32))
            entries.append((f"cached_nonlin_attn_{layer_idx}", (1, 1, ctx, nonlin_dim), np.float32))
            entries.append((f"cached_val1_{layer_idx}",       (ctx, 1, val_dim),      np.float32))
            entries.append((f"cached_val2_{layer_idx}",       (ctx, 1, val_dim),      np.float32))
            entries.append((f"cached_conv1_{layer_idx}",      (1, enc_dim, cnn_half), np.float32))
            entries.append((f"cached_conv2_{layer_idx}",      (1, enc_dim, cnn_half), np.float32))
            layer_idx += 1

    return entries


# ---------------------------------------------------------------------------
# X-ASR ONNX Runtime Recognizer (file-based)
# ---------------------------------------------------------------------------

class XAsrOnnxRecognizer:
    """Streaming ASR recognizer using pure ONNX Runtime.

    Implements the Zipformer2 transducer greedy decode loop:
    encoder -> (per-frame) decoder -> joiner -> argmax
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        joiner_path: str,
        tokens_path: str,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        num_threads: int = 1,
        provider: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim

        # Load tokens
        self.tokens = TokenTable(tokens_path)

        # ONNX session options
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = num_threads
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = [provider if provider != "cpu" else "CPUExecutionProvider"]
        available = ort.get_available_providers()
        if providers[0] not in available:
            providers = ["CPUExecutionProvider"]

        self.encoder_sess = ort.InferenceSession(encoder_path, opts, providers=providers)
        self.decoder_sess = ort.InferenceSession(decoder_path, opts, providers=providers)
        self.joiner_sess = ort.InferenceSession(joiner_path, opts, providers=providers)

        # Read metadata
        self._enc_meta = _parse_encoder_metadata(encoder_path)
        self.T = int(self._enc_meta["T"])
        self.decode_chunk_len = int(self._enc_meta["decode_chunk_len"])
        self.right_context = self.T - self.decode_chunk_len

        import onnx
        dm = onnx.load(decoder_path)
        dec_meta = {p.key: p.value for p in dm.metadata_props}
        self.context_size = int(dec_meta.get("context_size", 2))

        # Cache tensor names and shapes
        self._enc_input_names = [i.name for i in self.encoder_sess.get_inputs()]
        self._enc_output_names = [o.name for o in self.encoder_sess.get_outputs()]
        self._dec_input_name = self.decoder_sess.get_inputs()[0].name
        self._dec_output_name = self.decoder_sess.get_outputs()[0].name
        self._join_input_names = [i.name for i in self.joiner_sess.get_inputs()]
        self._join_output_name = self.joiner_sess.get_outputs()[0].name
        self._cache_shapes = _build_cache_shapes(self._enc_meta)

    def _init_caches(self) -> dict[str, np.ndarray]:
        """Initialize all encoder cache tensors to zeros."""
        caches = {name: np.zeros(shape, dtype=dt) for name, shape, dt in self._cache_shapes}
        embed_dim = ((self.feature_dim - 1) // 2 - 1) // 2
        caches["embed_states"] = np.zeros((1, 128, 3, embed_dim), dtype=np.float32)
        caches["processed_lens"] = np.zeros((1,), dtype=np.int64)
        return caches

    def _run_encoder(self, features: np.ndarray, caches: dict[str, np.ndarray]):
        """Run encoder, return (encoder_out, new_caches)."""
        feed = {}
        for name in self._enc_input_names:
            if name == "x":
                feed[name] = features
            elif name in caches:
                feed[name] = caches[name]
            else:
                raise ValueError(f"Missing encoder input: {name}")

        outputs = self.encoder_sess.run(self._enc_output_names, feed)
        out_map = dict(zip(self._enc_output_names, outputs))

        new_caches = {}
        for name in self._enc_input_names:
            if name == "x":
                continue
            new_name = f"new_{name}"
            new_caches[name] = out_map[new_name] if new_name in out_map else caches[name]

        return out_map["encoder_out"], new_caches

    def _run_decoder(self, token_ids: list[int]) -> np.ndarray:
        """Run decoder with context tokens, return decoder_out."""
        ctx = token_ids[-self.context_size:] if len(token_ids) >= self.context_size \
            else [0] * (self.context_size - len(token_ids)) + token_ids
        y = np.array([ctx], dtype=np.int64)
        return self.decoder_sess.run([self._dec_output_name], {self._dec_input_name: y})[0]

    def _run_joiner(self, enc_frame: np.ndarray, dec_out: np.ndarray) -> np.ndarray:
        """Run joiner on one encoder frame + decoder output, return logits."""
        feed = {self._join_input_names[0]: enc_frame, self._join_input_names[1]: dec_out}
        return self.joiner_sess.run([self._join_output_name], feed)[0]

    def recognize_file(self, wav_path: str) -> str:
        """Recognize a WAV file and return the transcript."""
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != self.sample_rate:
            import scipy.signal
            n = int(len(audio) * self.sample_rate / sr)
            audio = scipy.signal.resample(audio, n).astype(np.float32)
        return self.recognize_samples(audio)

    def recognize_samples(self, audio: np.ndarray) -> str:
        """Recognize audio samples (float32, sample_rate Hz). Returns transcript."""
        fbank = FbankExtractor(self.sample_rate, self.feature_dim)
        fbank.accept_waveform(audio)
        n = fbank.num_frames_ready

        if n < self.T:
            return ""

        caches = self._init_caches()
        num_processed = 0
        token_ids: list[int] = []
        decoder_out: Optional[np.ndarray] = None

        while num_processed + self.T <= n:
            chunk = fbank.get_frames(num_processed, self.T)
            features = chunk[np.newaxis, :, :]  # (1, T, feature_dim)
            num_processed += self.decode_chunk_len

            encoder_out, caches = self._run_encoder(features, caches)

            for t in range(encoder_out.shape[1]):
                if decoder_out is None:
                    decoder_out = self._run_decoder(token_ids)

                logits = self._run_joiner(encoder_out[:, t, :], decoder_out)
                pred_id = int(np.argmax(logits[0]))

                if pred_id != self.tokens.blank_id and pred_id != self.tokens.unk_id:
                    token_ids.append(pred_id)
                    decoder_out = self._run_decoder(token_ids)

        return self.tokens.decode_tokens(token_ids)


# ---------------------------------------------------------------------------
# Streaming recognizer (for real-time / WebSocket use)
# ---------------------------------------------------------------------------

class XAsrStreamingRecognizer:
    """Streaming recognizer that accepts audio incrementally.

    Designed for real-time use (WebSocket servers, microphone input).
    Maintains internal state (encoder caches, decoder output, feature extractor).

    Usage:
        rec = XAsrStreamingRecognizer(encoder, decoder, joiner, tokens)

        # Feed audio chunks as they arrive
        rec.accept_waveform(samples)  # float32, 16kHz
        while rec.is_ready():
            rec.decode_chunk()
        print(rec.get_partial_result())

        # Signal end of utterance
        rec.input_finished()
        while rec.is_ready():
            rec.decode_chunk()
        print(rec.get_final_result())

        # Reset for next utterance
        rec.reset()
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        joiner_path: str,
        tokens_path: str,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        num_threads: int = 1,
        provider: str = "cpu",
    ):
        # Reuse the core recognizer for model loading
        self._rec = XAsrOnnxRecognizer(
            encoder_path, decoder_path, joiner_path, tokens_path,
            sample_rate, feature_dim, num_threads, provider,
        )
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self.T = self._rec.T
        self.decode_chunk_len = self._rec.decode_chunk_len

        self._fbank = FbankExtractor(sample_rate, feature_dim)
        self._caches = self._rec._init_caches()
        self._num_processed = 0
        self._token_ids: list[int] = []
        self._decoder_out: Optional[np.ndarray] = None
        self._finished = False

    def accept_waveform(self, samples: np.ndarray):
        """Feed audio samples (float32, 16kHz). Can be called multiple times."""
        self._fbank.accept_waveform(samples)

    def is_ready(self) -> bool:
        """Check if enough frames are available for the next decode chunk."""
        return (not self._finished and
                self._num_processed + self.T <= self._fbank.num_frames_ready)

    def decode_chunk(self) -> int:
        """Decode one chunk. Returns number of new tokens emitted."""
        if not self.is_ready():
            return 0

        chunk = self._fbank.get_frames(self._num_processed, self.T)
        features = chunk[np.newaxis, :, :]
        self._num_processed += self.decode_chunk_len

        encoder_out, self._caches = self._rec._run_encoder(features, self._caches)

        tokens_before = len(self._token_ids)
        for t in range(encoder_out.shape[1]):
            if self._decoder_out is None:
                self._decoder_out = self._rec._run_decoder(self._token_ids)

            logits = self._rec._run_joiner(encoder_out[:, t, :], self._decoder_out)
            pred_id = int(np.argmax(logits[0]))

            if pred_id != self._rec.tokens.blank_id and pred_id != self._rec.tokens.unk_id:
                self._token_ids.append(pred_id)
                self._decoder_out = self._rec._run_decoder(self._token_ids)

        return len(self._token_ids) - tokens_before

    def input_finished(self):
        """Signal that no more audio will be fed."""
        self._finished = True

    def get_partial_result(self) -> str:
        """Get current partial transcript."""
        return self._rec.tokens.decode_tokens(self._token_ids)

    def get_final_result(self) -> str:
        """Get final transcript (call after input_finished + all decode_chunk)."""
        return self._rec.tokens.decode_tokens(self._token_ids)

    @property
    def is_endpoint(self) -> bool:
        """Simple endpoint detection: trailing silence >= 2.4s."""
        # trailing_blanks * frame_shift * subsampling_factor
        # We approximate: if no new tokens for many chunks, it's an endpoint
        return False  # Simplified; extend as needed

    def reset(self):
        """Reset state for a new utterance."""
        self._fbank.reset()
        self._caches = self._rec._init_caches()
        self._num_processed = 0
        self._token_ids = []
        self._decoder_out = None
        self._finished = False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="X-ASR-zh-en pure ONNX Runtime inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python x_asr_onnx_infer.py --wav audio.wav
  python x_asr_onnx_infer.py --wav audio.wav --chunk 160ms
  python x_asr_onnx_infer.py --wav audio.wav --chunk 960ms --provider cuda
        """,
    )
    parser.add_argument("--wav", type=str, required=True, help="Path to WAV file")
    parser.add_argument("--chunk", type=str, default="960ms",
                        choices=["160ms", "480ms", "960ms", "1920ms"],
                        help="Chunk size variant (default: 960ms)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Model directory (default: auto-detect)")
    parser.add_argument("--threads", type=int, default=1, help="ONNX Runtime threads")
    parser.add_argument("--provider", type=str, default="cpu",
                        help="Execution provider: cpu, cuda (default: cpu)")
    args = parser.parse_args()

    # Resolve model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        candidates = [
            Path(__file__).parent.parent / "checkpoint" / "x-asr-models" / f"chunk-{args.chunk}-model",
            Path(f"checkpoint/x-asr-models/chunk-{args.chunk}-model"),
        ]
        model_dir = next((d for d in candidates if d.exists()), candidates[-1])

    if not model_dir.exists():
        print(f"Error: model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    encoder = str(model_dir / f"encoder-{args.chunk}.onnx")
    decoder = str(model_dir / f"decoder-{args.chunk}.onnx")
    joiner = str(model_dir / f"joiner-{args.chunk}.onnx")
    tokens = str(model_dir / "tokens.txt")

    for p in [encoder, decoder, joiner, tokens]:
        if not os.path.exists(p):
            print(f"Error: not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Chunk: {args.chunk}  Model: {model_dir}")

    recognizer = XAsrOnnxRecognizer(
        encoder, decoder, joiner, tokens,
        num_threads=args.threads,
        provider=args.provider,
    )

    print(f"Recognizing: {args.wav}")
    t0 = time.time()
    text = recognizer.recognize_file(args.wav)
    elapsed = time.time() - t0

    print(f"\nResult: {text}")
    print(f"Time:   {elapsed:.3f}s")


if __name__ == "__main__":
    main()
