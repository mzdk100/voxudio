from torch import hann_window, nn, stft, sqrt

_hann_window = {}


def spectrogram(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    global _hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in _hann_window:
        _hann_window[wnsize_dtype_device] = hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    if y.dim() < 3:
        y = y.unsqueeze(1)
    y = nn.functional.pad(
        y,
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(0)

    spec = stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec
