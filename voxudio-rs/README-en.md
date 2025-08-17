# Voxudio

[中文版本](README.md)

Voxudio is a high-performance audio processing library written in Rust, focusing on speech processing and voice conversion capabilities. It provides a complete set of tools for audio capture, playback, voice activity detection, speaker feature extraction, voice conversion, OPUS codec and online feature extraction.

## Features

- 🎤 **Audio Device Management**
    - Audio Capture: Supports capturing data from various audio input devices
    - Audio Playback: Supports real-time audio playback
- 🎵 **Audio Codec Support**
    - OPUS: Supports encoding and decoding OPUS audio format
- 🔍 **Voice Activity Detection (VAD)**
    - Real-time detection of speech activity in audio
    - Accurate distinction between speech and non-speech segments
- 👤 **Speaker Embedding Extraction (SEE)**
    - Extracts 256-dimensional speaker feature vectors
    - Supports speaker identification and verification
- 🎭 **Tone Color Conversion (TCC)**
    - Real-time voice conversion
    - Preserves original speech content and emotion
- 🧑‍🔬 **Online Feature Extractor (FBank/MFCC/Whisper FBank)**
    - Supports real-time extraction of Filter Bank (FBank), Mel-Frequency Cepstral Coefficients (MFCC), and Whisper FBank features from audio signals
    - Algorithms mainly based on [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
- 📱 **Cross-platform Support**
    - Supports Windows, Linux, macOS
    - Android platform support
    - iOS platform not fully tested yet

## Model Files

Due to their large size, model files are not included in version control. Please download the model files from:
https://github.com/mzdk100/voxudio/releases/tag/model

After downloading, place the model files in the `checkpoint` folder in the project root directory.

## Installation

Add the following dependency to your `Cargo.toml` file:

```shell
cargo add voxudio
```

## Usage Examples

1. [Audio Capture](examples/ac.rs)
2. [Audio Playback](examples/ap.rs)
3. [Voice Activity Detection](examples/vad.rs)
4. [Tone Color Conversion](examples/tcc.rs)
5. [OPUS Codec](examples/oc.rs)
6. [Online Feature Extraction](examples/offe.rs)
7. [Android Usage Example](examples/android)
   To run Android example:
    1. Ensure Android SDK and NDK are installed
    2. Navigate to examples/android directory
    3. On Windows run:
   ```bash
   run.bat
   ```
   On Linux/macOS run:
   ```bash
   ./run.sh
   ```

## Performance Optimization

- Uses ONNX Runtime for efficient model inference
- Asynchronous processing based on Tokio
- Optimized audio data processing pipeline

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](../LICENSE) file for details.

## Contributing

We welcome issues and pull requests!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a pull request

## Contact

If you have any questions or suggestions, please submit an issue or contact us via:

- Issue Tracker: github.com/mzdk100/voxudio/issues
- Email: mzdk100@foxmail.com
