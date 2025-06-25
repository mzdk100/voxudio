# Voxudio

[中文版本](README.md)

Voxudio is a project focused on speech processing and voice conversion, providing a complete set of tools for audio capture, playback, voice activity detection, speaker feature extraction, and voice conversion.

## Project Overview

Voxudio project offers implementations in multiple languages to meet different development environments and performance requirements:

- **Python Implementation**: Located in the project root directory, suitable for rapid prototyping and research
- **Rust Implementation**: Located in the [voxudio-rs](voxudio-rs) directory, providing high performance and cross-platform support

## Core Features

- 🎤 **Audio Device Management**
    - Audio Capture: Supports capturing data from various audio input devices
    - Audio Playback: Supports real-time audio playback
- 🔍 **Voice Activity Detection (VAD)**
    - Real-time detection of speech activity in audio
    - Accurate distinction between speech and non-speech segments
- 👤 **Speaker Embedding Extraction (SEE)**
    - Extracts 256-dimensional speaker feature vectors
    - Supports speaker identification and verification
- 🎭 **Tone Color Conversion (TCC)**
    - Real-time voice conversion
    - Preserves original speech content and emotion

## Directory Structure

```
voxudio/
├── asset/              # Resource files
├── checkpoint/         # Model checkpoints
├── model/             # Model definitions
├── voxudio-rs/        # Rust implementation
├── *.py               # Python implementation files
└── LICENSE            # License file
```

## Usage Guide

### Python Implementation

The Python implementation provides the following main modules:

- `vad.py`: Voice Activity Detection
- `see.py`: Speaker Embedding Extraction
- `tcc.py`: Tone Color Conversion

Example usage:

```python
# Voice Activity Detection example
import vad

# Speaker Embedding Extraction example
import see

# Tone Color Conversion example
import tcc
```

### Rust Implementation

The Rust implementation offers higher performance and broader platform support. For detailed information, please refer to [voxudio-rs/README.md](voxudio-rs/README.md).

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

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
