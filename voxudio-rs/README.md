# Voxudio

[English Version](README-en.md)

Voxudio 是一个用 Rust 编写的高性能音频处理库，专注于语音处理和音色转换功能。它提供了一套完整的工具，用于音频采集、播放、语音活动检测、说话人特征提取和音色转换。

## 特性

- 🎤 **音频设备管理**
    - 音频采集：支持从各种音频输入设备采集数据
    - 音频播放：支持实时音频播放
- 🎵 **音频编解码支持**
    - OPUS：支持OPUS音频格式的编码和解码
- 🔍 **语音活动检测 (VAD)**
    - 实时检测音频中的语音活动
    - 精确区分语音和非语音部分
- 👤 **说话人特征提取 (SEE)**
    - 提取 256 维说话人特征向量
    - 支持说话人识别和验证
- 🎭 **音色转换 (TCC)**
    - 实时音色转换
    - 保持原始语音内容和情感
- 📱 **跨平台支持**
    - 支持 Windows、Linux、macOS
    - Android 平台支持
    - iOS 平台尚未全面测试

## 模型文件

由于模型文件较大，未包含在版本控制中。请从以下地址下载模型文件：
https://github.com/mzdk100/voxudio/releases/tag/model

下载后将模型文件放置在项目根目录的 `checkpoint` 文件夹中。

## 安装

将以下依赖添加到你的 `Cargo.toml` 文件中：

```shell
cargo add voxudio
```

## 使用示例

1. [音频采集](examples/ac.rs)
2. [音频播放](examples/ap.rs)
3. [语音活动检测](examples/vad.rs)
4. [音色转换](examples/tcc.rs)
5. [Android 使用示例](examples/android)
   运行 Android 示例：
    1. 确保已安装 Android SDK 和 NDK
    2. 进入 examples/android 目录
    3. 在 Windows 上运行：
   ```bash
   run.bat
   ``
   在 Linux/macOS 上运行：
   ```bash
   ./run.sh
   ```

## 性能优化

- 使用 ONNX Runtime 进行高效的模型推理
- 基于 Tokio 的异步处理
- 优化的音频数据处理流程

## 许可证

本项目采用 Apache-2.0 许可证。详见 [LICENSE](../LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建你的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的修改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个 Pull Request

## 联系我们

如果你有任何问题或建议，欢迎提交 Issue 或通过以下方式联系我们：

- Issue Tracker: github.com/mzdk100/voxudio/issues
- Email: mzdk100@foxmail.com
