# Voxudio

[English Version](README-en.md)

Voxudio 是一个专注于语音处理和音色转换的项目，提供了一套完整的工具，用于音频采集、播放、语音活动检测、说话人特征提取和音色转换。

## 项目概述

Voxudio 项目提供了多种语言实现，以满足不同开发环境和性能需求：

- **Python 实现**：位于项目根目录，适合快速原型开发和研究
- **Rust 实现**：位于 [voxudio-rs](voxudio-rs) 目录，提供高性能和跨平台支持

## 核心功能

- 🎤 **音频设备管理**
    - 音频采集：支持从各种音频输入设备捕获数据
    - 音频播放：支持实时音频播放
- 🔍 **语音活动检测 (VAD)**
    - 实时检测音频中的语音活动
    - 准确区分语音和非语音段
- 👤 **说话人特征提取 (SEE)**
    - 提取256维说话人特征向量
    - 支持说话人识别和验证
- 🎭 **音色转换 (TCC)**
    - 实时语音音色转换
    - 保留原始语音内容和情感

## 目录结构

```
voxudio/
├── asset/              # 资源文件
├── checkpoint/         # 模型检查点
├── model/              # 模型定义
├── voxudio-rs/         # Rust 实现
├── *.py                # Python 实现文件
└── LICENSE             # 许可证文件
```

## 使用指南

### Python 实现

Python 实现提供了以下主要模块：

- `vad.py`: 语音活动检测
- `see.py`: 说话人特征提取
- `tcc.py`: 音色转换

示例用法：

```python
# 语音活动检测示例
import vad

# 说话人特征提取示例
import see

# 音色转换示例
import tcc
```

### Rust 实现

Rust 实现提供了更高的性能和更广泛的平台支持，详细信息请参阅 [voxudio-rs/README.md](voxudio-rs/README.md)。

## 许可证

本项目采用 Apache-2.0 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交问题和拉取请求！

1. Fork 本项目
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个拉取请求

## 联系方式

如果您有任何问题或建议，请提交 issue 或通过以下方式联系我们：

- Issue Tracker: github.com/mzdk100/voxudio/issues
- 电子邮件: mzdk100@foxmail.com
