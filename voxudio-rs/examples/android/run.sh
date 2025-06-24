#!/bin/bash

adb push ../../../checkpoint/speaker_embedding_extractor.onnx /data/local/tmp/speaker_embedding_extractor.onnx && \
adb push ../../../checkpoint/tone_color_converter.onnx /data/local/tmp/tone_color_converter.onnx && \
adb push ../../../checkpoint/voice_activity_detector.onnx /data/local/tmp/voice_activity_detector.onnx && \
adb push ../../../asset/test6.wav /data/local/tmp/test6.wav && \
adb push ../../../asset/bajie.mp3 /data/local/tmp/bajie.mp3 &&^
adb logcat -c && \
cargo apk2 run -p voxudio-android --release