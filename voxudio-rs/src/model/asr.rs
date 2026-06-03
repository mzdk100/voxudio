//! 流式自动语音识别（ASR）模块
//!
//! 基于 X-ASR-zh-en 模型的 Zipformer2 transducer 架构，
//! 使用纯 ONNX Runtime 实现低延迟、高质量的流式识别。
//!
//! # 架构概述
//!
//! 本模块实现了流式语音识别功能，支持多种延迟配置（160ms/480ms/960ms/1920ms）。
//! 识别流程分为三个主要步骤：
//!
//! 1. **特征提取**：调用方负责使用 [`OnlineFbankFeatureExtractor`][`crate::knf::OnlineFbankFeatureExtractor`] 提取 FBank 特征
//! 2. **编码器（Encoder）**：处理特征块，输出声学编码，维护跨块的缓存状态
//! 3. **解码器（Decoder）+ 加入器（Joiner）**：基于 transducer 架构，逐帧解码为 token
//!
//! # 主要类型
//!
//! - [`AutomaticSpeechRecognizer`]: 流式 ASR 主类型，`recognize()` 接受 FBank 特征并返回逐 token 的 `Stream`
//! - [`AutomaticSpeechRecognizerLegacy`]: 旧版非流式 ASR（通过 [`AutomaticSpeechRecognizer::new_legacy`] 构建）
//!
//! # 使用示例
//!
//! ```rust,no_run
//! use voxudio::*;
//! use futures_util::StreamExt;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // 1. 创建 ASR 识别器（以 960ms 延迟模型为例）
//! let mut asr = AutomaticSpeechRecognizer::new(
//!     "encoder-960ms.onnx",
//!     "decoder-960ms.onnx",
//!     "joiner-960ms.onnx",
//! )?;
//!
//! // 2. 提取 FBank 特征（特征提取由调用方完成）
//! let features: Vec<f32> = Default::default();  // FBank 特征，长度 = n_frames * 80
//!
//! // 3. 流式识别
//! let mut stream = asr.recognize(&features);
//! while let Some(token) = stream.next().await {
//!     print!("{}", token?);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # 特征提取配置
//!
//! 使用本模块前，需确保 FBank 特征提取参数与模型训练时一致：
//!
//! ```rust,no_run
//! use voxudio::*;
//!
//! fn main() -> anyhow::Result<()> {
//!     let mut offe = OnlineFbankFeatureExtractor::fbank()?
//!         .with_frame_opts(FrameExtractionOptions {
//!             samp_freq: 16000f32,
//!             frame_shift_ms: 10.0,
//!             frame_length_ms: 25.0,
//!             dither: 0.00003,  // 默认值，必须与训练对齐
//!             preemph_coeff: 0.97,
//!             remove_dc_offset: true,
//!             window_type: "povey",
//!             snip_edges: false,
//!             ..Default::default()
//!         })?
//!         .with_mel_opts(MelBanksOptions {
//!             num_bins: 80,
//!             low_freq: 20.0,
//!             high_freq: -400.0,
//!             vtln_low: 100.0,
//!             vtln_high: -500.0,
//!             ..Default::default()
//!         })?
//!         .build()?;
//!
//! Ok(())
//! }
//! ```

mod legacy;
mod tokens;

use {
    super::{OperationError, get_session_builder},
    async_stream::stream,
    futures_util::{Stream, stream},
    ndarray::{Array1, Array2, Array3, ArrayD, IxDyn},
    ort::{
        inputs,
        session::{RunOptions, Session, SessionInputValue},
        value::TensorRef,
    },
    std::{collections::HashMap, path::Path, pin::Pin},
    tokens::TOKENS,
};

/// SentencePiece token 解码：CJK 字符前的 `▁` 直接去掉，其他替换为空格
fn decode_sp_token(token: &str) -> String {
    if let Some(rest) = token.strip_prefix('▁') {
        if rest.chars().next().is_some_and(is_cjk) {
            rest.to_owned()
        } else {
            format!(" {}", rest)
        }
    } else {
        token.to_owned()
    }
}

/// 判断字符是否属于 CJK 范围（中日韩统一表意文字、CJK 符号与标点、全角形式等）
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{3000}'..='\u{303F}'  // CJK Symbols and Punctuation
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{4E00}'..='\u{9FFF}' // CJK Unified Ideographs
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{FF00}'..='\u{FFEF}' // Fullwidth Forms
        | '\u{20000}'..='\u{2A6DF}' // CJK Extension B
    )
}

/// 旧版非流式 ASR 识别器。
///
/// 通过此 re-export，用户可以从 `voxudio::model::asr` 直接访问
/// [`AutomaticSpeechRecognizerLegacy`] 类型，无需额外导入 `legacy` 模块。
///
/// # 使用示例
///
/// ```rust,no_run
/// use voxudio::AutomaticSpeechRecognizerLegacy;
///
/// let asr = AutomaticSpeechRecognizerLegacy::new("model.onnx")?;
/// # Ok::<(), voxudio::OperationError>(())
/// ```
///
/// # 参见
///
/// - [`AutomaticSpeechRecognizer::new_legacy`]：通过流式 ASR 类型创建 legacy 实例的便捷方法
/// - [`AutomaticSpeechRecognizerLegacy::recognize`]：执行非流式识别
pub use legacy::AutomaticSpeechRecognizerLegacy;

// ---------------------------------------------------------------------------
// CacheValue — encoder cache 张量存储
// ---------------------------------------------------------------------------

/// encoder cache 张量值，支持 f32 和 i64 两种数据类型。
enum CacheValue {
    F32(ArrayD<f32>),
    I64(Array1<i64>),
}

impl Clone for CacheValue {
    fn clone(&self) -> Self {
        match self {
            CacheValue::F32(arr) => CacheValue::F32(arr.clone()),
            CacheValue::I64(arr) => CacheValue::I64(arr.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder 元数据解析
// ---------------------------------------------------------------------------

/// 从 encoder ONNX 模型的 metadata 中解析架构配置。
fn parse_encoder_metadata(session: &Session) -> Result<HashMap<String, String>, OperationError> {
    let meta = session.metadata()?;
    let mut map = HashMap::new();
    for key in [
        "num_encoder_layers",
        "encoder_dims",
        "query_head_dims",
        "value_head_dims",
        "num_heads",
        "cnn_module_kernels",
        "left_context_len",
        "T",
        "decode_chunk_len",
    ] {
        if let Some(val) = meta.custom(key) {
            map.insert(key.to_string(), val);
        }
    }
    Ok(map)
}

/// 解析逗号分隔的整数字符串。
fn parse_int_list(s: &str) -> Vec<i32> {
    s.split(',')
        .filter_map(|x| x.trim().parse::<i32>().ok())
        .collect()
}

/// 构建 encoder cache 的 shape 列表（用于初始化零张量）。
fn build_cache_shapes(meta: &HashMap<String, String>) -> Vec<(String, Vec<i64>)> {
    let layers_per_stack = parse_int_list(&meta["num_encoder_layers"]);
    let encoder_dims = parse_int_list(&meta["encoder_dims"]);
    let query_head_dims = parse_int_list(&meta["query_head_dims"]);
    let value_head_dims = parse_int_list(&meta["value_head_dims"]);
    let num_heads = parse_int_list(&meta["num_heads"]);
    let cnn_kernels = parse_int_list(&meta["cnn_module_kernels"]);
    let left_ctx = parse_int_list(&meta["left_context_len"]);

    let mut entries = Vec::new();
    let mut layer_idx: i32 = 0;

    for (stack, &n_layers) in layers_per_stack.iter().enumerate() {
        let enc_dim = encoder_dims[stack];
        let key_dim = query_head_dims[stack] * num_heads[stack];
        let val_dim = value_head_dims[stack] * num_heads[stack];
        let ctx = left_ctx[stack];
        let cnn_half = cnn_kernels[stack] / 2;
        let nonlin_dim = 3 * enc_dim / 4;

        for _ in 0..n_layers {
            entries.push((
                format!("cached_key_{layer_idx}"),
                vec![ctx as i64, 1, key_dim as i64],
            ));
            entries.push((
                format!("cached_nonlin_attn_{layer_idx}"),
                vec![1, 1, ctx as i64, nonlin_dim as i64],
            ));
            entries.push((
                format!("cached_val1_{layer_idx}"),
                vec![ctx as i64, 1, val_dim as i64],
            ));
            entries.push((
                format!("cached_val2_{layer_idx}"),
                vec![ctx as i64, 1, val_dim as i64],
            ));
            entries.push((
                format!("cached_conv1_{layer_idx}"),
                vec![1, enc_dim as i64, cnn_half as i64],
            ));
            entries.push((
                format!("cached_conv2_{layer_idx}"),
                vec![1, enc_dim as i64, cnn_half as i64],
            ));
            layer_idx += 1;
        }
    }

    entries
}

/// 初始化 encoder caches，包括 per-layer caches、embed_states 和 processed_lens。
fn init_caches(
    cache_shapes: &[(String, Vec<i64>)],
    feature_dim: i32,
) -> HashMap<String, CacheValue> {
    let mut caches = HashMap::new();

    // Per-layer caches (f32)
    for (name, shape) in cache_shapes {
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let total: usize = shape_usize.iter().product();
        let arr = ArrayD::from_shape_vec(IxDyn(&shape_usize), vec![0.0f32; total]).unwrap();
        caches.insert(name.clone(), CacheValue::F32(arr));
    }

    // embed_states: (1, 128, 3, embed_dim), f32
    let embed_dim = ((feature_dim - 1) / 2 - 1) / 2;
    let embed_shape: Vec<usize> = vec![1, 128, 3, embed_dim as usize];
    let embed_total: usize = embed_shape.iter().product();
    let embed_arr = ArrayD::from_shape_vec(IxDyn(&embed_shape), vec![0.0f32; embed_total]).unwrap();
    caches.insert("embed_states".to_string(), CacheValue::F32(embed_arr));

    // processed_lens: (1,), i64
    let lens_arr = Array1::from_vec(vec![0i64]);
    caches.insert("processed_lens".to_string(), CacheValue::I64(lens_arr));

    caches
}

/// 构建 encoder 输入 (x + caches) 并运行 encoder session。
async fn build_and_run_encoder(
    encoder_session: &mut Session,
    run_options: &RunOptions,
    features: &Array3<f32>,
    caches: &HashMap<String, CacheValue>,
) -> Result<(Array3<f32>, HashMap<String, CacheValue>), OperationError> {
    let mut inputs: Vec<(&str, SessionInputValue)> = Vec::new();

    // "x" 输入
    let x_tensor: TensorRef<f32> = TensorRef::from_array_view(features)?;
    inputs.push(("x", SessionInputValue::View(x_tensor.into_dyn())));

    // cache 输入 — 按模型定义的输入顺序传递
    let input_names: Vec<String> = encoder_session
        .inputs()
        .iter()
        .skip(1)
        .map(|i| i.name().to_string())
        .collect();
    for name in &input_names {
        if let Some(cache) = caches.get(name) {
            match cache {
                CacheValue::F32(arr) => {
                    let tensor = TensorRef::from_array_view(arr)?;
                    inputs.push((name.as_str(), SessionInputValue::View(tensor.into_dyn())));
                }
                CacheValue::I64(arr) => {
                    let tensor = TensorRef::from_array_view(arr)?;
                    inputs.push((name.as_str(), SessionInputValue::View(tensor.into_dyn())));
                }
            }
        }
    }

    let outputs = encoder_session.run_async(inputs, run_options)?.await?;

    // 提取 encoder_out: (1, num_frames, encoder_dim)
    let (shape, data) = outputs["encoder_out"].try_extract_tensor::<f32>()?;
    let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let enc_out =
        Array3::from_shape_vec((shape_vec[0], shape_vec[1], shape_vec[2]), data.to_vec())?;

    // 提取新的 caches
    let mut new_caches = HashMap::new();
    for name in caches.keys() {
        let new_name = format!("new_{name}");
        if let Some(val) = outputs.get(&new_name) {
            let new_value = match &caches[name] {
                CacheValue::F32(old_arr) => {
                    let (s, d) = val.try_extract_tensor::<f32>()?;
                    let sv: Vec<usize> = s.iter().map(|&d| d as usize).collect();
                    CacheValue::F32(
                        ArrayD::from_shape_vec(IxDyn(&sv), d.to_vec())
                            .unwrap_or_else(|_| old_arr.clone()),
                    )
                }
                CacheValue::I64(old_arr) => {
                    if let Ok((s, d)) = val.try_extract_tensor::<i64>() {
                        let sv: Vec<usize> = s.iter().map(|&d| d as usize).collect();
                        let total: usize = sv.iter().product();
                        CacheValue::I64(Array1::from_vec(d[..total].to_vec()))
                    } else {
                        CacheValue::I64(old_arr.clone())
                    }
                }
            };
            new_caches.insert(name.clone(), new_value);
        } else {
            new_caches.insert(name.clone(), caches[name].clone());
        }
    }

    Ok((enc_out, new_caches))
}

// ---------------------------------------------------------------------------
// 流式 ASR Recognizer
// ---------------------------------------------------------------------------

//noinspection SpellCheckingInspection
/// 流式自动语音识别器。
///
/// 基于 X-ASR-zh-en 模型的 Zipformer2 transducer 架构，
/// 使用 ONNX Runtime 进行推理，支持多种Chumk配置（160ms/480ms/960ms/1920ms）。
///
/// # 功能特性
///
/// - **流式识别**：逐块处理音频特征，低延迟输出识别结果
/// - **多模型支持**：可根据应用场景选择不同Chumk的模型
/// - **Token 流式输出**：通过 [`Stream`] 逐 token 输出，支持实时显示
///
/// # 内部架构
///
/// 识别过程由三个 ONNX 模型协同完成：
///
/// 1. **Encoder（编码器）**：处理输入特征块，输出声学编码
///    - 输入：`(batch=1, frames, feature_dim=80)`
///    - 输出：`(batch=1, output_frames, encoder_dim)`
///    - 维护跨块的缓存状态（attention cache, convolution cache 等）
///
/// 2. **Decoder（解码器）**：根据已识别的 token 序列，输出语义编码
///    - 输入：`(batch=1, context_size)` 的 token ID 序列
///    - 输出：`(batch=1, decoder_dim)`
///
/// 3. **Joiner（加入器）**：融合声学编码和语义编码，输出 token logits
///    - 输入：`encoder_out` + `decoder_out`
///    - 输出：`(batch=1, vocab_size)` 的 logits
///
/// # 使用示例
///
/// ```rust,no_run
/// use voxudio::*;
/// use futures_util::StreamExt;
///
/// # async fn example() -> anyhow::Result<()> {
/// // 创建识别器（使用 960ms 模型）
/// let mut asr = AutomaticSpeechRecognizer::new(
///     "encoder-960ms.onnx",
///     "decoder-960ms.onnx",
///     "joiner-960ms.onnx",
/// )?;
///
/// // 提取特征（假设已完成）
/// let features: Vec<f32> = Default::default();  // FBank 特征
///
/// // 流式识别
/// let mut stream = asr.recognize(&features);
/// while let Some(result) = stream.next().await {
///     let token = result?;
///     print!("{}", token);
/// }
/// # Ok(())
/// # }
/// ```
///
/// # 特征要求
///
/// - 采样率：16000 Hz
/// - 特征类型：FBank（80 维）
/// - 帧长：25ms
/// - 帧移：10ms
/// - 特征提取参数必须与模型训练时完全一致（参见模块级文档）
///
/// # 性能说明
///
/// - 更高Chumk的模型（如 1920ms）通常有更好的识别性能和精度
/// - 实际延迟取决于音频长度和模型配置
///
/// FBank 特征提取由调用方负责，
/// 本类型只负责解码过程。
pub struct AutomaticSpeechRecognizer {
    encoder_session: Session,
    decoder_session: Session,
    joiner_session: Session,
    run_options: RunOptions,
    context_size: i64,

    // Encoder 配置
    t_frames: i64,
    decode_chunk_len: i64,

    // Cache 初始化信息（用于 reset）
    cache_shapes: Vec<(String, Vec<i64>)>,
}

impl AutomaticSpeechRecognizer {
    /// FBank 特征维度（bin 数量）。
    ///
    /// 通常为 80，与模型训练时的特征配置一致。
    /// 此常量用于验证输入特征的正确性，以及计算特征缓冲区大小。
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// use voxudio::*;
    /// let features = vec![0f32; AutomaticSpeechRecognizer::NUM_BINS as usize * 10 /* n_frames */];
    /// ```
    pub const NUM_BINS: i32 = AutomaticSpeechRecognizerLegacy::NUM_BINS;

    /// 创建新的流式 ASR 识别器（便捷构造函数）。
    ///
    /// 此方法是 [`Self::with_config`] 的简化版本，使用默认配置。
    /// 模型参数（如 `t_frames`、`decode_chunk_len`）将从 ONNX 模型的 metadata 中自动读取。
    ///
    /// # 参数
    ///
    /// - `encoder_path`: encoder ONNX 模型文件路径（如 `encoder-960ms.onnx`）
    /// - `decoder_path`: decoder ONNX 模型文件路径（如 `decoder-960ms.onnx`）
    /// - `joiner_path`: joiner ONNX 模型文件路径（如 `joiner-960ms.onnx`）
    ///
    /// # 返回值
    ///
    /// 成功时返回 [`AutomaticSpeechRecognizer`] 实例，失败时返回 [`OperationError`]。
    ///
    /// # 错误
    ///
    /// 可能返回以下错误：
    /// - 模型文件不存在或无法读取
    /// - ONNX 模型格式错误
    /// - 模型 metadata 缺失关键字段（`T`、`decode_chunk_len`、`context_size`）
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// use voxudio::*;
    ///
    /// let asr = AutomaticSpeechRecognizer::new(
    ///     "models/encoder-960ms.onnx",
    ///     "models/decoder-960ms.onnx",
    ///     "models/joiner-960ms.onnx",
    /// )?;
    /// # Ok::<(), voxudio::OperationError>(())
    /// ```
    ///
    /// # 参见
    ///
    /// - [`Self::with_config`]：带完整配置的构造函数
    /// - [`Self::recognize`]：执行流式识别
    pub fn new<P>(encoder_path: P, decoder_path: P, joiner_path: P) -> Result<Self, OperationError>
    where
        P: AsRef<Path>,
    {
        Self::with_config(encoder_path, decoder_path, joiner_path)
    }

    /// 带完整配置的构造函数（内部使用）。
    ///
    /// 从 ONNX 模型的 metadata 中读取模型参数，包括：
    /// - `T`：encoder 输入帧数（如 34, 192 等）
    /// - `decode_chunk_len`：每次解码的帧数（通常小于 `T`，因为有 lookahead）
    /// - `context_size`：decoder 输入的上下文大小
    ///
    /// # 参数
    ///
    /// - `encoder_path`: encoder ONNX 模型路径
    /// - `decoder_path`: decoder ONNX 模型路径
    /// - `joiner_path`: joiner ONNX 模型路径
    ///
    /// # 实现细节
    ///
    /// 1. 加载三个 ONNX 模型并创建 session
    /// 2. 从 encoder metadata 读取 `T` 和 `decode_chunk_len`
    /// 3. 从 decoder metadata 读取 `context_size`
    /// 4. 构建 cache shapes（用于初始化 encoder 的状态缓存）
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// use voxudio::*;
    ///
    /// let asr = AutomaticSpeechRecognizer::with_config(
    ///     "encoder-960ms.onnx",
    ///     "decoder-960ms.onnx",
    ///     "joiner-960ms.onnx",
    /// )?;
    /// # Ok::<(), voxudio::OperationError>(())
    /// ```
    pub fn with_config<P>(
        encoder_path: P,
        decoder_path: P,
        joiner_path: P,
    ) -> Result<Self, OperationError>
    where
        P: AsRef<Path>,
    {
        // 创建 ONNX sessions
        let encoder_session = Self::build_session(&encoder_path)?;
        let decoder_session = Self::build_session(&decoder_path)?;
        let joiner_session = Self::build_session(&joiner_path)?;
        let run_options = RunOptions::new()?;

        // 读取 encoder 元数据
        let meta = parse_encoder_metadata(&encoder_session)?;
        let t_frames = meta.get("T").and_then(|s| s.parse().ok()).unwrap_or(34);
        let decode_chunk_len = meta
            .get("decode_chunk_len")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        // 读取 decoder 元数据
        let context_size = {
            let dec_meta = decoder_session.metadata()?;
            dec_meta
                .custom("context_size")
                .and_then(|s| s.parse().ok())
                .unwrap_or(2)
        };

        // 初始化 cache shapes
        let cache_shapes = build_cache_shapes(&meta);

        Ok(Self {
            encoder_session,
            decoder_session,
            joiner_session,
            run_options,
            context_size,
            t_frames,
            decode_chunk_len,
            cache_shapes,
        })
    }

    /// 创建 legacy ASR 识别器（非流式，基于 LFR+CMVN）。
    ///
    /// 此方法提供一个便捷方式来创建旧版非流式 ASR 识别器，
    /// 适用于不需要流式输出的场景（如批量处理音频文件）。
    ///
    /// # 参数
    ///
    /// - `model_path`: 旧版 ASR ONNX 模型文件路径
    ///
    /// # 返回值
    ///
    /// 成功时返回 [`AutomaticSpeechRecognizerLegacy`] 实例。
    ///
    /// # 与非流式识别的区别
    ///
    /// - **Legacy ASR**：一次性处理整个音频，输出完整文本，无标点符号
    /// - **流式 ASR**：逐块处理，逐 token 输出，含有标点符号，适合实时场景
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// use voxudio::*;
    ///
    /// let asr_legacy = AutomaticSpeechRecognizer::new_legacy(
    ///     "models/automatic_speech_recognizer.onnx"
    /// )?;
    /// # Ok::<(), voxudio::OperationError>(())
    /// ```
    ///
    /// # 参见
    ///
    /// - [`AutomaticSpeechRecognizerLegacy`]：旧版 ASR 类型
    /// - [`AutomaticSpeechRecognizerLegacy::recognize`]：执行非流式识别
    pub fn new_legacy<P: AsRef<Path>>(
        model_path: P,
    ) -> Result<AutomaticSpeechRecognizerLegacy, OperationError> {
        AutomaticSpeechRecognizerLegacy::new(model_path)
    }

    // ---- 内部工具 ----

    fn build_session<P: AsRef<Path>>(path: P) -> Result<Session, OperationError> {
        Ok(get_session_builder()?.commit_from_file(path)?)
    }

    async fn decode_frame(
        &mut self,
        enc_frame: &Array2<f32>,
        decoder_out: &mut Option<Array2<f32>>,
        token_ids: &mut Vec<i64>,
        blank_id: i64,
        unk_id: i64,
    ) -> Result<Option<(String, i64)>, OperationError> {
        // 确保 decoder_out 已初始化
        if decoder_out.is_none() {
            let ids = token_ids.clone();
            *decoder_out = Some(self.run_decoder(&ids).await?);
        }

        let dec_out = decoder_out.clone().unwrap();
        let logits = self.run_joiner(enc_frame, &dec_out).await?;

        let pred_id = logits
            .row(0)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i64)
            .unwrap_or(0);

        // 只跳过 blank 和 unk，不跳过 sos_eos
        if pred_id != blank_id && pred_id != unk_id {
            let text = decode_sp_token(TOKENS[pred_id as usize]);
            token_ids.push(pred_id);
            let ids = token_ids.clone();
            *decoder_out = Some(self.run_decoder(&ids).await?);
            Ok(Some((text, pred_id)))
        } else {
            Ok(None)
        }
    }

    async fn run_decoder(&mut self, token_ids: &[i64]) -> Result<Array2<f32>, OperationError> {
        let ctx: Vec<i64> = if token_ids.len() >= self.context_size as usize {
            token_ids[token_ids.len() - self.context_size as usize..].to_vec()
        } else {
            let mut pad = vec![0i64; (self.context_size as usize) - token_ids.len()];
            pad.extend_from_slice(token_ids);
            pad
        };
        let ctx_arr = Array2::from_shape_vec((1, ctx.len()), ctx)?;

        let outputs = self
            .decoder_session
            .run_async(
                inputs![
                    "y" => TensorRef::from_array_view(&ctx_arr)?,
                ],
                &self.run_options,
            )?
            .await?;

        let out = outputs
            .values()
            .next()
            .ok_or_else(|| OperationError::Ort("decoder returned no outputs".to_string()))?;
        let (shape, data) = out.try_extract_tensor::<f32>()?;
        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let out_arr = Array2::from_shape_vec((shape_vec[0], shape_vec[1]), data.to_vec())?;

        Ok(out_arr)
    }

    async fn run_joiner(
        &mut self,
        enc_frame: &Array2<f32>,
        dec_out: &Array2<f32>,
    ) -> Result<Array2<f32>, OperationError> {
        let outputs = self
            .joiner_session
            .run_async(
                inputs![
                    "encoder_out" => TensorRef::from_array_view(enc_frame)?,
                    "decoder_out" => TensorRef::from_array_view(dec_out)?,
                ],
                &self.run_options,
            )?
            .await?;

        let out = outputs
            .values()
            .next()
            .ok_or_else(|| OperationError::Ort("joiner returned no outputs".to_string()))?;
        let (shape, data) = out.try_extract_tensor::<f32>()?;
        let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let out_arr = Array2::from_shape_vec((shape_vec[0], shape_vec[1]), data.to_vec())?;

        Ok(out_arr)
    }

    // ---- 公开流式接口 ----

    /// 流式识别 FBank 特征，返回逐 token 输出的 Stream。
    ///
    /// 这是本类型的核心方法，实现了流式语音识别功能。
    /// 方法接受预提取的 FBank 特征，通过 Zipformer2 transducer 架构逐帧解码，
    /// 并通过 Stream 逐个输出识别到的 token 文本。
    ///
    /// # 识别流程
    ///
    /// 1. **分块处理**：将特征按 `t_frames` 大小分块输入 encoder
    /// 2. **流式解码**：对每个 encoder 输出帧，通过 decoder + joiner 解码为 token
    /// 3. **过滤 blank/unk**：只输出有效 token（跳过 blank 和 unknown）
    /// 4. **尾部处理**：对不足一个 chunk 的剩余特征进行零填充后处理
    ///
    /// # 参数
    ///
    /// - `features`: 扁平化的 FBank 特征数组
    ///   - 长度必须为 `n_frames * NUM_BINS`（通常为 `n_frames * 80`）
    ///   - 数据类型：`f32`
    ///   - 布局：按时间顺序排列，每 `NUM_BINS` 个元素为一帧
    ///
    /// # 返回值
    ///
    /// `Pin<Box<dyn Stream<Item = Result<String, OperationError>> + Send + '_>>`
    ///
    /// - 每个 `Ok(String)` 是一个识别出的 token 文本片段
    /// - 每个 `Err(OperationError)` 是处理过程中的错误
    /// - Stream 结束时表示所有帧已处理完毕
    ///
    /// # Token 输出格式
    ///
    /// - 中文 token：直接输出中文字符（如 `"今"`、`"夜"`）
    /// - 英文 token：以空格开头（如 `" hello"`、`" world"`）
    /// - 标点：直接输出（如 `"，"`、`"。"`）
    ///
    /// # 性能特性
    ///
    /// - **首 token 延迟**：取决于模型配置（160ms/480ms/960ms/1920ms）
    /// - **吞吐量**：与音频长度和处理设备（CPU/GPU）相关
    /// - **内存占用**：与 `t_frames` 和 cache 大小相关
    ///
    /// # 错误处理
    ///
    /// 此方法本身不返回 `Result`，而是将错误通过 Stream 传递。
    /// 调用方需要通过 `while let Some(result) = stream.next().await` 来处理可能的错误。
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// use voxudio::*;
    /// use futures_util::StreamExt;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// // 1. 创建识别器
    /// let mut asr = AutomaticSpeechRecognizer::new(
    ///     "encoder-960ms.onnx",
    ///     "decoder-960ms.onnx",
    ///     "joiner-960ms.onnx",
    /// )?;
    ///
    /// // 2. 准备特征（假设已提取完成）
    /// let n_frames = 100;
    /// let features = vec![0.0f32; AutomaticSpeechRecognizer::NUM_BINS as usize * n_frames];
    ///
    /// // 3. 流式识别
    /// let mut stream = asr.recognize(&features);
    /// let mut full_text = String::new();
    ///
    /// while let Some(result) = stream.next().await {
    ///     match result {
    ///         Ok(token) => {
    ///             print!("{}", token);
    ///             full_text.push_str(&token);
    ///         }
    ///         Err(e) => eprintln!("识别错误: {}", e),
    ///     }
    /// }
    /// println!("\n完整识别结果: {}", full_text);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # 参见
    ///
    /// - [模块级文档](`crate::model::asr`)：关于特征提取配置和模型选择的详细说明
    /// - [`OnlineFbankFeatureExtractor`][`crate::knf::OnlineFbankFeatureExtractor`]：用于提取 FBank 特征
    /// - [`AutomaticSpeechRecognizerLegacy::recognize`]：非流式识别版本
    pub fn recognize<'a>(
        &'a mut self,
        features: &'a [f32],
    ) -> Pin<Box<dyn Stream<Item = Result<String, OperationError>> + Send + '_>> {
        let n_frames = features.len() / Self::NUM_BINS as usize;
        if n_frames == 0 {
            return Box::pin(stream::empty());
        }

        Box::pin(stream! {
            let t_frames = self.t_frames as usize;
            let decode_chunk_len = self.decode_chunk_len;

            let blank_id = TOKENS
                .iter()
                .enumerate()
                .find(|i| i.1 == &"<blk>")
                .map_or(0, |i| i.0 as i64);
            let unk_id = TOKENS
                .iter()
                .enumerate()
                .find(|i| i.1 == &"<unk>")
                .map_or(0, |i| i.0 as i64);

            // 逐 chunk 处理，收集所有产出的 token
            // 解码所有 encoder 输出帧（含 lookahead 帧）
            let mut num_processed: i64 = 0;
            let n_frames_i64 = n_frames as i64;

            // 内部状态
            let mut encoder_caches = init_caches(&self.cache_shapes, Self::NUM_BINS);
            let mut token_ids = Vec::new();
            let mut decoder_out = None;

            while num_processed + self.t_frames <= n_frames_i64 {
                // 取 T 帧特征 -> (1, T, feature_dim)
                let start = num_processed as usize;
                let mut chunk_features = Vec::with_capacity(t_frames * Self::NUM_BINS as usize);
                for i in 0..t_frames {
                    let offset = (start + i) * Self::NUM_BINS as usize;
                    chunk_features.extend_from_slice(&features[offset..offset + Self::NUM_BINS as usize]);
                }
                let chunk_arr =
                    Array3::from_shape_vec((1, t_frames, Self::NUM_BINS as _), chunk_features)?;
                num_processed += decode_chunk_len;

                // 运行 encoder
                let (encoder_out, new_caches) =
                    build_and_run_encoder(&mut self.encoder_session, &self.run_options, &chunk_arr, &encoder_caches).await?;
                encoder_caches = new_caches;

                let (_, num_frames, encoder_dim) = encoder_out.dim();

                // 解码所有输出帧（含 lookahead 帧）
                for t in 0..num_frames {
                    let enc_frame = encoder_out
                        .slice(ndarray::s![.., t, ..])
                        .to_owned()
                        .into_shape_with_order((1, encoder_dim))?;

                    if let Some((text, _)) = self.decode_frame(
                        &enc_frame,
                        &mut decoder_out,
                        &mut token_ids,
                        blank_id,
                        unk_id,
                    ).await? {
                        yield Ok(text);
                    }
                }
            }

            // 处理尾部不足 T 帧的剩余特征（零填充至 T 帧）
            if num_processed < n_frames_i64 {
                let avail = (n_frames_i64 - num_processed) as usize;
                let start = num_processed as usize;
                let mut padded = vec![0.0f32; t_frames * Self::NUM_BINS as usize];
                for i in 0..avail {
                    let src_offset = (start + i) * Self::NUM_BINS as usize;
                    let dst_offset = i * Self::NUM_BINS as usize;
                    padded[dst_offset..dst_offset + Self::NUM_BINS as usize]
                        .copy_from_slice(&features[src_offset..src_offset + Self::NUM_BINS as usize]);
                }
                let chunk_arr = Array3::from_shape_vec((1, t_frames, Self::NUM_BINS as _), padded)?;

                let (encoder_out, _) =
                    build_and_run_encoder(&mut self.encoder_session, &self.run_options, &chunk_arr, &encoder_caches).await?;

                let (_, num_frames, encoder_dim) = encoder_out.dim();

                // 解码所有输出帧（含 lookahead 帧）
                for t in 0..num_frames {
                    let enc_frame = encoder_out
                        .slice(ndarray::s![.., t, ..])
                        .to_owned()
                        .into_shape_with_order((1, encoder_dim))?;

                    if let Some((text, _)) = self.decode_frame(
                        &enc_frame,
                        &mut decoder_out,
                        &mut token_ids,
                        blank_id,
                        unk_id,
                    ).await? {
                        yield Ok(text);
                    }
                }
            }
        })
    }
}
