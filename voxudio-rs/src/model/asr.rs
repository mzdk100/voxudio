mod tokens;

use {
    super::{super::OperationError, get_session_builder},
    ndarray::{Array1, Array3},
    ort::{
        inputs,
        session::{RunOptions, Session},
        value::TensorRef,
    },
    std::path::Path,
    tokens::TOKENS,
};

/// 自动语音识别器（ASR），用于将音频特征转换为文本。
///
/// # 示例
/// ```rust
/// use voxudio::*;
/// // 加载模型
/// let asr = AutomaticSpeechRecognizer::new("../checkpoint/automatic_speech_recognizer.onnx").unwrap();
/// ```
pub struct AutomaticSpeechRecognizer {
    model: Session,
}

//noinspection SpellCheckingInspection
impl AutomaticSpeechRecognizer {
    /// FBank特征维度，通常为80。
    pub const NUM_BINS: i32 = 80;

    /// 创建一个新的自动语音识别器实例。
    ///
    /// # 参数
    /// - `model_path`: ONNX模型文件路径
    ///
    /// # 返回值
    /// - `Result<Self, OperationError>`: 成功返回识别器实例，失败返回错误
    ///
    /// # 示例
    /// ```rust
    /// use voxudio::*;
    /// let asr = AutomaticSpeechRecognizer::new("../checkpoint/automatic_speech_recognizer.onnx").unwrap();
    /// ```
    pub fn new<P>(model_path: P) -> Result<Self, OperationError>
    where
        P: AsRef<Path>,
    {
        let model = get_session_builder()?.commit_from_file(model_path)?;
        Ok(Self { model })
    }

    /// 识别输入的FBank特征，返回识别文本。
    ///
    /// # 参数
    /// - `features`: FBank特征数组（如通过 OnlineFbankFeatureExtractor 提取）
    ///
    /// # 返回值
    /// - `Result<String, OperationError>`: 成功返回识别文本，失败返回错误
    ///
    /// # 示例
    /// ```rust
    /// use voxudio::*;
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    /// let mut asr = AutomaticSpeechRecognizer::new("../checkpoint/automatic_speech_recognizer.onnx")?;
    /// let features = vec![0.0; 80*10]; // 假设已提取特征
    /// let text = asr.recognize(&features).await.unwrap();
    /// println!("{}", text);
    ///
    /// Ok(())
    /// # }
    /// ```
    pub async fn recognize(&mut self, features: &[f32]) -> Result<String, OperationError> {
        let meta_data = self.model.metadata()?;
        let lfr_window_size = meta_data
            .custom("lfr_window_size")
            .map_or(Ok(0), |s| s.parse())
            .unwrap_or_default();
        let lfr_window_shift = meta_data
            .custom("lfr_window_shift")
            .map_or(Ok(0), |s| s.parse())
            .unwrap_or_default();
        let neg_mean = meta_data
            .custom("neg_mean")
            .unwrap_or_default()
            .split(',')
            .filter_map(|i| i.trim().parse::<f32>().ok())
            .collect::<Vec<_>>();
        let inv_stddev = meta_data
            .custom("inv_stddev")
            .unwrap_or_default()
            .split(',')
            .filter_map(|i| i.trim().parse::<f32>().ok())
            .collect::<Vec<_>>();
        drop(meta_data);
        let feat_dim = (Self::NUM_BINS * lfr_window_size) as usize;

        // 1. Apply LFR
        // 2. Apply CMVN
        //
        // Please refer to
        // https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45555.pdf
        // for what LFR means
        //
        // "Lower Frame Rate Neural Network Acoustic Models"
        let mut features = Self::apply_lfr(&features, lfr_window_size, lfr_window_shift);
        Self::apply_cmvn(&mut features, &neg_mean, &inv_stddev);
        let num_frames = features.len() / feat_dim;
        let x = Array3::from_shape_vec((1, num_frames, feat_dim), features)?;
        let x_length = Array1::from_elem(1, num_frames as i32);

        let options = RunOptions::new()?;
        let outputs = self
            .model
            .run_async(
                inputs![
                    "speech" => TensorRef::from_array_view(&x)?,
                    "speech_lengths" => TensorRef::from_array_view(&x_length)?,
                ],
                &options,
            )?
            .await?;

        let (shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        let (num_tokens, vocab_size) = (shape[1] as usize, shape[2] as usize);
        let eos_id = TOKENS
            .iter()
            .enumerate()
            .find(|i| i.1 == &"</s>")
            .map_or(0, |i| i.0);

        let mut res = String::with_capacity(num_tokens * std::mem::size_of::<char>());
        let mut mergeable = false;
        let mut last_sym: Option<&str> = None;
        for k in 0..num_tokens {
            let max_idx = logits[(k * vocab_size)..(k * vocab_size + vocab_size)]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map_or(eos_id, |i| i.0);
            if max_idx == eos_id {
                break;
            }

            let mut sym = TOKENS[max_idx];
            if !sym.ends_with("@@") {
                let p = sym.as_bytes()[0];
                if p < 0x80 {
                    // ascii
                    if mergeable {
                        mergeable = false;
                        res.push_str(sym);
                    } else {
                        res.push(' ');
                        res.push_str(sym);
                    }
                } else {
                    // 非ascii
                    mergeable = false;
                    if let Some(last_sym) = last_sym {
                        let prev_p = last_sym.as_bytes()[0];
                        if prev_p < 0x80 {
                            res.push(' ');
                        }
                    }
                    res.push_str(sym);
                }
            } else {
                // 以@@结尾
                sym = &sym[..sym.len() - 2];
                if mergeable {
                    res.push_str(sym);
                } else {
                    res.push(' ');
                    res.push_str(sym);
                    mergeable = true;
                }
            }

            last_sym = Some(sym);
        }

        Ok(res)
    }

    fn apply_lfr(input: &[f32], lfr_window_size: i32, lfr_window_shift: i32) -> Vec<f32> {
        let in_num_frames = input.len() / Self::NUM_BINS as usize;
        let out_num_frames = (in_num_frames as i32 - lfr_window_size) / lfr_window_shift + 1;
        let out_feat_dim = Self::NUM_BINS * lfr_window_size;

        let mut output = vec![0.0; (out_num_frames * out_feat_dim) as usize];

        let mut in_offset = 0;
        let mut out_offset = 0;

        for _ in 0..out_num_frames {
            output[out_offset..out_offset + out_feat_dim as usize]
                .copy_from_slice(&input[in_offset..in_offset + out_feat_dim as usize]);
            out_offset += out_feat_dim as usize;
            in_offset += (lfr_window_shift * Self::NUM_BINS) as usize;
        }

        output
    }

    fn apply_cmvn(v: &mut [f32], neg_mean: &[f32], inv_stddev: &[f32]) {
        let dim = neg_mean.len();
        let num_frames = v.len() / dim;

        for i in 0..num_frames {
            let offset = i * dim;
            for k in 0..dim {
                v[offset + k] = (v[offset + k] + neg_mean[k]) * inv_stddev[k];
            }
        }
    }
}
