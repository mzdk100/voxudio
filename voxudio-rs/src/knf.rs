mod raw;

use {
    crate::OperationError,
    std::{ffi::CString, slice::from_raw_parts},
};

fn create_string(text: &str) -> Result<raw::std::string, OperationError> {
    Ok(unsafe { raw::knf::create_string(CString::new(text)?.into_raw()) })
}

pub type OnlineMfccFeatureExtractor = OnlineFbankFeatureExtractor<raw::knf::MfccOptions>;
//noinspection SpellCheckingInspection
pub type OnlineWhisperFbankFeatureExtractor =
    OnlineFbankFeatureExtractor<raw::knf::WhisperFeatureOptions>;

//noinspection SpellCheckingInspection
pub struct FrameExtractionOptions<'a> {
    /// in milliseconds.
    pub samp_freq: f32,
    /// in milliseconds.
    pub frame_shift_ms: f32,
    pub frame_length_ms: f32,
    /// Amount of dithering, 0.0 means no dither.
    /// Value 0.00003f is equivalent to 1.0 in kaldi.
    pub dither: f32,
    /// Preemphasis coefficient.
    pub preemph_coeff: f32,
    /// Subtract mean of wave before FFT.
    pub remove_dc_offset: bool,
    /// e.g. Hamming window
    /// May be "hamming", "rectangular", "povey", "hanning", "hann", "sine", "blackman".
    /// "povey" is a window I made to be similar to Hamming but to go to zero at the edges, it's pow((0.5 - 0.5*cos(n/N*2*pi)), 0.85) I just don't think the
    /// Hamming window makes sense as a windowing function.
    pub window_type: &'a str,
    pub round_to_power_of_two: bool,
    pub blackman_coeff: f32,
    pub snip_edges: bool,
}

impl<'a> Default for FrameExtractionOptions<'a> {
    //noinspection SpellCheckingInspection
    fn default() -> Self {
        Self {
            samp_freq: 16000f32,
            frame_shift_ms: 10f32,
            frame_length_ms: 25f32,
            dither: 0.00003,
            preemph_coeff: 0.97,
            remove_dc_offset: true,
            window_type: "povey",
            round_to_power_of_two: true,
            blackman_coeff: 0.42,
            snip_edges: true,
        }
    }
}

impl<'a> FrameExtractionOptions<'a> {
    fn into_raw(self) -> Result<raw::knf::FrameExtractionOptions, OperationError> {
        Ok(raw::knf::FrameExtractionOptions {
            blackman_coeff: self.blackman_coeff,
            dither: self.dither,
            frame_length_ms: self.frame_length_ms,
            frame_shift_ms: self.frame_shift_ms,
            preemph_coeff: self.preemph_coeff,
            remove_dc_offset: self.remove_dc_offset,
            round_to_power_of_two: self.round_to_power_of_two,
            samp_freq: self.samp_freq,
            snip_edges: self.snip_edges,
            window_type: create_string(self.window_type)?,
        })
    }
}

//noinspection SpellCheckingInspection
//noinspection GrazieInspection
pub struct MelBanksOptions<'a> {
    /// e.g. 25; number of triangular bins
    pub num_bins: i32,
    /// e.g. 20; lower frequency cutoff
    pub low_freq: f32,
    /// an upper frequency cutoff; 0 -> no cutoff, negative
    /// ->added to the Nyquist frequency to get the cutoff.
    pub high_freq: f32,
    /// vtln lower cutoff of warping function.
    pub vtln_low: f32,
    /// vtln upper cutoff of warping function: if negative, added
    /// to the Nyquist frequency to get the cutoff.
    pub vtln_high: f32,
    pub debug_mel: bool,
    /// htk_mode is a "hidden" config, it does not show up on command line.
    /// Enables more exact compatibility with HTK, for testing purposes.  Affects mel-energy flooring and reproduces a bug in HTK.
    pub htk_mode: bool,
    /// Note that if you set is_librosa, you probably need to set low_freq to 0.
    /// Please see <https://librosa.org/doc/main/generated/librosa.filters.mel.html>
    pub is_librosa: bool,
    /// used only when is_librosa=true
    /// Possible values: "", slaney. We don't support a numeric value here, but it can be added on demand.
    /// See <https://librosa.org/doc/main/generated/librosa.filters.mel.html>
    pub norm: &'a str,
    /// used only when is_librosa is true
    pub use_slaney_mel_scale: bool,
    /// used only when is_librosa is true
    pub floor_to_int_bin: bool,
}

impl<'a> Default for MelBanksOptions<'a> {
    fn default() -> Self {
        Self {
            num_bins: 25,
            low_freq: 20f32,
            high_freq: 0f32,
            vtln_low: 100f32,
            vtln_high: -500f32,
            debug_mel: false,
            htk_mode: false,
            is_librosa: false,
            norm: "slaney",
            use_slaney_mel_scale: true,
            floor_to_int_bin: false,
        }
    }
}

impl<'a> MelBanksOptions<'a> {
    fn into_raw(self) -> Result<raw::knf::MelBanksOptions, OperationError> {
        Ok(raw::knf::MelBanksOptions {
            debug_mel: self.debug_mel,
            floor_to_int_bin: self.floor_to_int_bin,
            high_freq: self.high_freq,
            htk_mode: self.htk_mode,
            is_librosa: self.is_librosa,
            low_freq: self.low_freq,
            norm: create_string(self.norm)?,
            num_bins: self.num_bins,
            use_slaney_mel_scale: self.use_slaney_mel_scale,
            vtln_high: self.vtln_high,
            vtln_low: self.vtln_low,
        })
    }
}

//noinspection SpellCheckingInspection
pub struct OnlineFbankFeatureExtractorBuilder {
    opts: raw::knf::FbankOptions,
}

//noinspection SpellCheckingInspection
impl OnlineFbankFeatureExtractorBuilder {
    //noinspection SpellCheckingInspection
    fn new() -> Result<Self, OperationError> {
        let frame_opts = FrameExtractionOptions::default();
        let mel_opts = MelBanksOptions::default();
        let opts = raw::knf::FbankComputer_Options {
            frame_opts: frame_opts.into_raw()?,
            mel_opts: mel_opts.into_raw()?,
            energy_floor: 0f32, // active iff use_energy==true
            // If true, put energy last (if using energy)
            // If false, put energy first
            htk_compat: false, // active iff use_energy==true
            // If true, compute log_energy before preemphasis and windowing
            // If false, compute log_energy after preemphasis and windowing
            raw_energy: true, // active iff use_energy==true
            // append an extra dimension with energy to the filter banks
            use_energy: false,
            // if true (default), produce log-filterbank, else linear
            use_log_fbank: true,
            // if true (default), use power in filterbank
            // analysis, else magnitude.
            use_power: true,
        };

        Ok(Self { opts })
    }

    /// 设置帧参数。
    pub fn with_frame_opts(
        &mut self,
        frame_opts: FrameExtractionOptions,
    ) -> Result<&mut Self, OperationError> {
        self.opts.frame_opts = frame_opts.into_raw()?;
        Ok(self)
    }

    /// 设置梅尔滤波器组参数。
    pub fn with_mel_opts(
        &mut self,
        mel_opts: MelBanksOptions,
    ) -> Result<&mut Self, OperationError> {
        self.opts.mel_opts = mel_opts.into_raw()?;
        Ok(self)
    }

    /// 设置能量下限（仅在 use_energy=true 时生效）。
    pub fn with_energy_floor(&mut self, energy_floor: f32) -> &mut Self {
        self.opts.energy_floor = energy_floor;
        self
    }

    /// 设置 HTK 兼容模式（仅在 use_energy=true 时生效）。
    pub fn with_htk_compat(&mut self, htk_compat: bool) -> &mut Self {
        self.opts.htk_compat = htk_compat;
        self
    }

    /// 设置是否在预加重和加窗前计算能量（仅在 use_energy=true 时生效）。
    pub fn with_raw_energy(&mut self, raw_energy: bool) -> &mut Self {
        self.opts.raw_energy = raw_energy;
        self
    }

    /// 设置是否在滤波器组后附加能量维度。
    pub fn with_use_energy(&mut self, use_energy: bool) -> &mut Self {
        self.opts.use_energy = use_energy;
        self
    }

    /// 设置是��输出对数滤波器组（true）或线性滤波器组（false）。
    pub fn with_use_log_fbank(&mut self, use_log_fbank: bool) -> &mut Self {
        self.opts.use_log_fbank = use_log_fbank;
        self
    }

    /// 设置滤波器组分析时是否使用功率（true）或幅值（false）。
    pub fn with_use_power(&mut self, use_power: bool) -> &mut Self {
        self.opts.use_power = use_power;
        self
    }

    pub fn build(&self) -> Result<OnlineFbankFeatureExtractor, OperationError> {
        Ok(OnlineFbankFeatureExtractor::from(self.opts))
    }
}

pub struct OnlineMfccFeatureExtractorBuilder {
    opts: raw::knf::MfccOptions,
}

//noinspection SpellCheckingInspection
impl OnlineMfccFeatureExtractorBuilder {
    //noinspection GrazieInspection
    fn new() -> Result<Self, OperationError> {
        let frame_opts = FrameExtractionOptions::default();
        let mel_opts = MelBanksOptions::default();
        let opts = raw::knf::MfccOptions {
            frame_opts: frame_opts.into_raw()?,
            mel_opts: mel_opts.into_raw()?,
            // Constant that controls scaling of MFCCs
            cepstral_lifter: 22.,
            // Floor on energy (absolute, not relative) in MFCC computation. Only makes a difference if use_energy=true;
            // only necessary if dither=0.0.
            // Suggested values: 0.1 or 1.0
            energy_floor: 0.,
            // If true, put energy or C0 last and use a factor of
            // sqrt(2) on C0.
            // Warning: not sufficient to get HTK compatible features
            // (need to change other parameters)
            htk_compat: false,
            // Number of cepstra in MFCC computation (including C0)
            num_ceps: 13,
            // If true, compute energy before preemphasis and windowing
            raw_energy: true,
            // Use energy (not C0) in MFCC computation
            use_energy: true,
        };

        Ok(Self { opts })
    }

    /// 设置帧参数。
    pub fn with_frame_opts(
        &mut self,
        frame_opts: FrameExtractionOptions,
    ) -> Result<&mut Self, OperationError> {
        self.opts.frame_opts = frame_opts.into_raw()?;
        Ok(self)
    }

    /// 设置梅尔滤波器组参数。
    pub fn with_mel_opts(
        &mut self,
        mel_opts: MelBanksOptions,
    ) -> Result<&mut Self, OperationError> {
        self.opts.mel_opts = mel_opts.into_raw()?;
        Ok(self)
    }

    /// 设置倒谱提升系数（控制 MFCC 缩放）。
    pub fn with_cepstral_lifter(&mut self, cepstral_lifter: f32) -> &mut Self {
        self.opts.cepstral_lifter = cepstral_lifter;
        self
    }

    /// ���置能量下限（仅在 use_energy=true 时生效）。
    pub fn with_energy_floor(&mut self, energy_floor: f32) -> &mut Self {
        self.opts.energy_floor = energy_floor;
        self
    }

    /// 设置 HTK 兼容模式。
    pub fn with_htk_compat(&mut self, htk_compat: bool) -> &mut Self {
        self.opts.htk_compat = htk_compat;
        self
    }

    /// 设置倒谱系数数量（包括 C0）。
    pub fn with_num_ceps(&mut self, num_ceps: i32) -> &mut Self {
        self.opts.num_ceps = num_ceps;
        self
    }

    /// 设置是否在预加重和加窗前计算能量。
    pub fn with_raw_energy(&mut self, raw_energy: bool) -> &mut Self {
        self.opts.raw_energy = raw_energy;
        self
    }

    /// 设置 MFCC 计算是否使用能量（而非 C0）。
    pub fn with_use_energy(&mut self, use_energy: bool) -> &mut Self {
        self.opts.use_energy = use_energy;
        self
    }

    pub fn build(&self) -> Result<OnlineMfccFeatureExtractor, OperationError> {
        Ok(OnlineFbankFeatureExtractor::from(self.opts))
    }
}

//noinspection SpellCheckingInspection
pub struct OnlineWhisperFbankFeatureExtractorBuilder {
    opts: raw::knf::WhisperFeatureOptions,
}

impl OnlineWhisperFbankFeatureExtractorBuilder {
    fn new() -> Result<Self, OperationError> {
        let frame_opts = FrameExtractionOptions::default();
        let opts = raw::knf::WhisperFeatureOptions {
            frame_opts: frame_opts.into_raw()?,
            dim: 80,
        };

        Ok(Self { opts })
    }

    /// 设置帧参数。
    pub fn with_frame_opts(
        &mut self,
        frame_opts: FrameExtractionOptions,
    ) -> Result<&mut Self, OperationError> {
        self.opts.frame_opts = frame_opts.into_raw()?;
        Ok(self)
    }

    /// 设置每一帧输出维度（如 80）。
    pub fn with_dim(&mut self, dim: i32) -> &mut Self {
        self.opts.dim = dim;
        self
    }

    pub fn build(&self) -> Result<OnlineWhisperFbankFeatureExtractor, OperationError> {
        Ok(OnlineFbankFeatureExtractor::from(self.opts))
    }
}

//noinspection SpellCheckingInspection
pub trait FbankOptions {
    fn run<const SR: usize>(&self, input: &[f32], len: &mut i32) -> *mut f32;
}

impl FbankOptions for raw::knf::FbankOptions {
    fn run<const SR: usize>(&self, input: &[f32], len: &mut i32) -> *mut f32 {
        unsafe {
            raw::knf::fbank_extract(self as *const _, SR as _, input.as_ptr() as _, input.len() as _, len)
        }
    }
}

impl FbankOptions for raw::knf::MfccOptions {
    fn run<const SR: usize>(&self, input: &[f32], len: &mut i32) -> *mut f32 {
        unsafe {
            raw::knf::mfcc_extract(self as *const _, SR as _, input.as_ptr() as _, input.len() as _, len)
        }
    }
}

impl FbankOptions for raw::knf::WhisperFeatureOptions {
    fn run<const SR: usize>(&self, input: &[f32], len: &mut i32) -> *mut f32 {
        unsafe {
            raw::knf::whisper_fbank_extract(
                self as *const _,
                SR as _,
                input.as_ptr() as _,
                input.len() as _,
                len,
            )
        }
    }
}

//noinspection SpellCheckingInspection
/// 在线滤波器组特征提取器。
///
/// 用于从音频信号中提取滤波器组（FBank）、MFCC 或 Whisper FBank 特征。
/// 支持自定义采样率、帧参数、梅尔参数等。
///
/// # 示例
/// ```rust
/// use voxudio::OnlineFbankFeatureExtractor;
/// fn main() -> anyhow::Result<()> {
/// // 构建 FBank 特征提取器
/// let extractor = OnlineFbankFeatureExtractor::fbank()?
///     .with_frame_opts(Default::default())?
///     .with_energy_floor(1.0)
///     .build()?;
/// // 提取特征
/// let audio = (0..1600).map(|i| f32::sin(i as _)).collect::<Vec<_>>();
/// let features = extractor.extract::<16000>(&audio);
/// Ok(())
/// }
/// ```
///
/// 也可通过 `mfcc()` 或 `whisper_fbank()` 构建其他类型特征提取器。
pub struct OnlineFbankFeatureExtractor<O = raw::knf::FbankOptions> {
    opts: O,
}

impl<O> From<O> for OnlineFbankFeatureExtractor<O>
where
    O: FbankOptions,
{
    fn from(value: O) -> Self {
        Self { opts: value }
    }
}

//noinspection SpellCheckingInspection
impl OnlineFbankFeatureExtractor {
    /// 构建 FBank 特征提取器的 builder。
    /// 可通过链式调用设置参数，最后调用 build() 得到特征提取器。
    pub fn fbank() -> Result<OnlineFbankFeatureExtractorBuilder, OperationError> {
        OnlineFbankFeatureExtractorBuilder::new()
    }

    /// 构建 MFCC 特征提取器的 builder。
    /// 可通过链式调用设置参数，最后调用 build() 得到特征提取器。
    pub fn mfcc() -> Result<OnlineMfccFeatureExtractorBuilder, OperationError> {
        OnlineMfccFeatureExtractorBuilder::new()
    }

    /// 构建 Whisper FBank 特征提取器的 builder。
    /// 可通过链式调用设置参数，最后调用 build() 得到特征提取器。
    pub fn whisper_fbank() -> Result<OnlineWhisperFbankFeatureExtractorBuilder, OperationError> {
        OnlineWhisperFbankFeatureExtractorBuilder::new()
    }

    /// 构建默认 FBank 特征提取器。
    /// 等价于 `Self::fbank()?.build()`。
    pub fn new() -> Result<OnlineFbankFeatureExtractor, OperationError> {
        Self::fbank()?.build()
    }
}

//noinspection SpellCheckingInspection
impl<O> OnlineFbankFeatureExtractor<O> {
    /// 提取音频特征。
    ///
    /// # 参数
    /// * `SR` - 采样率（常用如 16000）。
    /// * `audio` - 音频数据（f32 数组，-1.0到1.0之间）。
    ///
    /// # 返回
    /// 特征向量（Vec<f32>）。
    ///
    /// # 示例
    /// ```rust
    /// use voxudio::OnlineFbankFeatureExtractor;
    /// fn main() -> anyhow::Result<()> {
    /// let extractor = OnlineFbankFeatureExtractor::new()?;
    /// let audio = (0..1600).map(|i| f32::sin(i as _)).collect::<Vec<_>>();
    /// let features = extractor.extract::<16000>(&audio);
    /// Ok(())
    /// }
    /// ```
    pub fn extract<const SR: usize>(&self, audio: &[f32]) -> Vec<f32>
    where
        O: FbankOptions,
    {
        let audio = audio.iter().map(|i| i * 32768f32).collect::<Vec<_>>();
        let mut ret = 0;
        let ptr = self.opts.run::<SR>(&audio, &mut ret);
        unsafe {
            let res = from_raw_parts(ptr, ret as _).to_owned();
            raw::knf::free_result(ptr);
            res
        }
    }
}
