use std::env::set_var;
use std::{env::var, path::Path};

// 收集所有源文件名到数组中
//noinspection SpellCheckingInspection
const SOURCE_FILES_OPUS: [&str; 128] = [
    // 基本文件
    "opus.c",
    "opus_decoder.c",
    "mathops.c",
    "dec_API.c",
    "tables_other.c",
    "decode_pulses.c",
    "tables_pulses_per_block.c",
    "shell_coder.c",
    "code_signs.c",
    "decode_indices.c",
    "tables_LTP.c",
    "tables_pitch_lag.c",
    "tables_gain.c",
    "NLSF_unpack.c",
    "decode_frame.c",
    "CNG.c",
    "NLSF2A.c",
    "table_LSF_cos.c",
    "LPC_fit.c",
    "LPC_inv_pred_gain.c",
    "bwexpander_32.c",
    "decode_core.c",
    "LPC_analysis_filter.c",
    "decode_parameters.c",
    "NLSF_decode.c",
    "NLSF_stabilize.c",
    "sort.c",
    "gain_quant.c",
    "log2lin.c",
    "lin2log.c",
    "decode_pitch.c",
    "pitch_est_tables.c",
    "bwexpander.c",
    "PLC.c",
    "sum_sqr_shift.c",
    "decoder_set_fs.c",
    "tables_NLSF_CB_NB_MB.c",
    "tables_NLSF_CB_WB.c",
    "resampler.c",
    "resampler_rom.c",
    "resampler_private_up2_HQ.c",
    "resampler_private_down_FIR.c",
    "resampler_private_AR2.c",
    "resampler_private_IIR_FIR.c",
    "init_decoder.c",
    "stereo_decode_pred.c",
    "stereo_MS_to_LR.c",
    "celt_decoder.c",
    "celt.c",
    "vq.c",
    "cwrs.c",
    "entenc.c",
    "celt_lpc.c",
    "pitch.c",
    "quant_bands.c",
    "laplace.c",
    "entdec.c",
    "bands.c",
    "rate.c",
    "entcode.c",
    "modes.c",
    "mdct.c",
    "kiss_fft.c",
    "opus_encoder.c",
    "analysis.c",
    "mlp_data.c",
    "mlp.c",
    "repacketizer.c",
    "extensions.c",
    "enc_API.c",
    "control_codec.c",
    "control_audio_bandwidth.c",
    "init_encoder.c",
    "VAD.c",
    "sigm_Q15.c",
    "ana_filt_bank_1.c",
    "process_NLSFs.c",
    "NLSF_encode.c",
    "NLSF_del_dec_quant.c",
    "NLSF_VQ.c",
    "interpolate.c",
    "NLSF_VQ_weights_laroia.c",
    "NSQ_del_dec.c",
    "NSQ.c",
    "quant_LTP_gains.c",
    "VQ_WMat_EC.c",
    "A2NLSF.c",
    "LP_variable_cutoff.c",
    "biquad_alt.c",
    "HP_variable_cutoff.c",
    "encode_indices.c",
    "encode_pulses.c",
    "control_SNR.c",
    "check_control_input.c",
    "stereo_encode_pred.c",
    "stereo_LR_to_MS.c",
    "stereo_quant_pred.c",
    "stereo_find_predictor.c",
    "inner_prod_aligned.c",
    "celt_encoder.c",
    "resampler_down2_3.c",
    "resampler_down2.c",
    // float 子目录文件
    "float/encode_frame_FLP.c",
    "float/wrappers_FLP.c",
    "float/process_gains_FLP.c",
    "float/find_pred_coefs_FLP.c",
    "float/residual_energy_FLP.c",
    "float/LPC_analysis_filter_FLP.c",
    "float/energy_FLP.c",
    "float/LTP_analysis_filter_FLP.c",
    "float/find_LTP_FLP.c",
    "float/corrMatrix_FLP.c",
    "float/inner_product_FLP.c",
    "float/scale_vector_FLP.c",
    "float/find_LPC_FLP.c",
    "float/burg_modified_FLP.c",
    "float/LTP_scale_ctrl_FLP.c",
    "float/scale_copy_vector_FLP.c",
    "float/find_pitch_lags_FLP.c",
    "float/apply_sine_window_FLP.c",
    "float/pitch_analysis_core_FLP.c",
    "float/sort_FLP.c",
    "float/autocorrelation_FLP.c",
    "float/k2a_FLP.c",
    "float/schur_FLP.c",
    "float/bwexpander_FLP.c",
    "float/noise_shape_analysis_FLP.c",
    "float/warped_autocorrelation_FLP.c",
];
const SOURCE_FILES_KNF: [&str; 12] = [
    "feature-fbank.cc",
    "feature-window.cc",
    "mel-computations.cc",
    "rfft.cc",
    "feature-functions.cc",
    "kaldi-math.cc",
    "kiss_fftr.c",
    "kiss_fft.c",
    "online-feature.cc",
    "feature-mfcc.cc",
    "whisper-feature.cc",
    "knf.cc",
];

//noinspection SpellCheckingInspection
fn compile_opus() {
    let out_dir = var("OUT_DIR").unwrap();
    if bindgen::builder()
        .header("src/opus/opus.h")
        .generate()
        .unwrap()
        .write_to_file(Path::new(&out_dir).join("opus.rs"))
        .is_ok()
    {
        println!("cargo:rerun-if-changed=src/opus");
    }

    // 定义源文件基础路径
    let src_path = Path::new("src/opus");

    // 创建构建配置
    let mut build = cc::Build::new();

    // 添加包含目录
    build.include("src/opus").include("src/opus/float");

    // 添加所有源文件
    for file in &SOURCE_FILES_OPUS {
        build.file(src_path.join(file));
    }

    // 添加定义并编译
    build
        .define("OPUS_BUILD", None)
        .define("NONTHREADSAFE_PSEUDOSTACK", None)
        .define("ENABLE_ASSERTIONS", None)
        .compile("opus")
}

fn compile_knf() {
    let out_dir = var("OUT_DIR").unwrap();

    // 定义源文件基础路径
    let src_path = Path::new("src/knf");

    // 创建构建配置
    let mut build = cc::Build::new();

    // 添加所有源文件
    for file in &SOURCE_FILES_KNF {
        build.file(src_path.join(file));
    }

    // 编译
    build.compile("knf");

    if var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        let compiler = build.get_compiler();
        let compiler_path = compiler.path();
        unsafe {
            set_var("CLANG_PATH", compiler_path);
        }
    }

    if bindgen::builder()
        .clang_arg("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
        .clang_args(["-x", "c++"])
        .header("src/knf/knf.h")
        .allowlist_item("knf::.*")
        .opaque_type("std::.*")
        .enable_cxx_namespaces()
        .generate_comments(true)
        .wrap_unsafe_ops(true)
        .generate()
        .unwrap()
        .write_to_file(Path::new(&out_dir).join("knf.rs"))
        .is_ok()
    {
        println!("cargo:rerun-if-changed=src/knf");
        // 强制重新运行构建脚本
        println!("cargo:rerun-if-changed=build.rs");
    }
}

fn main() {
    let features = var("CARGO_CFG_FEATURE").unwrap();
    if features.contains("opus") {
        compile_opus();
    }
    if features.contains("knf") {
        compile_knf();
    }
}
