#include <iostream>
#include <stdexcept>
#include <cstring>
#include "knf.h"

namespace knf {
    std::string create_string(const char *text) {
        return std::string(text);
    }

    void free_result(float* ptr) {
        delete[] ptr;
    }

    float *fbank_extract(const FbankComputer::Options *opts_ptr, int sampling_rate, const float *waveform, int32_t len, int32_t *out_len) {
        try {
            if (opts_ptr == nullptr) {
                *out_len = 0;
                return new float[0];
            }

            if (sampling_rate <= 0 || sampling_rate > 192000) {
                *out_len = 0;
                return new float[0];
            }

            if (len < 0) {
                *out_len = 0;
                return new float[0];
            }

            if (waveform == nullptr && len > 0) {
                *out_len = 0;
                return new float[0];
            }

            if (opts_ptr->mel_opts.num_bins <= 0 || opts_ptr->mel_opts.num_bins > 256) {
                *out_len = 0;
                return new float[0];
            }

            // 创建一个新的 Options 对象，从指针获取数值，但重新初始化字符串
            FbankComputer::Options opts;
            opts.frame_opts.samp_freq = opts_ptr->frame_opts.samp_freq;
            opts.frame_opts.frame_shift_ms = opts_ptr->frame_opts.frame_shift_ms;
            opts.frame_opts.frame_length_ms = opts_ptr->frame_opts.frame_length_ms;
            opts.frame_opts.dither = opts_ptr->frame_opts.dither;
            opts.frame_opts.preemph_coeff = opts_ptr->frame_opts.preemph_coeff;
            opts.frame_opts.remove_dc_offset = opts_ptr->frame_opts.remove_dc_offset;
            opts.frame_opts.window_type = std::string("povey");  // 使用默认值而不是传入的值
            opts.frame_opts.round_to_power_of_two = opts_ptr->frame_opts.round_to_power_of_two;
            opts.frame_opts.blackman_coeff = opts_ptr->frame_opts.blackman_coeff;
            opts.frame_opts.snip_edges = opts_ptr->frame_opts.snip_edges;

            opts.mel_opts = opts_ptr->mel_opts;
            opts.energy_floor = opts_ptr->energy_floor;
            opts.htk_compat = opts_ptr->htk_compat;
            opts.raw_energy = opts_ptr->raw_energy;
            opts.use_energy = opts_ptr->use_energy;
            opts.use_log_fbank = opts_ptr->use_log_fbank;
            opts.use_power = opts_ptr->use_power;

            OnlineFbank fbank(opts);

            if (len > 0) {
                fbank.AcceptWaveform(sampling_rate, (float*)waveform, len);
            }

            fbank.InputFinished();
            int32_t feature_dim = opts.mel_opts.num_bins;
            int32_t n = fbank.NumFramesReady();

            *out_len = n * feature_dim;
            float *features = new float[*out_len];
            float *p = features;

            for (int32_t i = 0; i != n; ++i) {
                const float *f = fbank.GetFrame(i);
                std::copy(f, f + feature_dim, p);
                p += feature_dim;
            }

            return features;
        } catch (const std::exception &e) {
            *out_len = 0;
            return new float[0];
        } catch (...) {
            *out_len = 0;
            return new float[0];
        }
    }

    float *mfcc_extract(const MfccComputer::Options *opts, int sampling_rate, const float *waveform, int32_t len, int32_t *out_len) {
        try {
            OnlineMfcc mfcc(*opts);
            mfcc.AcceptWaveform(sampling_rate, (float*)waveform, len);
            mfcc.InputFinished();
            int32_t feature_dim = opts->num_ceps;
            int32_t n = mfcc.NumFramesReady();

            *out_len = n * feature_dim;
            float *features = new float[*out_len];
            float *p = features;

            for (int32_t i = 0; i != n; ++i) {
                const float *f = mfcc.GetFrame(i);
                std::copy(f, f + feature_dim, p);
                p += feature_dim;
            }

            return features;
        } catch (const std::exception &e) {
            std::cerr << "Exception in mfcc_extract: " << e.what() << std::endl;
            *out_len = 0;
            return new float[0];
        } catch (...) {
            std::cerr << "Unknown exception in mfcc_extract" << std::endl;
            *out_len = 0;
            return new float[0];
        }
    }

    float *whisper_fbank_extract(const WhisperFeatureComputer::Options *opts, int sampling_rate, const float *waveform, int32_t len, int32_t *out_len) {
        try {
            OnlineWhisperFbank whisper_fbank(*opts);
            whisper_fbank.AcceptWaveform(sampling_rate, (float*)waveform, len);
            whisper_fbank.InputFinished();
            int32_t feature_dim = opts->dim;
            int32_t n = whisper_fbank.NumFramesReady();

            *out_len = n * feature_dim;
            float *features = new float[*out_len];
            float *p = features;

            for (int32_t i = 0; i != n; ++i) {
                const float *f = whisper_fbank.GetFrame(i);
                std::copy(f, f + feature_dim, p);
                p += feature_dim;
            }

            return features;
        } catch (const std::exception &e) {
            std::cerr << "Exception in whisper_fbank_extract: " << e.what() << std::endl;
            *out_len = 0;
            return new float[0];
        } catch (...) {
            std::cerr << "Unknown exception in whisper_fbank_extract" << std::endl;
            *out_len = 0;
            return new float[0];
        }
    }
}