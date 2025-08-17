#include <iostream>
#include "knf.h"

namespace knf {
    std::string create_string(const char *text) {
        return std::string(text);
    }

    void free_result(float* ptr) {
        delete[] ptr;
    }

    float *fbank_extract(FbankComputer::Options opts, int sampling_rate, float *waveform, int32_t len, int32_t *out_len) {
        OnlineFbank fbank(opts);
        fbank.AcceptWaveform(sampling_rate, waveform, len);
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
    }

    float *mfcc_extract(MfccComputer::Options opts, int sampling_rate, float *waveform, int32_t len, int32_t *out_len) {
        OnlineMfcc mfcc(opts);
        mfcc.AcceptWaveform(sampling_rate, waveform, len);
        mfcc.InputFinished();
        int32_t feature_dim = opts.num_ceps;
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
    }

    float *whisper_fbank_extract(WhisperFeatureComputer::Options opts, int sampling_rate, float *waveform, int32_t len, int32_t *out_len) {
        OnlineWhisperFbank whisper_fbank(opts);
        whisper_fbank.AcceptWaveform(sampling_rate, waveform, len);
        whisper_fbank.InputFinished();
        int32_t feature_dim = opts.dim;
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
    }
}