#include <string>
#include <vector>
#include "online-feature.h"

namespace knf {
    std::string create_string(const char *text);
    void free_result(float* ptr);
    float *fbank_extract(const FbankComputer::Options *opts, int sampling_rate, const float *waveform, int32_t len, int32_t *out_len);
    float *mfcc_extract(const MfccComputer::Options *opts, int sampling_rate, const float *waveform, int32_t len, int32_t *out_len);
    float *whisper_fbank_extract(const WhisperFeatureComputer::Options *opts, int sampling_rate, const float *waveform, int32_t len, int32_t *out_len);

}