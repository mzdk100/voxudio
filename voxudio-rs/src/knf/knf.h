#include <string>
#include <vector>
#include "online-feature.h"

namespace knf {
    std::string create_string(const char *text);
    void free_result(float* ptr);
    float *fbank_extract(FbankComputer::Options opts, int sampling_rate, float *waveform, int32_t len, int32_t *out_len);
    float *mfcc_extract(MfccComputer::Options opts, int sampling_rate, float *waveform, int32_t len, int32_t *out_len);
    float *whisper_fbank_extract(WhisperFeatureComputer::Options opts, int sampling_rate, float *waveform, int32_t len, int32_t *out_len);

}