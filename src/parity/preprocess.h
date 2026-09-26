#pragma once

#include <cstddef>

namespace code2edge {

constexpr std::size_t kInputSamples = 16000;
constexpr std::size_t kFftSize = 400;
constexpr std::size_t kHopLength = 160;
constexpr std::size_t kFreqBins = 201;
constexpr std::size_t kFrames = 101;

bool compute_power_spectrogram(
    const float* input,
    std::size_t input_length,
    float* output,
    std::size_t output_length
);

}  // namespace code2edge