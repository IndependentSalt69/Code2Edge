#include "preprocess.h"

#include <cmath>
#include <complex>
#include <vector>

namespace code2edge {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Reflection equivalent to PyTorch-style reflect padding:
// index ... 3 2 1 0 1 2 3 ...
// therefore -1 -> 1, -2 -> 2, etc.
std::size_t reflect_index(
    long index,
    std::size_t length
) {
    if (length < 2) {
        return 0;
    }

    const long period =
        static_cast<long>(2 * (length - 1));

    long x = index % period;

    if (x < 0) {
        x += period;
    }

    if (x >= static_cast<long>(length)) {
        x = period - x;
    }

    return static_cast<std::size_t>(x);
}

float periodic_hann(std::size_t n) {
    return 0.5f - 0.5f * std::cos(
        2.0f * kPi *
        static_cast<float>(n) /
        static_cast<float>(kFftSize)
    );
}

}  // namespace


bool compute_power_spectrogram(
    const float* input,
    std::size_t input_length,
    float* output,
    std::size_t output_length
) {
    if (input == nullptr || output == nullptr) {
        return false;
    }

    if (input_length != kInputSamples) {
        return false;
    }

    if (output_length != kFreqBins * kFrames) {
        return false;
    }

    // ---------------------------------------------------------
    // PyTorch/Torchaudio center=True:
    //
    // n_fft = 400
    // therefore pad 200 samples on each side.
    //
    // pad_mode = reflect
    // ---------------------------------------------------------
    constexpr std::size_t kPad = kFftSize / 2;

    std::vector<float> padded(
        input_length + 2 * kPad
    );

    for (std::size_t i = 0; i < padded.size(); ++i) {
        const long source =
            static_cast<long>(i) -
            static_cast<long>(kPad);

        padded[i] =
            input[reflect_index(source, input_length)];
    }

    // ---------------------------------------------------------
    // One frame at a time.
    //
    // Output layout:
    //   [frequency_bin][frame]
    //
    // This matches the C-order layout of
    // torchaudio's [201, 101] spectrogram.
    // ---------------------------------------------------------

    for (std::size_t frame = 0; frame < kFrames; ++frame) {
        const std::size_t start =
            frame * kHopLength;

        float windowed[kFftSize];

        for (std::size_t n = 0; n < kFftSize; ++n) {
            windowed[n] =
                padded[start + n] *
                periodic_hann(n);
        }

        // -----------------------------------------------------
        // Direct 400-point DFT.
        //
        // Python reference:
        //   onesided=True
        //   power=2.0
        //   normalized=False
        //
        // DFT:
        //   X[k] = sum_n x[n] exp(-j 2 pi k n / N)
        // -----------------------------------------------------

        for (std::size_t k = 0; k < kFreqBins; ++k) {
            float real = 0.0f;
            float imag = 0.0f;

            for (std::size_t n = 0; n < kFftSize; ++n) {
                const float angle =
                    -2.0f * kPi *
                    static_cast<float>(k * n) /
                    static_cast<float>(kFftSize);

                const float c = std::cos(angle);
                const float s = std::sin(angle);

                real += windowed[n] * c;
                imag += windowed[n] * s;
            }

            const float power =
                real * real + imag * imag;

            output[k * kFrames + frame] = power;
        }
    }

    return true;
}

}  // namespace code2edge