#include "preprocess.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr
            << "Usage: parity_runner "
            << "<input_f32.bin> "
            << "<output_f32.bin>\n";
        return 2;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];

    std::ifstream input_file(
        input_path,
        std::ios::binary
    );

    if (!input_file) {
        std::cerr << "Cannot open input: "
                  << input_path << "\n";
        return 1;
    }

    std::vector<float> input(
        code2edge::kInputSamples
    );

    const std::streamsize expected_bytes =
        static_cast<std::streamsize>(
            input.size() * sizeof(float)
        );

    input_file.read(
        reinterpret_cast<char*>(input.data()),
        expected_bytes
    );

    if (input_file.gcount() != expected_bytes) {
        std::cerr
            << "Input file has wrong size: expected "
            << expected_bytes
            << " bytes, got "
            << input_file.gcount()
            << " bytes\n";
        return 1;
    }

    std::vector<float> output(
        code2edge::kFreqBins *
        code2edge::kFrames
    );

    if (!code2edge::compute_power_spectrogram(
            input.data(),
            input.size(),
            output.data(),
            output.size())) {
        std::cerr << "Power spectrogram failed\n";
        return 1;
    }

    std::ofstream output_file(
        output_path,
        std::ios::binary
    );

    if (!output_file) {
        std::cerr << "Cannot open output: "
                  << output_path << "\n";
        return 1;
    }

    output_file.write(
        reinterpret_cast<const char*>(
            output.data()
        ),
        static_cast<std::streamsize>(
            output.size() * sizeof(float)
        )
    );

    std::cout
        << "wrote "
        << output.size()
        << " float32 values\n";

    return 0;
}