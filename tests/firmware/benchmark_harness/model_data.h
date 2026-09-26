#ifndef CODE2EDGE_MODEL_DATA_H
#define CODE2EDGE_MODEL_DATA_H

#include <stdint.h>

/*
 * Code2Edge frozen INT8 TFLite model.
 * SHA-256: 0cd6cefbcba738c13d028ffd9a8ad73d6c88f0046368de974d3aa38c47926876
 * Size: 172216 bytes
 * Input: int8[1,1,64,101], scale=0.018517991527915, zero_point=-51
 * Output: int8[1,12], scale=0.049993276596069336, zero_point=3
 * Weight quantization: symmetric INT8, channel-wise
 * Calibration: 500-sample frozen golden corpus, true min/max QSV aggregation
 */

#if defined(__cplusplus)
extern "C" {
#endif

extern const unsigned char g_model_data[];
extern const int g_model_data_len;

#if defined(__cplusplus)
}
#endif

#endif /* CODE2EDGE_MODEL_DATA_H */
