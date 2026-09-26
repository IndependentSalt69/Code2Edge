/*
 * Code2Edge: STM32U585 DS-CNN Model Runner Implementation
 *
 * Target:    Arduino UNO Q (STMicroelectronics STM32U585 ARM Cortex-M33 @ 160 MHz)
 * Core:      arduino:zephyr (1.0.0)
 *
 * Implements full INT8 quantized inference for the frozen tiny-kws DS-CNN model (119k params).
 * Zero heap memory allocation (no malloc/free).
 * Static tensor arena footprint: 166,560 bytes (~162.6 KB).
 * Hardware cycle timing via Cortex-M33 DWT_CYCCNT.
 */

#include "model_runner.h"
#include "model_data.h"
#include <Arduino.h>
#include <string.h>
#include <math.h>

// ==============================================================================
// 1. Model Labels (Contract: 12 classes)
// ==============================================================================

const char* const kModelLabels[MODEL_NUM_CLASSES] = {
    "silence",
    "unknown",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go"
};

// ==============================================================================
// 2. DWT Cycle Counter Register Access
// ==============================================================================

#define DWT_CYCCNT_REG (*((volatile uint32_t*)0xE0001004))

static inline uint32_t get_dwt_cycles(void) {
    return DWT_CYCCNT_REG;
}

// ==============================================================================
// 3. Static Tensor Arena Allocation (.bss, zero dynamic allocation)
// ==============================================================================

// Activation dimensions & SRAM footprint:
// 1. Stem Ring Buffer: 3 rows x (51 x 160)= 24,480 bytes (streamed into Block 0 DW)
// 2. Activation Buf A: 1 x 17 x 26 x 160  = 70,720 bytes (also temporarily holds 7,770 B padded input)
// 3. Activation Buf B: 1 x 17 x 26 x 160  = 70,720 bytes
// 4. Sum Temp Buffer:  160 x int32        = 640 bytes
// Total static SRAM footprint:            = 166,560 bytes (~162.6 KB)

#define ARENA_PAD_IN_SIZE     (74 * 105)      // 7,770 bytes
#define STEM_ROW_SIZE         (51 * 160)      // 8,160 bytes
#define ARENA_STEM_RING_ROWS  3               // 24,480 bytes
#define ARENA_BUF_A_SIZE      (17 * 26 * 160) // 70,720 bytes
#define ARENA_BUF_B_SIZE      (17 * 26 * 160) // 70,720 bytes

alignas(16) static int8_t s_arena_stem_ring[ARENA_STEM_RING_ROWS * STEM_ROW_SIZE];
alignas(16) static int8_t s_arena_buf_a[ARENA_BUF_A_SIZE];
alignas(16) static int8_t s_arena_buf_b[ARENA_BUF_B_SIZE];
alignas(16) static int32_t s_arena_sum[160];

// Structure to hold pointers to Flash-resident model parameters
struct ModelWeights {
    const int8_t *stem_w;      // [1, 10, 4, 160] (6,400 B)
    const int32_t *stem_b;     // [160] (640 B)

    const int8_t *b0_dw_w;     // [1, 3, 3, 160] (1,440 B)
    const int32_t *b0_dw_b;    // [160] (640 B)
    const int8_t *b0_pw_w;     // [160, 1, 1, 160] (25,600 B)
    const int32_t *b0_pw_b;    // [160] (640 B)

    const int8_t *b1_dw_w;     // [1, 3, 3, 160] (1,440 B)
    const int32_t *b1_dw_b;    // [160] (640 B)
    const int8_t *b1_pw_w;     // [160, 1, 1, 160] (25,600 B)
    const int32_t *b1_pw_b;    // [160] (640 B)

    const int8_t *b2_dw_w;     // [1, 3, 3, 160] (1,440 B)
    const int32_t *b2_dw_b;    // [160] (640 B)
    const int8_t *b2_pw_w;     // [160, 1, 1, 160] (25,600 B)
    const int32_t *b2_pw_b;    // [160] (640 B)

    const int8_t *b3_dw_w;     // [1, 3, 3, 160] (1,440 B)
    const int32_t *b3_dw_b;    // [160] (640 B)
    const int8_t *b3_pw_w;     // [160, 1, 1, 160] (25,600 B)
    const int32_t *b3_pw_b;    // [160] (640 B)

    const int8_t *fc_w;        // [12, 160] (1,920 B)
    const int32_t *fc_b;       // [12] (48 B)
};

static struct ModelWeights s_weights;
static bool s_model_initialized = false;

// Per-channel effective scaling factors (computed from model quantization scales)
static float s_M_stem[160];
static float s_M_b0_dw[160];
static float s_M_b0_pw[160];
static float s_M_b1_dw[160];
static float s_M_b1_pw[160];
static float s_M_b2_dw[160];
static float s_M_b2_pw[160];
static float s_M_b3_dw[160];
static float s_M_b3_pw[160];
static float s_M_sum;
static float s_M_fc[12];

// ==============================================================================
// 4. FlatBuffer Pointer Resolution
// ==============================================================================

static inline uint32_t read_u32(const uint8_t *data, uint32_t pos) {
    return (uint32_t)data[pos] | ((uint32_t)data[pos + 1] << 8) |
           ((uint32_t)data[pos + 2] << 16) | ((uint32_t)data[pos + 3] << 24);
}

static inline int32_t read_i32(const uint8_t *data, uint32_t pos) {
    return (int32_t)read_u32(data, pos);
}

static inline uint16_t read_u16(const uint8_t *data, uint32_t pos) {
    return (uint16_t)data[pos] | ((uint16_t)data[pos + 1] << 8);
}

static uint32_t fb_get_field(const uint8_t *data, uint32_t table_pos, uint32_t field_idx) {
    int32_t vtable_delta = read_i32(data, table_pos);
    uint32_t vtable_pos = table_pos - vtable_delta;
    uint16_t vtable_size = read_u16(data, vtable_pos);
    uint32_t offset_idx = 4 + 2 * field_idx;
    if (offset_idx + 2 > vtable_size) return 0;
    uint16_t field_offset = read_u16(data, vtable_pos + offset_idx);
    if (field_offset == 0) return 0;
    return table_pos + field_offset;
}

static uint32_t fb_get_subtable(const uint8_t *data, uint32_t table_pos, uint32_t field_idx) {
    uint32_t field_pos = fb_get_field(data, table_pos, field_idx);
    if (field_pos == 0) return 0;
    uint32_t offset = read_u32(data, field_pos);
    return field_pos + offset;
}

static const uint8_t* fb_get_buffer_data(const uint8_t *data, uint32_t buffer_idx, uint32_t *len_out) {
    uint32_t root_pos = read_u32(data, 0);
    uint32_t bufs_field = fb_get_field(data, root_pos, 4);
    if (bufs_field == 0) return NULL;
    uint32_t bufs_vec = bufs_field + read_u32(data, bufs_field);
    uint32_t num_bufs = read_u32(data, bufs_vec);
    if (buffer_idx >= num_bufs) return NULL;

    uint32_t buf_table_pos = (bufs_vec + 4 + 4 * buffer_idx) + read_u32(data, bufs_vec + 4 + 4 * buffer_idx);
    uint32_t data_field = fb_get_field(data, buf_table_pos, 0);
    if (data_field == 0) {
        if (len_out) *len_out = 0;
        return NULL;
    }
    uint32_t data_vec = data_field + read_u32(data, data_field);
    uint32_t data_len = read_u32(data, data_vec);
    if (len_out) *len_out = data_len;
    return data + data_vec + 4;
}

static void fb_read_float_vector(const uint8_t *data, uint32_t field_pos, float *dest, uint32_t max_count) {
    if (field_pos == 0) return;
    uint32_t vec_pos = field_pos + read_u32(data, field_pos);
    uint32_t len = read_u32(data, vec_pos);
    uint32_t count = (len < max_count) ? len : max_count;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t u = read_u32(data, vec_pos + 4 + i * 4);
        memcpy(&dest[i], &u, sizeof(float));
    }
}

// ==============================================================================
// 5. Model Initialization & Weight Resolution
// ==============================================================================

bool model_runner_init(void) {
    if (s_model_initialized) {
        return true;
    }

    const uint8_t *data = g_model_data;
    uint32_t len = (uint32_t)g_model_data_len;

    if (len < 8 || data[4] != 'T' || data[5] != 'F' || data[6] != 'L' || data[7] != '3') {
        return false;
    }

    // Map Flash pointers to constant weight and bias buffers directly (Zero copy)
    uint32_t blen = 0;
    s_weights.fc_w    = (const int8_t*)  fb_get_buffer_data(data, 1,  &blen); // [12, 160]
    s_weights.fc_b    = (const int32_t*) fb_get_buffer_data(data, 2,  &blen); // [12]
    s_weights.b3_pw_b = (const int32_t*) fb_get_buffer_data(data, 3,  &blen); // [160]
    s_weights.b3_dw_b = (const int32_t*) fb_get_buffer_data(data, 4,  &blen); // [160]
    s_weights.b2_pw_b = (const int32_t*) fb_get_buffer_data(data, 5,  &blen); // [160]
    s_weights.b2_dw_b = (const int32_t*) fb_get_buffer_data(data, 6,  &blen); // [160]
    s_weights.b1_pw_b = (const int32_t*) fb_get_buffer_data(data, 7,  &blen); // [160]
    s_weights.b1_dw_b = (const int32_t*) fb_get_buffer_data(data, 8,  &blen); // [160]
    s_weights.b0_pw_b = (const int32_t*) fb_get_buffer_data(data, 9,  &blen); // [160]
    s_weights.b0_dw_b = (const int32_t*) fb_get_buffer_data(data, 10, &blen); // [160]
    s_weights.stem_b  = (const int32_t*) fb_get_buffer_data(data, 11, &blen); // [160]

    s_weights.b3_pw_w = (const int8_t*)  fb_get_buffer_data(data, 12, &blen); // [160, 1, 1, 160]
    s_weights.b3_dw_w = (const int8_t*)  fb_get_buffer_data(data, 13, &blen); // [1, 3, 3, 160]
    s_weights.b2_pw_w = (const int8_t*)  fb_get_buffer_data(data, 14, &blen); // [160, 1, 1, 160]
    s_weights.b2_dw_w = (const int8_t*)  fb_get_buffer_data(data, 15, &blen); // [1, 3, 3, 160]
    s_weights.b1_pw_w = (const int8_t*)  fb_get_buffer_data(data, 16, &blen); // [160, 1, 1, 160]
    s_weights.b1_dw_w = (const int8_t*)  fb_get_buffer_data(data, 17, &blen); // [1, 3, 3, 160]
    s_weights.b0_pw_w = (const int8_t*)  fb_get_buffer_data(data, 18, &blen); // [160, 1, 1, 160]
    s_weights.b0_dw_w = (const int8_t*)  fb_get_buffer_data(data, 19, &blen); // [1, 3, 3, 160]
    s_weights.stem_w  = (const int8_t*)  fb_get_buffer_data(data, 20, &blen); // [1, 10, 4, 160]

    if (!s_weights.fc_w || !s_weights.stem_w || !s_weights.b0_dw_w) {
        return false;
    }

    // Extract per-channel scales for all layers from SubGraph 0 tensors table
    uint32_t root_pos = read_u32(data, 0);
    uint32_t sgs_field = fb_get_field(data, root_pos, 2);
    uint32_t sgs_vec = sgs_field + read_u32(data, sgs_field);
    uint32_t sg0_pos = (sgs_vec + 4) + read_u32(data, sgs_vec + 4);
    uint32_t tensors_field = fb_get_field(data, sg0_pos, 0);
    uint32_t tensors_vec = tensors_field + read_u32(data, tensors_field);

    auto get_tensor_scale = [&](uint32_t t_idx, float *out_scales, uint32_t count) {
        uint32_t t_pos = (tensors_vec + 4 + 4 * t_idx) + read_u32(data, tensors_vec + 4 + 4 * t_idx);
        uint32_t q_pos = fb_get_subtable(data, t_pos, 4);
        if (q_pos) {
            uint32_t scale_field = fb_get_field(data, q_pos, 2);
            fb_read_float_vector(data, scale_field, out_scales, count);
        }
    };

    const float s_in  = MODEL_INPUT_SCALE;
    const float s_t26 = 0.05369709059596062f;
    const float s_t27 = 0.06842003017663956f;
    const float s_t28 = 0.06879869103431702f;
    const float s_t29 = 0.08805365115404129f;
    const float s_t30 = 0.06626978516578674f;
    const float s_t31 = 0.07592766731977463f;
    const float s_t32 = 0.059932198375463486f;
    const float s_t33 = 0.20041866600513458f;
    const float s_t34 = 0.1043703556060791f;
    const float s_t35 = 2.475351333618164f;
    const float s_out = MODEL_OUTPUT_SCALE;

    float sw_buf[160];

    // Stem Conv Multipliers (T20 weight -> T26 out)
    get_tensor_scale(20, sw_buf, 160);
    for (int c = 0; c < 160; ++c) s_M_stem[c] = (s_in * sw_buf[c]) / s_t26;

    // Block 0 DW (T19 -> T27)
    get_tensor_scale(19, sw_buf, 160);
    for (int c = 0; c < 160; ++c) s_M_b0_dw[c] = (s_t26 * sw_buf[c]) / s_t27;

    // Block 0 PW (T18 -> T28)
    get_tensor_scale(18, sw_buf, 160);
    for (int c = 0; c < 160; ++c) s_M_b0_pw[c] = (s_t27 * sw_buf[c]) / s_t28;

    // Block 1 DW (T17 -> T29)
    get_tensor_scale(17, sw_buf, 160);
    for (int c = 0; c < 160; ++c) s_M_b1_dw[c] = (s_t28 * sw_buf[c]) / s_t29;

    // Block 1 PW (T16 -> T30)
    get_tensor_scale(16, sw_buf, 160);
    for (int c = 0; c < 160; ++c) s_M_b1_pw[c] = (s_t29 * sw_buf[c]) / s_t30;

    // Block 2 DW (T15 -> T31)
    get_tensor_scale(15, sw_buf, 160);
    for (int c = 0; c < 160; ++c) s_M_b2_dw[c] = (s_t30 * sw_buf[c]) / s_t31;

    // Block 2 PW (T14 -> T32)
    get_tensor_scale(14, sw_buf, 160);
    for (int c = 0; c < 160; ++c) s_M_b2_pw[c] = (s_t31 * sw_buf[c]) / s_t32;

    // Block 3 DW (T13 -> T33)
    get_tensor_scale(13, sw_buf, 160);
    for (int c = 0; c < 160; ++c) s_M_b3_dw[c] = (s_t32 * sw_buf[c]) / s_t33;

    // Block 3 PW (T12 -> T34)
    get_tensor_scale(12, sw_buf, 160);
    for (int c = 0; c < 160; ++c) s_M_b3_pw[c] = (s_t33 * sw_buf[c]) / s_t34;

    // Sum reduction
    s_M_sum = s_t34 / s_t35;

    // FC Multipliers (T1 -> T36)
    get_tensor_scale(1, sw_buf, 12);
    for (int c = 0; c < 12; ++c) s_M_fc[c] = (s_t35 * sw_buf[c]) / s_out;

    s_model_initialized = true;
    return true;
}

// ==============================================================================
// 6. Preprocessing Input Quantization
// ==============================================================================

void model_runner_quantize_input(const float *features_f32, int8_t *features_int8) {
    const float inv_scale = 1.0f / MODEL_INPUT_SCALE;
    const int32_t zp = MODEL_INPUT_ZERO_POINT; // -51

    for (size_t i = 0; i < MODEL_INPUT_SIZE; ++i) {
        int32_t q = (int32_t)roundf(features_f32[i] * inv_scale) + zp;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        features_int8[i] = (int8_t)q;
    }
}

// ==============================================================================
// 7. Core Neural Network Layer Kernels
// ==============================================================================

static inline int8_t clamp_relu_int8(int32_t acc, float M, int32_t zp_out) {
    int32_t val = (int32_t)roundf((float)acc * M) + zp_out;
    if (val < zp_out) val = zp_out; // ReLU lower bound (-128)
    if (val > 127)    val = 127;    // Upper INT8 bound
    return (int8_t)val;
}

/**
 * Compute 1 row of Stem 2D Convolution:
 * Input: [74, 105, 1] -> Output row: [51, 160] (8,160 bytes)
 */
static void compute_stem_conv_row(const int8_t *in_pad, int h, int8_t *out_row) {
    const int32_t in_zp = MODEL_INPUT_ZERO_POINT; // -51
    const int32_t out_zp = -128;
    const int8_t *w = s_weights.stem_w;
    const int32_t *b = s_weights.stem_b;

    for (int w_pos = 0; w_pos < 51; ++w_pos) {
        int32_t acc[160];
        memcpy(acc, b, sizeof(acc));

        for (int kh = 0; kh < 10; ++kh) {
            int in_row = (h * 2 + kh) * 105;
            for (int kw = 0; kw < 4; ++kw) {
                int32_t px = (int32_t)in_pad[in_row + (w_pos * 2 + kw)] - in_zp;
                const int8_t *w_ptr = w + (kh * 4 + kw) * 160;

                for (int c = 0; c < 160; ++c) {
                    acc[c] += px * (int32_t)w_ptr[c];
                }
            }
        }

        int8_t *out_ptr = out_row + w_pos * 160;
        for (int c = 0; c < 160; ++c) {
            out_ptr[c] = clamp_relu_int8(acc[c], s_M_stem[c], out_zp);
        }
    }
}

/**
 * Compute 1 row of Block 0 Depthwise Convolution (3x3 Kernel, Stride 2, Pad 1):
 * Inputs: 3 Stem rows (minus1, 0, plus1).
 * Output: 1 row in Buffer B: [26, 160] (4,160 bytes).
 */
static void compute_block0_dw_row(
    const int8_t *r_minus1, const int8_t *r_0, const int8_t *r_plus1,
    int8_t *out_row
) {
    const int32_t in_zp = -128;
    const int32_t out_zp = -128;
    const int8_t *w = s_weights.b0_dw_w;
    const int32_t *b = s_weights.b0_dw_b;
    const float *M = s_M_b0_dw;

    const int8_t *row_ptrs[3] = { r_minus1, r_0, r_plus1 };

    for (int w_pos = 0; w_pos < 26; ++w_pos) {
        int in_w_base = w_pos * 2 - 1; // Pad 1 offset with stride 2
        int32_t acc[160];
        memcpy(acc, b, sizeof(acc));

        for (int kh = 0; kh < 3; ++kh) {
            const int8_t *cur_row = row_ptrs[kh];
            if (!cur_row) continue; // Padded row -> (pad_val - in_zp) = 0

            for (int kw = 0; kw < 3; ++kw) {
                int c_col = in_w_base + kw;
                if (c_col < 0 || c_col >= 51) continue; // Padded column

                const int8_t *in_ptr = cur_row + c_col * 160;
                const int8_t *w_ptr = w + (kh * 3 + kw) * 160;

                for (int c = 0; c < 160; ++c) {
                    acc[c] += ((int32_t)in_ptr[c] - in_zp) * (int32_t)w_ptr[c];
                }
            }
        }

        int8_t *out_ptr = out_row + w_pos * 160;
        for (int c = 0; c < 160; ++c) {
            out_ptr[c] = clamp_relu_int8(acc[c], M[c], out_zp);
        }
    }
}

/**
 * Depthwise 2D Convolution (3x3 Kernel, Stride 1, SAME padding = pad 1):
 * Input/Output: [17, 26, 160] (70,720 bytes).
 */
static void run_depthwise_stride1(
    const int8_t *in_act,
    const int8_t *w, const int32_t *b, const float *M,
    int8_t *out_act
) {
    const int32_t in_zp = -128;
    const int32_t out_zp = -128;

    for (int h = 0; h < 17; ++h) {
        int in_h_base = h - 1;
        for (int w_pos = 0; w_pos < 26; ++w_pos) {
            int in_w_base = w_pos - 1;
            int32_t acc[160];
            memcpy(acc, b, sizeof(acc));

            for (int kh = 0; kh < 3; ++kh) {
                int r = in_h_base + kh;
                if (r < 0 || r >= 17) continue;

                for (int kw = 0; kw < 3; ++kw) {
                    int c_col = in_w_base + kw;
                    if (c_col < 0 || c_col >= 26) continue;

                    const int8_t *in_ptr = in_act + (r * 26 + c_col) * 160;
                    const int8_t *w_ptr = w + (kh * 3 + kw) * 160;

                    for (int c = 0; c < 160; ++c) {
                        acc[c] += ((int32_t)in_ptr[c] - in_zp) * (int32_t)w_ptr[c];
                    }
                }
            }

            int8_t *out_ptr = out_act + (h * 26 + w_pos) * 160;
            for (int c = 0; c < 160; ++c) {
                out_ptr[c] = clamp_relu_int8(acc[c], M[c], out_zp);
            }
        }
    }
}

/**
 * Pointwise 2D Convolution (1x1 Kernel, 160 in -> 160 out):
 * Spatial size: 17 * 26 = 442.
 */
static void run_pointwise_1x1(
    const int8_t *in_act,
    const int8_t *w, const int32_t *b, const float *M,
    int8_t *out_act
) {
    const int32_t in_zp = -128;
    const int32_t out_zp = -128;

    for (int p = 0; p < 17 * 26; ++p) {
        const int8_t *in_ptr = in_act + p * 160;
        int8_t *out_ptr = out_act + p * 160;

        for (int c_out = 0; c_out < 160; ++c_out) {
            int32_t acc = b[c_out];
            const int8_t *w_row = w + c_out * 160;

            for (int c_in = 0; c_in < 160; ++c_in) {
                acc += ((int32_t)in_ptr[c_in] - in_zp) * (int32_t)w_row[c_in];
            }
            out_ptr[c_out] = clamp_relu_int8(acc, M[c_out], out_zp);
        }
    }
}

// ==============================================================================
// 8. End-to-End Model Forward Pass
// ==============================================================================

bool model_runner_invoke(const int8_t *input_int8, struct ModelInferenceResult *result_out) {
    if (!s_model_initialized) {
        if (!model_runner_init()) {
            return false;
        }
    }

    uint32_t t_start = get_dwt_cycles();

    // 1. Layer 0 & 1: Reshape [1, 1, 64, 101] and Pad to [1, 74, 105, 1] with -51 (in_zp)
    // Alias temporary pad buffer in s_arena_buf_a (7,770 bytes of 70,720 bytes)
    int8_t *pad_in = s_arena_buf_a;
    memset(pad_in, (int8_t)(-51), ARENA_PAD_IN_SIZE);
    for (int h = 0; h < 64; ++h) {
        int dst_offset = (h + 5) * 105 + 2; // Top pad = 5, Left pad = 2
        int src_offset = h * 101;
        memcpy(pad_in + dst_offset, input_int8 + src_offset, 101);
    }

    // 2. Streamed Stem Conv + Block 0 DW using 3-row rolling ring buffer
    // Stem Conv produces 33 rows [0..32]; Block 0 DW produces 17 rows [0..16].
    int next_stem_row_to_compute = 0;

    for (int h = 0; h < 17; ++h) {
        int r_need_a = 2 * h - 1;
        int r_need_b = 2 * h;
        int r_need_c = 2 * h + 1;

        int max_r_need = (r_need_c < 33) ? r_need_c : 32;
        while (next_stem_row_to_compute <= max_r_need) {
            int8_t *dest_ring_slot = s_arena_stem_ring + (next_stem_row_to_compute % ARENA_STEM_RING_ROWS) * STEM_ROW_SIZE;
            compute_stem_conv_row(pad_in, next_stem_row_to_compute, dest_ring_slot);
            next_stem_row_to_compute++;
        }

        const int8_t *ptr_a = (r_need_a >= 0) ? (s_arena_stem_ring + (r_need_a % ARENA_STEM_RING_ROWS) * STEM_ROW_SIZE) : NULL;
        const int8_t *ptr_b = s_arena_stem_ring + (r_need_b % ARENA_STEM_RING_ROWS) * STEM_ROW_SIZE;
        const int8_t *ptr_c = (r_need_c < 33) ? (s_arena_stem_ring + (r_need_c % ARENA_STEM_RING_ROWS) * STEM_ROW_SIZE) : NULL;

        int8_t *b0_dw_out_row = s_arena_buf_b + h * (26 * 160);
        compute_block0_dw_row(ptr_a, ptr_b, ptr_c, b0_dw_out_row);
    }

    // 3. Block 0 Pointwise Conv: Buffer B [17, 26, 160] -> Buffer A [17, 26, 160]
    run_pointwise_1x1(s_arena_buf_b, s_weights.b0_pw_w, s_weights.b0_pw_b, s_M_b0_pw, s_arena_buf_a);

    // 4. Block 1 (DW stride 1: A -> B; PW: B -> A)
    run_depthwise_stride1(s_arena_buf_a, s_weights.b1_dw_w, s_weights.b1_dw_b, s_M_b1_dw, s_arena_buf_b);
    run_pointwise_1x1(s_arena_buf_b, s_weights.b1_pw_w, s_weights.b1_pw_b, s_M_b1_pw, s_arena_buf_a);

    // 5. Block 2 (DW stride 1: A -> B; PW: B -> A)
    run_depthwise_stride1(s_arena_buf_a, s_weights.b2_dw_w, s_weights.b2_dw_b, s_M_b2_dw, s_arena_buf_b);
    run_pointwise_1x1(s_arena_buf_b, s_weights.b2_pw_w, s_weights.b2_pw_b, s_M_b2_pw, s_arena_buf_a);

    // 6. Block 3 (DW stride 1: A -> B; PW: B -> A)
    run_depthwise_stride1(s_arena_buf_a, s_weights.b3_dw_w, s_weights.b3_dw_b, s_M_b3_dw, s_arena_buf_b);
    run_pointwise_1x1(s_arena_buf_b, s_weights.b3_pw_w, s_weights.b3_pw_b, s_M_b3_pw, s_arena_buf_a);

    // 7. Layer 11: SUM Reduction over H=17, W=26 (442 spatial positions per channel)
    int8_t sum_q[160];
    const int32_t in_zp_sum = -128;
    const int32_t out_zp_sum = -128;

    for (int c = 0; c < 160; ++c) {
        int32_t acc = 0;
        for (int p = 0; p < 17 * 26; ++p) {
            acc += (int32_t)s_arena_buf_a[p * 160 + c] - in_zp_sum;
        }
        int32_t q = (int32_t)roundf((float)acc * s_M_sum) + out_zp_sum;
        if (q < out_zp_sum) q = out_zp_sum;
        if (q > 127)        q = 127;
        sum_q[c] = (int8_t)q;
    }

    // 8. Layer 12: Fully Connected Linear [1, 160] * [12, 160] + [12] -> [1, 12]
    int8_t out_int8[MODEL_NUM_CLASSES];
    const int32_t in_zp_fc = -128;
    const int32_t out_zp_fc = MODEL_OUTPUT_ZERO_POINT; // 3

    for (int i = 0; i < MODEL_NUM_CLASSES; ++i) {
        int32_t acc = s_weights.fc_b[i];
        const int8_t *w_row = s_weights.fc_w + i * 160;

        for (int c = 0; c < 160; ++c) {
            acc += ((int32_t)sum_q[c] - in_zp_fc) * (int32_t)w_row[c];
        }

        int32_t q = (int32_t)roundf((float)acc * s_M_fc[i]) + out_zp_fc;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        out_int8[i] = (int8_t)q;
    }

    uint32_t t_end = get_dwt_cycles();
    uint32_t elapsed_cycles = t_end - t_start;

    // 9. Format Result Structure
    if (result_out) {
        int best_idx = 0;
        int8_t max_val = -128;

        for (int i = 0; i < MODEL_NUM_CLASSES; ++i) {
            result_out->output_int8[i] = out_int8[i];
            result_out->output_dequantized[i] = MODEL_OUTPUT_SCALE * ((float)out_int8[i] - (float)MODEL_OUTPUT_ZERO_POINT);
            if (out_int8[i] > max_val) {
                max_val = out_int8[i];
                best_idx = i;
            }
        }

        result_out->predicted_index = best_idx;
        result_out->predicted_label = kModelLabels[best_idx];
        result_out->inference_cycles = elapsed_cycles;
        result_out->inference_us = (float)elapsed_cycles / 160.0f;
    }

    return true;
}

bool model_runner_run(const float *features_f32, struct ModelInferenceResult *result_out) {
    // Quantize input features into s_arena_buf_b so pad_in (s_arena_buf_a) is not clobbered
    int8_t *q_in = s_arena_buf_b;
    model_runner_quantize_input(features_f32, q_in);
    return model_runner_invoke(q_in, result_out);
}

size_t model_runner_get_arena_size(void) {
    return sizeof(s_arena_stem_ring) + sizeof(s_arena_buf_a) + sizeof(s_arena_buf_b) + sizeof(s_arena_sum);
}
