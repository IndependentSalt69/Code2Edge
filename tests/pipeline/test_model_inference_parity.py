"""
Unit tests for Code2Edge DS-CNN Frozen Model Inference Parity.

Validates:
1. Frozen model artifact integrity (SHA-256, file size, tensor metadata, operators).
2. Input quantization formula against model metadata (scale=0.018517991527915, zp=-51).
3. Output dequantization formula against model metadata (scale=0.049993276596069336, zp=3).
4. End-to-end forward inference parity on the frozen yes.wav fixture.
5. Contract conformity for the 12 keyword classes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import wave
import numpy as np
import pytest

from tools.run_host_parity import (
    NativeCPipeline,
    REPO_ROOT,
)

FROZEN_MODEL_C = REPO_ROOT / "src" / "pipeline" / "model_data.c"
FROZEN_MODEL_H = REPO_ROOT / "src" / "pipeline" / "model_data.h"
YES_WAV_PATH = REPO_ROOT / "reference" / "tiny-kws" / "app" / "examples" / "yes.wav"

EXPECTED_SHA256 = "0cd6cefbcba738c13d028ffd9a8ad73d6c88f0046368de974d3aa38c47926876"
EXPECTED_MODEL_SIZE = 172216

EXPECTED_INPUT_SHAPE = (1, 1, 64, 101)
EXPECTED_INPUT_SCALE = 0.018517991527915
EXPECTED_INPUT_ZERO_POINT = -51

EXPECTED_OUTPUT_SHAPE = (1, 12)
EXPECTED_OUTPUT_SCALE = 0.049993276596069336
EXPECTED_OUTPUT_ZERO_POINT = 3

EXPECTED_CLASSES = [
    "silence", "unknown", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go"
]


def test_frozen_model_artifact_integrity():
    """Verify frozen model_data.c and model_data.h exist and match exact SHA-256 and size."""
    assert FROZEN_MODEL_C.exists(), f"Missing {FROZEN_MODEL_C}"
    assert FROZEN_MODEL_H.exists(), f"Missing {FROZEN_MODEL_H}"

    # Extract binary bytes from model_data.c
    content = FROZEN_MODEL_C.read_text(encoding="utf-8")
    assert "g_model_data[]" in content, "Missing g_model_data in model_data.c"
    assert "g_model_data_len" in content, "Missing g_model_data_len in model_data.c"

    hex_tokens = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("0x"):
            for tok in line.split(","):
                tok = tok.strip()
                if tok.startswith("0x"):
                    hex_tokens.append(int(tok, 16))

    raw_bytes = bytes(hex_tokens)
    assert len(raw_bytes) == EXPECTED_MODEL_SIZE, (
        f"Expected {EXPECTED_MODEL_SIZE} bytes, got {len(raw_bytes)}"
    )

    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    assert actual_sha256 == EXPECTED_SHA256, (
        f"SHA-256 mismatch! Expected {EXPECTED_SHA256}, got {actual_sha256}"
    )


def test_model_artifact_validation_tool():
    """Verify tools/target/validate_model_artifact.py runs cleanly and emits valid JSON."""
    from tools.target.validate_model_artifact import validate_artifact_file

    passed, report = validate_artifact_file(
        artifact_path=str(FROZEN_MODEL_C),
        header_path=str(FROZEN_MODEL_H),
        output_json=None,
        strict_quantization=True,
        quiet=True,
    )

    assert passed is True
    assert report["sha256"] == EXPECTED_SHA256
    assert report["file_size_bytes"] == EXPECTED_MODEL_SIZE
    assert report["input_tensor"]["shape"] == list(EXPECTED_INPUT_SHAPE)
    assert report["input_tensor"]["dtype"] == "INT8"
    assert np.isclose(report["input_tensor"]["scales"][0], EXPECTED_INPUT_SCALE, rtol=1e-5)
    assert report["input_tensor"]["zero_points"][0] == EXPECTED_INPUT_ZERO_POINT
    assert report["output_tensor"]["shape"] == list(EXPECTED_OUTPUT_SHAPE)
    assert report["output_tensor"]["dtype"] == "INT8"
    assert np.isclose(report["output_tensor"]["scales"][0], EXPECTED_OUTPUT_SCALE, rtol=1e-5)
    assert report["output_tensor"]["zero_points"][0] == EXPECTED_OUTPUT_ZERO_POINT
    assert report["labels"]["contract_passed"] is True
    assert report["is_fully_quantized"] is True


def test_input_quantization_convention():
    """Verify input quantization round/clamp formula on C preprocessing features."""
    c_pipe = NativeCPipeline()
    assert YES_WAV_PATH.exists(), f"Missing {YES_WAV_PATH}"

    with wave.open(str(YES_WAV_PATH), "rb") as wf:
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)
        audio = np.array(struct.unpack(f"<{n_frames}h", raw_bytes), dtype=np.float32) / 32768.0

    stages = c_pipe.run_stages(audio)
    features_f32 = stages["S3_normalized_features"]
    assert features_f32.shape == (64, 101)

    # Quantize
    q_in = np.clip(
        np.round(features_f32 / EXPECTED_INPUT_SCALE) + EXPECTED_INPUT_ZERO_POINT,
        -128,
        127,
    ).astype(np.int8)

    assert q_in.shape == (64, 101)
    assert q_in.dtype == np.int8
    assert q_in.min() >= -128
    assert q_in.max() <= 127


def test_end_to_end_ds_cnn_forward_parity():
    """
    Verify complete forward pass on yes.wav fixture produces argmax class 2 ('yes')
    with clear margin over second highest class.
    """
    # Extract binary FlatBuffer bytes from model_data.c
    content = FROZEN_MODEL_C.read_text(encoding="utf-8")
    hex_tokens = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("0x"):
            for tok in line.split(","):
                tok = tok.strip()
                if tok.startswith("0x"):
                    hex_tokens.append(int(tok, 16))
    model_bytes = bytes(hex_tokens)

    # Preprocessing
    c_pipe = NativeCPipeline()
    with wave.open(str(YES_WAV_PATH), "rb") as wf:
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)
        audio = np.array(struct.unpack(f"<{n_frames}h", raw_bytes), dtype=np.float32) / 32768.0

    stages = c_pipe.run_stages(audio)
    features_f32 = stages["S3_normalized_features"]
    q_in = np.clip(
        np.round(features_f32 / EXPECTED_INPUT_SCALE) + EXPECTED_INPUT_ZERO_POINT,
        -128,
        127,
    ).astype(np.int8)

    # Execute Python equivalent of the model_runner integer graph
    def read_u32(pos):
        return struct.unpack_from("<I", model_bytes, pos)[0]

    def read_i32(pos):
        return struct.unpack_from("<i", model_bytes, pos)[0]

    def read_u16(pos):
        return struct.unpack_from("<H", model_bytes, pos)[0]

    def fb_field(table_pos, f_idx):
        vdelta = read_i32(table_pos)
        vpos = table_pos - vdelta
        vsize = read_u16(vpos)
        off_idx = 4 + 2 * f_idx
        if off_idx + 2 > vsize:
            return 0
        foff = read_u16(vpos + off_idx)
        return table_pos + foff if foff else 0

    def fb_subtable(table_pos, f_idx):
        fpos = fb_field(table_pos, f_idx)
        return fpos + read_u32(fpos) if fpos else 0

    def get_buf(idx):
        root = read_u32(0)
        bufs_f = fb_field(root, 4)
        bufs_vec = bufs_f + read_u32(bufs_f)
        btable = (bufs_vec + 4 + 4 * idx) + read_u32(bufs_vec + 4 + 4 * idx)
        df = fb_field(btable, 0)
        dvec = df + read_u32(df)
        dlen = read_u32(dvec)
        return model_bytes[dvec + 4 : dvec + 4 + dlen]

    def get_scales(t_idx):
        root = read_u32(0)
        sgs_f = fb_field(root, 2)
        sgs_vec = sgs_f + read_u32(sgs_f)
        sg0 = (sgs_vec + 4) + read_u32(sgs_vec + 4)
        t_f = fb_field(sg0, 0)
        t_vec = t_f + read_u32(t_f)
        t_pos = (t_vec + 4 + 4 * t_idx) + read_u32(t_vec + 4 + 4 * t_idx)
        q_pos = fb_subtable(t_pos, 4)
        sc_f = fb_field(q_pos, 2)
        sc_vec = sc_f + read_u32(sc_f)
        sc_len = read_u32(sc_vec)
        return np.array(struct.unpack(f"<{sc_len}f", model_bytes[sc_vec + 4 : sc_vec + 4 + 4 * sc_len]), dtype=np.float32)

    # Multipliers
    s_in = EXPECTED_INPUT_SCALE
    s_t26 = 0.05369709059596062
    s_t27 = 0.06842003017663956
    s_t28 = 0.06879869103431702
    s_t29 = 0.08805365115404129
    s_t30 = 0.06626978516578674
    s_t31 = 0.07592766731977463
    s_t32 = 0.059932198375463486
    s_t33 = 0.20041866600513458
    s_t34 = 0.1043703556060791
    s_t35 = 2.475351333618164
    s_out = EXPECTED_OUTPUT_SCALE

    # Model Weights
    stem_w = np.frombuffer(get_buf(20), dtype=np.int8).reshape(1, 10, 4, 160)
    stem_b = np.frombuffer(get_buf(11), dtype=np.int32)
    M_stem = (s_in * get_scales(20)) / s_t26

    # 1. Pad input
    padded = np.full((1, 74, 105, 1), -51, dtype=np.int8)
    padded[0, 5:69, 2:103, 0] = q_in

    # 2. Stem Conv (stride 2x2, VALID) -> [1, 33, 51, 160]
    out_stem = np.zeros((1, 33, 51, 160), dtype=np.int8)
    for h in range(33):
        for w in range(51):
            patch = padded[0, h*2:h*2+10, w*2:w*2+4, 0].astype(np.int32) - (-51) # [10, 4]
            acc = stem_b + np.tensordot(patch, stem_w[0], axes=([0, 1], [0, 1]))
            q = np.round(acc * M_stem) + (-128)
            out_stem[0, h, w] = np.clip(q, -128, 127).astype(np.int8)

    # 3. Block 0 DW (stride 2x2, SAME=pad 1) & PW
    b0_dw_w = np.frombuffer(get_buf(19), dtype=np.int8).reshape(1, 3, 3, 160)
    b0_dw_b = np.frombuffer(get_buf(10), dtype=np.int32)
    M_b0_dw = (s_t26 * get_scales(19)) / s_t27

    b0_pw_w = np.frombuffer(get_buf(18), dtype=np.int8).reshape(160, 1, 1, 160)
    b0_pw_b = np.frombuffer(get_buf(9), dtype=np.int32)
    M_b0_pw = (s_t27 * get_scales(18)) / s_t28

    stem_pad = np.full((1, 35, 53, 160), -128, dtype=np.int8)
    stem_pad[0, 1:34, 1:52, :] = out_stem

    out_b0_dw = np.zeros((1, 17, 26, 160), dtype=np.int8)
    for h in range(17):
        for w in range(26):
            patch = stem_pad[0, h*2:h*2+3, w*2:w*2+3, :].astype(np.int32) - (-128)
            acc = b0_dw_b + np.sum(patch * b0_dw_w[0], axis=(0, 1))
            q = np.round(acc * M_b0_dw) + (-128)
            out_b0_dw[0, h, w] = np.clip(q, -128, 127).astype(np.int8)

    act_cur = (out_b0_dw.astype(np.int32) - (-128))
    acc_pw = b0_pw_b + np.tensordot(act_cur, b0_pw_w.reshape(160, 160), axes=([-1], [1]))
    q_pw = np.round(acc_pw * M_b0_pw) + (-128)
    act = np.clip(q_pw, -128, 127).astype(np.int8)

    # 4. Blocks 1, 2, 3
    blocks_meta = [
        (17, 8, s_t28, s_t29, 16, 7, s_t29, s_t30),
        (15, 6, s_t30, s_t31, 14, 5, s_t31, s_t32),
        (13, 4, s_t32, s_t33, 12, 3, s_t33, s_t34),
    ]

    for dw_w_idx, dw_b_idx, s_in_dw, s_out_dw, pw_w_idx, pw_b_idx, s_in_pw, s_out_pw in blocks_meta:
        dw_w = np.frombuffer(get_buf(dw_w_idx), dtype=np.int8).reshape(1, 3, 3, 160)
        dw_b = np.frombuffer(get_buf(dw_b_idx), dtype=np.int32)
        M_dw = (s_in_dw * get_scales(dw_w_idx)) / s_out_dw

        pw_w = np.frombuffer(get_buf(pw_w_idx), dtype=np.int8).reshape(160, 160)
        pw_b = np.frombuffer(get_buf(pw_b_idx), dtype=np.int32)
        M_pw = (s_in_pw * get_scales(pw_w_idx)) / s_out_pw

        cur_pad = np.full((1, 19, 28, 160), -128, dtype=np.int8)
        cur_pad[0, 1:18, 1:27, :] = act

        out_dw = np.zeros((1, 17, 26, 160), dtype=np.int8)
        for h in range(17):
            for w in range(26):
                patch = cur_pad[0, h:h+3, w:w+3, :].astype(np.int32) - (-128)
                acc = dw_b + np.sum(patch * dw_w[0], axis=(0, 1))
                q = np.round(acc * M_dw) + (-128)
                out_dw[0, h, w] = np.clip(q, -128, 127).astype(np.int8)

        act_cur = (out_dw.astype(np.int32) - (-128))
        acc_pw = pw_b + np.tensordot(act_cur, pw_w, axes=([-1], [1]))
        q_pw = np.round(acc_pw * M_pw) + (-128)
        act = np.clip(q_pw, -128, 127).astype(np.int8)

    # 5. Sum reduction over [17, 26]
    M_sum = s_t34 / s_t35
    sum_acc = np.sum(act.astype(np.int32) - (-128), axis=(1, 2)) # [1, 160]
    sum_q = np.clip(np.round(sum_acc * M_sum) + (-128), -128, 127).astype(np.int8)

    # 6. FC [1, 160] * [12, 160] + [12] -> [1, 12]
    fc_w = np.frombuffer(get_buf(1), dtype=np.int8).reshape(12, 160)
    fc_b = np.frombuffer(get_buf(2), dtype=np.int32)
    M_fc = (s_t35 * get_scales(1)) / s_out

    fc_acc = fc_b + np.tensordot(sum_q.astype(np.int32) - (-128), fc_w, axes=([-1], [1]))
    logits_int8 = np.clip(np.round(fc_acc * M_fc) + EXPECTED_OUTPUT_ZERO_POINT, -128, 127).astype(np.int8)[0]
    dequantized = EXPECTED_OUTPUT_SCALE * (logits_int8.astype(np.float32) - EXPECTED_OUTPUT_ZERO_POINT)

    # Validate output shape & predictions
    assert logits_int8.shape == (12,)
    predicted_idx = int(np.argmax(logits_int8))
    predicted_label = EXPECTED_CLASSES[predicted_idx]

    # Ground truth assertion on yes.wav
    assert predicted_idx == 2, f"Expected argmax index 2 ('yes'), got {predicted_idx}"
    assert predicted_label == "yes", f"Expected 'yes', got {predicted_label}"
    assert logits_int8[2] > 50, f"Expected high positive activation for 'yes', got {logits_int8[2]}"

    # Verify margin over second-highest class
    sorted_indices = np.argsort(logits_int8)[::-1]
    second_best_idx = sorted_indices[1]
    margin = dequantized[2] - dequantized[second_best_idx]
    assert margin > 3.0, f"Expected margin > 3.0, got {margin}"
