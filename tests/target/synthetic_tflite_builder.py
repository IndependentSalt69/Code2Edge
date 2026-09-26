"""
Helper module to build synthetic TFLite FlatBuffers binaries for unit testing.

NOTE: These models are SYNTHETIC mock models used strictly for contract and toolchain
validation testing. They do NOT contain trained weights or real inference weights.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import flatbuffers


def build_synthetic_tflite(
    input_shape: Sequence[int] = (1, 1, 64, 101),
    output_shape: Sequence[int] = (1, 12),
    input_dtype: int = 9,  # 9 = INT8, 0 = FLOAT32
    output_dtype: int = 9,  # 9 = INT8, 0 = FLOAT32
    weight_dtype: int = 9,  # 9 = INT8, 0 = FLOAT32
    input_scale: float = 0.05,
    input_zp: int = 0,
    output_scale: float = 0.00390625,
    output_zp: int = -128,
    opcodes: Sequence[int] = (3, 4, 9, 25),  # CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, SOFTMAX
    custom_op_name: Optional[str] = None,
    labels: Optional[Sequence[str]] = (
        "silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
    ),
    description: str = "SYNTHETIC_TEST_MODEL_NOT_REAL_WEIGHTS",
) -> bytes:
    """Construct a synthetic TFLite FlatBuffer binary."""
    b = flatbuffers.Builder(4096)

    s_desc = b.CreateString(description)
    s_t0 = b.CreateString("input_spectrogram")
    s_t1 = b.CreateString("conv_weights")
    s_t2 = b.CreateString("conv_features")
    s_t3 = b.CreateString("output_logits")
    s_meta_name = b.CreateString("labels")

    # Quantization params for Input T0
    b.StartVector(4, 1, 4)
    b.PrependFloat32(input_scale)
    v_s0 = b.EndVector()
    b.StartVector(8, 1, 8)
    b.PrependInt64(input_zp)
    v_zp0 = b.EndVector()
    b.StartObject(6)
    b.PrependUOffsetTRelativeSlot(2, v_s0, 0)
    b.PrependUOffsetTRelativeSlot(3, v_zp0, 0)
    q0 = b.EndObject()

    # Quantization params for Output T3
    b.StartVector(4, 1, 4)
    b.PrependFloat32(output_scale)
    v_s3 = b.EndVector()
    b.StartVector(8, 1, 8)
    b.PrependInt64(output_zp)
    v_zp3 = b.EndVector()
    b.StartObject(6)
    b.PrependUOffsetTRelativeSlot(2, v_s3, 0)
    b.PrependUOffsetTRelativeSlot(3, v_zp3, 0)
    q3 = b.EndObject()

    # Shapes
    # T0 (Input)
    b.StartVector(4, len(input_shape), 4)
    for dim in reversed(input_shape):
        b.PrependInt32(dim)
    sh0 = b.EndVector()

    # T1 (Weights)
    sh1_dims = [160, 1, 4, 10]
    b.StartVector(4, len(sh1_dims), 4)
    for dim in reversed(sh1_dims):
        b.PrependInt32(dim)
    sh1 = b.EndVector()

    # T2 (Intermediate)
    sh2_dims = [1, 160, 61, 92]
    b.StartVector(4, len(sh2_dims), 4)
    for dim in reversed(sh2_dims):
        b.PrependInt32(dim)
    sh2 = b.EndVector()

    # T3 (Output)
    b.StartVector(4, len(output_shape), 4)
    for dim in reversed(output_shape):
        b.PrependInt32(dim)
    sh3 = b.EndVector()

    # Tensors
    # Tensor 3 (Output)
    b.StartObject(9)
    b.PrependUOffsetTRelativeSlot(0, sh3, 0)
    b.PrependInt8Slot(1, output_dtype, 0)
    b.PrependUint32Slot(2, 0, 0)
    b.PrependUOffsetTRelativeSlot(3, s_t3, 0)
    b.PrependUOffsetTRelativeSlot(4, q3, 0)
    t3 = b.EndObject()

    # Tensor 2 (Intermediate)
    b.StartObject(9)
    b.PrependUOffsetTRelativeSlot(0, sh2, 0)
    b.PrependInt8Slot(1, weight_dtype, 0)
    b.PrependUint32Slot(2, 0, 0)
    b.PrependUOffsetTRelativeSlot(3, s_t2, 0)
    t2 = b.EndObject()

    # Tensor 1 (Weights)
    b.StartObject(9)
    b.PrependUOffsetTRelativeSlot(0, sh1, 0)
    b.PrependInt8Slot(1, weight_dtype, 0)
    b.PrependUint32Slot(2, 1, 0)
    b.PrependUOffsetTRelativeSlot(3, s_t1, 0)
    t1 = b.EndObject()

    # Tensor 0 (Input)
    b.StartObject(9)
    b.PrependUOffsetTRelativeSlot(0, sh0, 0)
    b.PrependInt8Slot(1, input_dtype, 0)
    b.PrependUint32Slot(2, 0, 0)
    b.PrependUOffsetTRelativeSlot(3, s_t0, 0)
    b.PrependUOffsetTRelativeSlot(4, q0, 0)
    t0 = b.EndObject()

    b.StartVector(4, 4, 4)
    b.PrependUOffsetTRelative(t3)
    b.PrependUOffsetTRelative(t2)
    b.PrependUOffsetTRelative(t1)
    b.PrependUOffsetTRelative(t0)
    v_tensors = b.EndVector()

    b.StartVector(4, 1, 4)
    b.PrependInt32(0)
    v_inputs = b.EndVector()

    b.StartVector(4, 1, 4)
    b.PrependInt32(3)
    v_outputs = b.EndVector()

    # Operators
    built_ops = []
    for op_idx, code in enumerate(opcodes):
        b.StartVector(4, 2, 4)
        b.PrependInt32(1)
        b.PrependInt32(0)
        op_in = b.EndVector()

        b.StartVector(4, 1, 4)
        b.PrependInt32(2 if op_idx < len(opcodes) - 1 else 3)
        op_out = b.EndVector()

        b.StartObject(8)
        b.PrependUint32Slot(0, op_idx, 0)
        b.PrependUOffsetTRelativeSlot(1, op_in, 0)
        b.PrependUOffsetTRelativeSlot(2, op_out, 0)
        built_ops.append(b.EndObject())

    b.StartVector(4, len(built_ops), 4)
    for op in reversed(built_ops):
        b.PrependUOffsetTRelative(op)
    v_ops = b.EndVector()

    # Subgraph
    b.StartObject(5)
    b.PrependUOffsetTRelativeSlot(0, v_tensors, 0)
    b.PrependUOffsetTRelativeSlot(1, v_inputs, 0)
    b.PrependUOffsetTRelativeSlot(2, v_outputs, 0)
    b.PrependUOffsetTRelativeSlot(3, v_ops, 0)
    subgraph = b.EndObject()

    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(subgraph)
    v_subgraphs = b.EndVector()

    # Operator Codes
    built_opcodes = []
    for code in opcodes:
        s_custom = b.CreateString(custom_op_name) if (code == 32 and custom_op_name) else 0
        b.StartObject(4)
        b.PrependInt8Slot(0, code if code < 127 else 127, 0)
        if s_custom:
            b.PrependUOffsetTRelativeSlot(1, s_custom, 0)
        b.PrependInt32Slot(3, code, 0)
        built_opcodes.append(b.EndObject())

    b.StartVector(4, len(built_opcodes), 4)
    for oc in reversed(built_opcodes):
        b.PrependUOffsetTRelative(oc)
    v_opcodes = b.EndVector()

    # Buffers
    # Buffer 0: Empty
    b.StartObject(3)
    buf0 = b.EndObject()

    # Buffer 1: Weight bytes
    b.StartVector(1, 64, 1)
    for _ in range(64):
        b.PrependByte(1)
    v_buf1_data = b.EndVector()
    b.StartObject(3)
    b.PrependUOffsetTRelativeSlot(0, v_buf1_data, 0)
    buf1 = b.EndObject()

    # Buffer 2: Labels
    buf_list = [buf0, buf1]
    v_meta = 0
    if labels is not None:
        label_text = "\n".join(labels) + "\n"
        label_bytes = label_text.encode("utf-8")
        b.StartVector(1, len(label_bytes), 1)
        for byte in reversed(label_bytes):
            b.PrependByte(byte)
        v_buf2_data = b.EndVector()
        b.StartObject(3)
        b.PrependUOffsetTRelativeSlot(0, v_buf2_data, 0)
        buf2 = b.EndObject()
        buf_list.append(buf2)

        # Metadata
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, s_meta_name, 0)
        b.PrependUint32Slot(1, 2, 0)
        meta0 = b.EndObject()

        b.StartVector(4, 1, 4)
        b.PrependUOffsetTRelative(meta0)
        v_meta = b.EndVector()

    b.StartVector(4, len(buf_list), 4)
    for buf in reversed(buf_list):
        b.PrependUOffsetTRelative(buf)
    v_buffers = b.EndVector()

    # Model
    b.StartObject(8)
    b.PrependUint32Slot(0, 3, 0)  # version
    b.PrependUOffsetTRelativeSlot(1, v_opcodes, 0)
    b.PrependUOffsetTRelativeSlot(2, v_subgraphs, 0)
    b.PrependUOffsetTRelativeSlot(3, s_desc, 0)
    b.PrependUOffsetTRelativeSlot(4, v_buffers, 0)
    if v_meta:
        b.PrependUOffsetTRelativeSlot(6, v_meta, 0)
    model = b.EndObject()

    b.Finish(model, file_identifier=b"TFL3")
    return b.Output()
