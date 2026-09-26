#!/usr/bin/env python3
"""
Code2Edge: Deployment Model Artifact Validator

Validates the deployable DS-CNN model artifact (.tflite / .h) against the Code2Edge
target deployment contracts for the STM32U585 MCU (ARM Cortex-M33 @ 160 MHz).

Checks:
  1. SHA-256 hash of artifact.
  2. Input shape contract (exactly [1, 1, 64, 101]).
  3. Output shape contract (exactly [1, 12]).
  4. Full quantization check (int8 input/weights/quantization parameters).
  5. Input/output dtype extraction.
  6. Input/output quantization scale and zero-point extraction.
  7. Enumeration of all operators used in the model.
  8. Flagging of operators unsupported for TFLite Micro + CMSIS-NN on Cortex-M33.
  9. File size in bytes and KB.
 10. Intermediate tensor metadata extraction from TFLite FlatBuffer graph.
 11. 12-class label contract validation:
     ['silence', 'unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
 12. Machine-readable JSON report generation under evidence/model/.
 13. Nonzero exit on any contract violation.

NOTE: This tool strictly performs static inspection of the artifact binary.
It does NOT fabricate runtime metrics (arena size, latency, Flash, SRAM, inference logits).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Target hardware constants for STM32U585
TARGET_ID = "STM32U585"
TARGET_RUNTIME = "tflite_micro_cmsis_nn"
EXPECTED_INPUT_SHAPE = [1, 1, 64, 101]
EXPECTED_OUTPUT_SHAPE = [1, 12]
EXPECTED_LABELS = [
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
    "go",
]

# TFLite Builtin Operator Enum Mapping
TFLITE_BUILTIN_OPERATORS: Dict[int, str] = {
    0: "ADD",
    1: "AVERAGE_POOL_2D",
    2: "CONCATENATION",
    3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D",
    5: "DEPTH_TO_SPACE",
    6: "DEQUANTIZE",
    7: "EMBEDDING_LOOKUP",
    8: "FLOOR",
    9: "FULLY_CONNECTED",
    10: "HASHTABLE_LOOKUP",
    11: "L2_NORMALIZATION",
    12: "L2_POOL_2D",
    13: "LOCAL_RESPONSE_NORMALIZATION",
    14: "LOGISTIC",
    15: "LSH_PROJECTION",
    16: "LSTM",
    17: "MAX_POOL_2D",
    18: "MUL",
    19: "RELU",
    20: "RELU_N1_TO_1",
    21: "RELU6",
    22: "RESHAPE",
    23: "RESIZE_BILINEAR",
    24: "RNN",
    25: "SOFTMAX",
    26: "SPACE_TO_DEPTH",
    27: "SVDF",
    28: "TANH",
    29: "CONCAT_EMBEDDINGS",
    30: "SKIP_GRAM",
    31: "CALL",
    32: "CUSTOM",
    33: "EMBEDDING_LOOKUP_SPARSE",
    34: "PAD",
    35: "UNIDIRECTIONAL_SEQUENCE_RNN",
    36: "GATHER",
    37: "BATCH_TO_SPACE_ND",
    38: "SPACE_TO_BATCH_ND",
    39: "TRANSPOSE",
    40: "MEAN",
    41: "SUB",
    42: "DIV",
    43: "SQUEEZE",
    44: "UNIDIRECTIONAL_SEQUENCE_LSTM",
    45: "STRIDED_SLICE",
    46: "BIDIRECTIONAL_SEQUENCE_RNN",
    47: "EXP",
    48: "TOPK_V2",
    49: "SPLIT",
    50: "LOG_SOFTMAX",
    51: "DELEGATE",
    52: "BIDIRECTIONAL_SEQUENCE_LSTM",
    53: "CAST",
    54: "PRELU",
    55: "MAXIMUM",
    56: "ARG_MAX",
    57: "MINIMUM",
    58: "LESS",
    59: "NEG",
    60: "PADV2",
    61: "GREATER",
    62: "GREATER_EQUAL",
    63: "LESS_EQUAL",
    64: "SELECT",
    65: "SLICE",
    66: "SIN",
    67: "TRANSPOSE_CONV",
    68: "SPARSE_TO_DENSE",
    69: "TILE",
    70: "EXPAND_DIMS",
    71: "EQUAL",
    72: "NOT_EQUAL",
    73: "LOG",
    74: "SUM",
    75: "SQRT",
    76: "RSQRT",
    77: "SHAPE",
    78: "POW",
    79: "ARG_MIN",
    80: "FAKE_QUANT",
    81: "REDUCE_PROD",
    82: "REDUCE_MAX",
    83: "PACK",
    84: "LOGICAL_OR",
    85: "ONE_HOT",
    86: "LOGICAL_AND",
    87: "LOGICAL_NOT",
    88: "UNPACK",
    89: "REDUCE_MIN",
    90: "FLOOR_DIV",
    91: "REDUCE_ANY",
    92: "SQUARE",
    93: "ZEROS_LIKE",
    94: "FILL",
    95: "FLOOR_MOD",
    96: "RANGE",
    97: "RESIZE_NEAREST_NEIGHBOR",
    98: "LEAKY_RELU",
    99: "SQUARED_DIFFERENCE",
    100: "MIRROR_PAD",
    101: "ABS",
    102: "SPLIT_V",
    103: "UNIQUE",
    104: "CEIL",
    105: "UNREVERSE",
    106: "ADD_N",
    107: "GATHER_ND",
    108: "COS",
    109: "WHERE",
    110: "RANK",
    111: "ELU",
    112: "REVERSE_SEQUENCE",
    113: "MATRIX_DIAG",
    114: "QUANTIZE",
    115: "MATRIX_SET_DIAG",
    116: "ROUND",
    117: "HARD_SWISH",
    118: "IF",
    119: "WHILE",
    120: "NON_MAX_SUPPRESSION_V4",
    121: "NON_MAX_SUPPRESSION_V5",
    122: "SCATTER_ND",
    123: "SELECT_V2",
    124: "DENSIFY",
    125: "SEGMENT_SUM",
    126: "BATCH_MATMUL",
    127: "PLACEHOLDER_FOR_GREATER_OP_CODES",
    128: "CUMSUM",
    129: "CALL_ONCE",
    130: "BROADCAST_TO",
    131: "RFFT2D",
    132: "CONV_3D",
    133: "IMAG",
    134: "REAL",
    135: "COMPLEX_ABS",
    136: "HASHTABLE",
    137: "HASHTABLE_FIND",
    138: "HASHTABLE_IMPORT",
    139: "HASHTABLE_SIZE",
    140: "REDUCE_ALL",
    141: "CONV_3D_TRANSPOSE",
    142: "VAR_HANDLE",
    143: "READ_VARIABLE",
    144: "ASSIGN_VARIABLE",
    145: "BROADCAST_ARGS",
    146: "RANDOM_STANDARD_NORMAL",
    147: "BUCKETIZE",
    148: "RANDOM_UNIFORM",
    149: "MULTINOMIAL",
    150: "GELU",
    151: "DYNAMIC_UPDATE_SLICE",
    152: "RELU_0_TO_1",
    153: "UNSORTED_SEGMENT_PROD",
    154: "UNSORTED_SEGMENT_MAX",
    155: "UNSORTED_SEGMENT_SUM",
    156: "ATAN2",
    157: "UNSORTED_SEGMENT_MIN",
    158: "SIGN",
    159: "BITCAST",
    160: "BITWISE_XOR",
    161: "RIGHT_SHIFT",
}

# TFLite Tensor DType Enum Mapping
TFLITE_TENSOR_TYPES: Dict[int, str] = {
    0: "FLOAT32",
    1: "FLOAT16",
    2: "INT32",
    3: "UINT8",
    4: "INT64",
    5: "STRING",
    6: "BOOL",
    7: "INT16",
    8: "COMPLEX64",
    9: "INT8",
    10: "FLOAT64",
    11: "COMPLEX128",
    12: "UINT64",
    13: "RESOURCE",
    14: "VARIANT",
    15: "UINT32",
    16: "UINT16",
    17: "INT4",
}

# Supported TFLite Micro / CMSIS-NN Operators for Cortex-M33 (STM32U585)
SUPPORTED_TFLM_OPS = {
    "ADD",
    "AVERAGE_POOL_2D",
    "CONCATENATION",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "DEQUANTIZE",
    "ELU",
    "EXPAND_DIMS",
    "FULLY_CONNECTED",
    "HARD_SWISH",
    "LEAKY_RELU",
    "LOGISTIC",
    "MAX_POOL_2D",
    "MAXIMUM",
    "MEAN",
    "MINIMUM",
    "MUL",
    "NEG",
    "PAD",
    "PADV2",
    "QUANTIZE",
    "RELU",
    "RELU6",
    "RESHAPE",
    "ROUND",
    "SOFTMAX",
    "SPLIT",
    "SPLIT_V",
    "SQUEEZE",
    "STRIDED_SLICE",
    "SUB",
    "TANH",
    "TRANSPOSE",
}


class TFLiteFlatBufferReader:
    """Low-level pure-Python FlatBuffer parser for TFLite models."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        if len(data) < 8:
            raise ValueError("Invalid TFLite binary: buffer too small (< 8 bytes)")
        self.root_pos = struct.unpack_from("<I", data, 0)[0]
        self.identifier = data[4:8]

    def _get_field(self, table_pos: int, field_idx: int) -> int:
        if table_pos <= 0 or table_pos + 4 > len(self.data):
            return 0
        vtable_delta = struct.unpack_from("<i", self.data, table_pos)[0]
        vtable_pos = table_pos - vtable_delta
        if vtable_pos < 0 or vtable_pos + 4 > len(self.data):
            return 0
        vtable_size = struct.unpack_from("<H", self.data, vtable_pos)[0]
        offset_idx = 4 + 2 * field_idx
        if offset_idx + 2 > vtable_size or vtable_pos + offset_idx + 2 > len(self.data):
            return 0
        field_offset = struct.unpack_from("<H", self.data, vtable_pos + offset_idx)[0]
        if field_offset == 0:
            return 0
        return table_pos + field_offset

    def _get_subtable(self, table_pos: int, field_idx: int) -> int:
        field_pos = self._get_field(table_pos, field_idx)
        if not field_pos or field_pos + 4 > len(self.data):
            return 0
        offset = struct.unpack_from("<I", self.data, field_pos)[0]
        if offset == 0 or field_pos + offset > len(self.data):
            return 0
        return field_pos + offset

    def _read_str(self, field_pos: int) -> str:
        if not field_pos or field_pos + 4 > len(self.data):
            return ""
        str_offset = struct.unpack_from("<I", self.data, field_pos)[0]
        str_pos = field_pos + str_offset
        if str_pos + 4 > len(self.data):
            return ""
        length = struct.unpack_from("<I", self.data, str_pos)[0]
        if str_pos + 4 + length > len(self.data):
            return ""
        return self.data[str_pos + 4 : str_pos + 4 + length].decode("utf-8", errors="replace")

    def _read_table_vector(self, field_pos: int) -> List[int]:
        if not field_pos or field_pos + 4 > len(self.data):
            return []
        vec_offset = struct.unpack_from("<I", self.data, field_pos)[0]
        vec_pos = field_pos + vec_offset
        if vec_pos + 4 > len(self.data):
            return []
        length = struct.unpack_from("<I", self.data, vec_pos)[0]
        res = []
        for i in range(length):
            elem_pos = vec_pos + 4 + i * 4
            if elem_pos + 4 > len(self.data):
                break
            tbl_offset = struct.unpack_from("<I", self.data, elem_pos)[0]
            res.append(elem_pos + tbl_offset)
        return res

    def _read_int32_vector(self, field_pos: int) -> List[int]:
        if not field_pos or field_pos + 4 > len(self.data):
            return []
        vec_offset = struct.unpack_from("<I", self.data, field_pos)[0]
        vec_pos = field_pos + vec_offset
        if vec_pos + 4 > len(self.data):
            return []
        length = struct.unpack_from("<I", self.data, vec_pos)[0]
        res = []
        for i in range(length):
            elem_pos = vec_pos + 4 + i * 4
            if elem_pos + 4 > len(self.data):
                break
            res.append(struct.unpack_from("<i", self.data, elem_pos)[0])
        return res

    def _read_float32_vector(self, field_pos: int) -> List[float]:
        if not field_pos or field_pos + 4 > len(self.data):
            return []
        vec_offset = struct.unpack_from("<I", self.data, field_pos)[0]
        vec_pos = field_pos + vec_offset
        if vec_pos + 4 > len(self.data):
            return []
        length = struct.unpack_from("<I", self.data, vec_pos)[0]
        res = []
        for i in range(length):
            elem_pos = vec_pos + 4 + i * 4
            if elem_pos + 4 > len(self.data):
                break
            res.append(struct.unpack_from("<f", self.data, elem_pos)[0])
        return res

    def _read_int64_vector(self, field_pos: int) -> List[int]:
        if not field_pos or field_pos + 4 > len(self.data):
            return []
        vec_offset = struct.unpack_from("<I", self.data, field_pos)[0]
        vec_pos = field_pos + vec_offset
        if vec_pos + 4 > len(self.data):
            return []
        length = struct.unpack_from("<I", self.data, vec_pos)[0]
        res = []
        for i in range(length):
            elem_pos = vec_pos + 4 + i * 8
            if elem_pos + 8 > len(self.data):
                break
            res.append(struct.unpack_from("<q", self.data, elem_pos)[0])
        return res

    def _read_bytes_vector(self, field_pos: int) -> bytes:
        if not field_pos or field_pos + 4 > len(self.data):
            return b""
        vec_offset = struct.unpack_from("<I", self.data, field_pos)[0]
        vec_pos = field_pos + vec_offset
        if vec_pos + 4 > len(self.data):
            return b""
        length = struct.unpack_from("<I", self.data, vec_pos)[0]
        if vec_pos + 4 + length > len(self.data):
            return b""
        return bytes(self.data[vec_pos + 4 : vec_pos + 4 + length])

    def parse(self) -> Dict[str, Any]:
        """Parse Model table and extract subgraphs, operators, tensors, metadata."""
        version_pos = self._get_field(self.root_pos, 0)
        version = struct.unpack_from("<I", self.data, version_pos)[0] if version_pos else 0

        desc_pos = self._get_field(self.root_pos, 3)
        description = self._read_str(desc_pos)

        # Buffers
        bufs_pos = self._get_field(self.root_pos, 4)
        buf_tables = self._read_table_vector(bufs_pos)
        buffers = []
        for b_pos in buf_tables:
            data_pos = self._get_field(b_pos, 0)
            data_bytes = self._read_bytes_vector(data_pos)
            buffers.append(data_bytes)

        # Operator codes
        opc_pos = self._get_field(self.root_pos, 1)
        opc_tables = self._read_table_vector(opc_pos)
        opcodes = []
        for oc_pos in opc_tables:
            dep_code_pos = self._get_field(oc_pos, 0)
            dep_code = struct.unpack_from("<b", self.data, dep_code_pos)[0] if dep_code_pos else 0
            custom_code = self._read_str(self._get_field(oc_pos, 1))
            code_pos = self._get_field(oc_pos, 3)
            code = struct.unpack_from("<i", self.data, code_pos)[0] if code_pos else 0
            if code == 0 and dep_code != 0:
                code = dep_code
            op_name = TFLITE_BUILTIN_OPERATORS.get(code, f"UNKNOWN_OP_{code}")
            if op_name == "CUSTOM" and custom_code:
                op_name = f"CUSTOM_{custom_code}"
            opcodes.append({
                "code": code,
                "name": op_name,
                "custom_code": custom_code,
            })

        # Subgraphs
        sg_pos = self._get_field(self.root_pos, 2)
        sg_tables = self._read_table_vector(sg_pos)
        subgraphs = []
        for s_pos in sg_tables:
            tensors_pos = self._get_field(s_pos, 0)
            t_tables = self._read_table_vector(tensors_pos)
            tensors = []
            for t_pos in t_tables:
                shape_pos = self._get_field(t_pos, 0)
                shape = self._read_int32_vector(shape_pos)
                type_pos = self._get_field(t_pos, 1)
                dtype_code = struct.unpack_from("<b", self.data, type_pos)[0] if type_pos else 0
                dtype_str = TFLITE_TENSOR_TYPES.get(dtype_code, f"UNKNOWN_DTYPE_{dtype_code}")
                buf_pos = self._get_field(t_pos, 2)
                buf_idx = struct.unpack_from("<I", self.data, buf_pos)[0] if buf_pos else 0
                name = self._read_str(self._get_field(t_pos, 3))

                # Quantization parameters
                q_pos = self._get_subtable(t_pos, 4)
                scales = []
                zero_points = []
                if q_pos:
                    scales = self._read_float32_vector(self._get_field(q_pos, 2))
                    zero_points = self._read_int64_vector(self._get_field(q_pos, 3))

                tensors.append({
                    "name": name,
                    "shape": shape,
                    "dtype": dtype_str,
                    "dtype_code": dtype_code,
                    "buffer_idx": buf_idx,
                    "scales": scales,
                    "zero_points": zero_points,
                })

            inputs = self._read_int32_vector(self._get_field(s_pos, 1))
            outputs = self._read_int32_vector(self._get_field(s_pos, 2))

            # Operators
            ops_pos = self._get_field(s_pos, 3)
            ops_tables = self._read_table_vector(ops_pos)
            operators = []
            for op_pos in ops_tables:
                opcode_idx_pos = self._get_field(op_pos, 0)
                opcode_idx = struct.unpack_from("<I", self.data, opcode_idx_pos)[0] if opcode_idx_pos else 0
                in_tensors = self._read_int32_vector(self._get_field(op_pos, 1))
                out_tensors = self._read_int32_vector(self._get_field(op_pos, 2))
                op_name = opcodes[opcode_idx]["name"] if opcode_idx < len(opcodes) else f"INVALID_OPCODE_{opcode_idx}"
                operators.append({
                    "opcode_idx": opcode_idx,
                    "op_name": op_name,
                    "inputs": in_tensors,
                    "outputs": out_tensors,
                })

            subgraphs.append({
                "tensors": tensors,
                "inputs": inputs,
                "outputs": outputs,
                "operators": operators,
            })

        # Metadata
        meta_pos = self._get_field(self.root_pos, 6)
        meta_tables = self._read_table_vector(meta_pos)
        metadata = []
        for m_pos in meta_tables:
            name = self._read_str(self._get_field(m_pos, 0))
            buf_pos = self._get_field(m_pos, 1)
            buf_idx = struct.unpack_from("<I", self.data, buf_pos)[0] if buf_pos else 0
            metadata.append({"name": name, "buffer_idx": buf_idx})

        return {
            "version": version,
            "description": description,
            "opcodes": opcodes,
            "subgraphs": subgraphs,
            "metadata": metadata,
            "buffers": buffers,
        }


def extract_bytes_from_c_header(header_content: str) -> bytes:
    """Extract FlatBuffer binary byte stream from C/C++ header array definition."""
    match = re.search(
        r"(?:const\s+)?(?:unsigned\s+char|uint8_t)\s+\w+\[\s*\]\s*(?:alignas\s*\(\s*\d+\s*\))?\s*=\s*\{([^}]+)\}",
        header_content,
        re.DOTALL,
    )
    if not match:
        # Fallback: search for any array with curly brace byte tokens
        match = re.search(r"=\s*\{([^}]+)\}\s*;", header_content, re.DOTALL)
        if not match:
            raise ValueError("No valid byte array found in C header file.")

    array_content = match.group(1)
    tokens = re.findall(r"0x[0-9a-fA-F]+|\d+", array_content)
    if not tokens:
        raise ValueError("Byte array in C header contains zero numeric elements.")
    byte_list = [int(tok, 0) for tok in tokens]
    return bytes(byte_list)


def validate_model_bytes(
    model_bytes: bytes,
    artifact_path: str,
    header_path: Optional[str] = None,
    expected_input_shape: Optional[Sequence[int]] = None,
    expected_output_shape: Optional[Sequence[int]] = None,
    expected_labels: Optional[Sequence[str]] = None,
    strict_quantization: bool = True,
) -> Dict[str, Any]:
    """
    Validate TFLite FlatBuffer binary bytes against Code2Edge deployment contracts.
    
    Returns structured validation report dictionary.
    """
    if expected_input_shape is None:
        expected_input_shape = EXPECTED_INPUT_SHAPE
    if expected_output_shape is None:
        expected_output_shape = EXPECTED_OUTPUT_SHAPE
    if expected_labels is None:
        expected_labels = EXPECTED_LABELS

    errors: List[str] = []
    warnings: List[str] = []

    # 1. SHA-256 and size
    sha256_hash = hashlib.sha256(model_bytes).hexdigest()
    file_size_bytes = len(model_bytes)
    file_size_kb = round(file_size_bytes / 1024.0, 2)

    # Cross check with header file if provided
    header_info = None
    if header_path:
        hp = Path(header_path)
        if not hp.exists():
            errors.append(f"Specified header file does not exist: {header_path}")
        else:
            try:
                hdr_bytes = extract_bytes_from_c_header(hp.read_text(encoding="utf-8"))
                hdr_sha256 = hashlib.sha256(hdr_bytes).hexdigest()
                header_info = {
                    "path": str(hp),
                    "size_bytes": len(hdr_bytes),
                    "sha256": hdr_sha256,
                    "matches_tflite_bytes": (hdr_sha256 == sha256_hash),
                }
                if hdr_sha256 != sha256_hash:
                    errors.append(
                        f"SHA-256 mismatch between C header array ({hdr_sha256}) and model file ({sha256_hash})"
                    )
            except Exception as e:
                errors.append(f"Failed to parse C header array from {header_path}: {e}")

    # 2. Parse FlatBuffer
    try:
        reader = TFLiteFlatBufferReader(model_bytes)
        parsed = reader.parse()
    except Exception as e:
        errors.append(f"Corrupted or invalid TFLite FlatBuffer: {e}")
        return {
            "schema_version": "1.0.0",
            "tool": "validate_model_artifact",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "artifact_path": artifact_path,
            "sha256": sha256_hash,
            "file_size_bytes": file_size_bytes,
            "file_size_kb": file_size_kb,
            "passed": False,
            "errors": errors,
            "warnings": warnings,
        }

    subgraphs = parsed.get("subgraphs", [])
    if not subgraphs:
        errors.append("Model contains 0 subgraphs.")
        return {
            "schema_version": "1.0.0",
            "tool": "validate_model_artifact",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "artifact_path": artifact_path,
            "sha256": sha256_hash,
            "file_size_bytes": file_size_bytes,
            "file_size_kb": file_size_kb,
            "passed": False,
            "errors": errors,
            "warnings": warnings,
        }

    main_sg = subgraphs[0]
    tensors = main_sg.get("tensors", [])
    inputs = main_sg.get("inputs", [])
    outputs = main_sg.get("outputs", [])
    operators = main_sg.get("operators", [])

    # 3. Input Tensor Verification
    input_tensor_info: Dict[str, Any] = {}
    if not inputs or inputs[0] >= len(tensors):
        errors.append("Model does not define a valid primary input tensor.")
    else:
        in_t = tensors[inputs[0]]
        input_tensor_info = {
            "index": inputs[0],
            "name": in_t["name"],
            "shape": in_t["shape"],
            "dtype": in_t["dtype"],
            "scales": in_t["scales"],
            "zero_points": in_t["zero_points"],
        }
        if list(in_t["shape"]) != list(expected_input_shape):
            errors.append(
                f"Input shape mismatch: model has {in_t['shape']}, expected {list(expected_input_shape)}"
            )
        if in_t["dtype"] not in ("INT8", "UINT8", "INT16"):
            if strict_quantization:
                errors.append(
                    f"Input dtype is '{in_t['dtype']}', expected quantized dtype ('INT8', 'UINT8', or 'INT16')"
                )
            else:
                warnings.append(f"Input dtype is unquantized '{in_t['dtype']}'.")

    # 4. Output Tensor Verification
    output_tensor_info: Dict[str, Any] = {}
    if not outputs or outputs[0] >= len(tensors):
        errors.append("Model does not define a valid primary output tensor.")
    else:
        out_t = tensors[outputs[0]]
        output_tensor_info = {
            "index": outputs[0],
            "name": out_t["name"],
            "shape": out_t["shape"],
            "dtype": out_t["dtype"],
            "scales": out_t["scales"],
            "zero_points": out_t["zero_points"],
        }
        if list(out_t["shape"]) != list(expected_output_shape):
            errors.append(
                f"Output shape mismatch: model has {out_t['shape']}, expected {list(expected_output_shape)}"
            )

    # 5. Quantization Checks
    is_fully_quantized = True
    quantized_tensor_count = 0
    float_tensor_count = 0
    for idx, t in enumerate(tensors):
        if t["dtype"] in ("FLOAT32", "FLOAT64", "FLOAT16"):
            float_tensor_count += 1
            # If intermediate activation or weight is float, mark unquantized
            if idx in inputs or idx in outputs or t["buffer_idx"] != 0:
                is_fully_quantized = False
        elif t["dtype"] in ("INT8", "UINT8", "INT16", "INT32"):
            quantized_tensor_count += 1

    if strict_quantization and not is_fully_quantized:
        errors.append(
            f"Model is not fully quantized: contains {float_tensor_count} floating-point tensors."
        )

    # 6. Operators & Target Compatibility
    ops_used: List[str] = []
    unsupported_ops: List[str] = []
    for op in operators:
        op_name = op["op_name"]
        if op_name not in ops_used:
            ops_used.append(op_name)
        if op_name not in SUPPORTED_TFLM_OPS:
            if op_name not in unsupported_ops:
                unsupported_ops.append(op_name)

    if unsupported_ops:
        errors.append(
            f"Model contains {len(unsupported_ops)} operators unsupported by {TARGET_RUNTIME} on {TARGET_ID}: {unsupported_ops}"
        )

    # 7. 12-Class Label Contract Validation
    labels_found: Optional[List[str]] = None
    buffers = parsed.get("buffers", [])
    metadata = parsed.get("metadata", [])
    for meta in metadata:
        if "label" in meta["name"].lower():
            b_idx = meta["buffer_idx"]
            if b_idx < len(buffers) and buffers[b_idx]:
                raw_text = buffers[b_idx].decode("utf-8", errors="replace")
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                if lines:
                    labels_found = lines
                    break

    labels_contract_passed = True
    if labels_found is not None:
        if labels_found != list(expected_labels):
            errors.append(
                f"Embedded label contract mismatch:\nFound:    {labels_found}\nExpected: {list(expected_labels)}"
            )
            labels_contract_passed = False
    else:
        # If no label metadata is embedded, verify output dimension matches exactly 12 classes
        if output_tensor_info and output_tensor_info.get("shape"):
            out_classes = output_tensor_info["shape"][-1]
            if out_classes != len(expected_labels):
                errors.append(
                    f"Model output dimension ({out_classes}) does not match required {len(expected_labels)} classes."
                )
                labels_contract_passed = False
            else:
                warnings.append(
                    f"No embedded label metadata found in TFLite; validated 12-class output dimension against contract: {list(expected_labels)}"
                )

    passed = (len(errors) == 0)

    report = {
        "schema_version": "1.0.0",
        "tool": "validate_model_artifact",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifact_path": artifact_path,
        "target": {
            "target_id": TARGET_ID,
            "runtime": TARGET_RUNTIME,
            "compatible": (len(unsupported_ops) == 0),
        },
        "sha256": sha256_hash,
        "file_size_bytes": file_size_bytes,
        "file_size_kb": file_size_kb,
        "header_file": header_info,
        "input_tensor": input_tensor_info,
        "output_tensor": output_tensor_info,
        "is_fully_quantized": is_fully_quantized,
        "operators_used": ops_used,
        "unsupported_operators": unsupported_ops,
        "tensor_count": len(tensors),
        "operator_count": len(operators),
        "labels": {
            "expected": list(expected_labels),
            "found_in_metadata": labels_found,
            "contract_passed": labels_contract_passed,
        },
        "tensors": [
            {
                "index": i,
                "name": t["name"],
                "shape": t["shape"],
                "dtype": t["dtype"],
                "buffer_idx": t["buffer_idx"],
                "scales": t["scales"],
                "zero_points": t["zero_points"],
            }
            for i, t in enumerate(tensors)
        ],
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
    }

    return report


def validate_artifact_file(
    artifact_path: str | Path,
    header_path: Optional[str | Path] = None,
    output_json: Optional[str | Path] = None,
    strict_quantization: bool = True,
    quiet: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate a .tflite or .h artifact file and output a JSON evidence report."""
    art_path = Path(artifact_path)
    if not art_path.exists():
        raise FileNotFoundError(f"Model artifact file not found: {art_path}")

    # Read binary bytes
    if art_path.suffix.lower() in (".h", ".hpp", ".c", ".cpp"):
        model_bytes = extract_bytes_from_c_header(art_path.read_text(encoding="utf-8"))
    else:
        model_bytes = art_path.read_bytes()

    report = validate_model_bytes(
        model_bytes=model_bytes,
        artifact_path=str(art_path),
        header_path=str(header_path) if header_path else None,
        strict_quantization=strict_quantization,
    )

    # Write output JSON evidence
    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not quiet:
            print(f"[Code2Edge] Wrote model artifact evidence to {out_p}")

    if not quiet:
        print_validation_summary(report)

    return report["passed"], report


def print_validation_summary(report: Dict[str, Any]) -> None:
    """Print human-readable validation summary."""
    status = "PASS" if report["passed"] else "FAIL"
    print("\n" + "=" * 70)
    print(f"Code2Edge DS-CNN Deployment Model Validation: [{status}]")
    print("=" * 70)
    print(f"Artifact Path     : {report['artifact_path']}")
    print(f"SHA-256           : {report['sha256']}")
    print(f"File Size         : {report['file_size_bytes']} bytes ({report['file_size_kb']} KB)")
    print(f"Target Hardware   : {report['target']['target_id']} ({report['target']['runtime']})")

    in_t = report.get("input_tensor", {})
    if in_t:
        print(f"Input Tensor      : shape={in_t.get('shape')} dtype={in_t.get('dtype')} scale={in_t.get('scales')} zp={in_t.get('zero_points')}")
    out_t = report.get("output_tensor", {})
    if out_t:
        print(f"Output Tensor     : shape={out_t.get('shape')} dtype={out_t.get('dtype')} scale={out_t.get('scales')} zp={out_t.get('zero_points')}")

    print(f"Fully Quantized   : {report.get('is_fully_quantized')}")
    print(f"Operators Used    : {', '.join(report.get('operators_used', []))}")
    if report.get("unsupported_operators"):
        print(f"Unsupported Ops   : {', '.join(report['unsupported_operators'])}")

    if report.get("warnings"):
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  [!] {w}")

    if report.get("errors"):
        print("\nErrors:")
        for e in report["errors"]:
            print(f"  [X] {e}")

    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate deployable DS-CNN model artifact against Code2Edge STM32U585 deployment contracts."
    )
    parser.add_argument(
        "model_file",
        nargs="?",
        default=None,
        help="Path to .tflite model binary or C header (.h) containing model array.",
    )
    parser.add_argument(
        "--model-file",
        "-m",
        dest="model_file_opt",
        default=None,
        help="Path to .tflite model binary.",
    )
    parser.add_argument(
        "--header-file",
        "-H",
        default=None,
        help="Path to generated C header file (e.g. src/pipeline/model_data.h) to cross-verify against .tflite.",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        default="evidence/model/model_artifact_validation.json",
        help="Path for machine-readable JSON validation report (default: evidence/model/model_artifact_validation.json).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Enforce strict int8 full quantization and shape contracts (default: True).",
    )
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Allow warnings instead of hard failure on non-int8 dtypes.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_file_opt or args.model_file

    if not model_path:
        print("[Code2Edge] Error: No model file provided. Usage: python tools/target/validate_model_artifact.py <path/to/model.tflite>", file=sys.stderr)
        return 1

    try:
        passed, report = validate_artifact_file(
            artifact_path=model_path,
            header_path=args.header_file,
            output_json=args.output_json,
            strict_quantization=args.strict,
            quiet=args.quiet,
        )
        return 0 if passed else 1
    except Exception as e:
        print(f"[Code2Edge] Fatal validation error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
