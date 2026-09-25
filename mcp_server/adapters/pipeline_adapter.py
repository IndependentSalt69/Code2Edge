"""
mcp_server/adapters/pipeline_adapter.py

Adapter for Person A's pipeline and parity outputs. Read-only: never writes
into src/pipeline/, reference/, or tools/.

profile_model and inspect_pipeline are wired to Person A's real outputs
(reference/pipeline_manifest.json, reference/tiny-kws/assets/metrics.json,
reference/normalization.json, reference/corpus_manifest.json,
reference/tiny-kws/src/model.py, reference/tiny-kws/src/common.py) as of
2026-09-26 (Prompt 10A). See docs/interface-requests.md for the two gaps
this surfaced: no exported quantized model, and the real 4-stage pipeline
not matching contracts/inspect_pipeline.schema.json's 7-stage enum.

run_parity_test and quantize_model are still stubs: Person A's C++ so far
only compares a single stage (power spectrum, via tools/compare_power.py)
against a single sample, not the full stage-wise JSON harness our
contract needs (Prompt 10B, once that lands).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mcp_server._ids import new_run_id, utcnow_iso

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((_REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def run_profile_model(repo_path: str, model_file: str,
                      labels_file: str = "", config_file: str = "") -> dict[str, Any]:
    """Return profile_model Output.model for the tiny-kws DS-CNN.

    No exported .tflite/.onnx exists yet (see docs/interface-requests.md),
    so total_params/total_macs/layers are derived analytically from
    reference/tiny-kws/src/model.py's DSCNN(width=160, n_blocks=4,
    n_classes=12) definition — the exact architecture the checkpoint used,
    confirmed because the analytical param count matches
    assets/metrics.json's n_parameters exactly (119372 == 119372).
    """
    metrics = _load_json("reference/tiny-kws/assets/metrics.json")
    reported_params = metrics["n_parameters"]

    width, n_blocks, n_classes = 160, 4, 12
    in_h, in_w = 64, 101  # log-mel spectrogram, from LogMel.forward's docstring

    def conv_out(size: int, kernel: int, stride: int, pad: int) -> int:
        return (size + 2 * pad - kernel) // stride + 1

    layers: list[dict[str, Any]] = []

    # Stem: Conv2d(1, width, kernel=(10,4), stride=(2,2), padding=(5,2))
    stem_h = conv_out(in_h, 10, 2, 5)
    stem_w = conv_out(in_w, 4, 2, 2)
    stem_params = 1 * width * 10 * 4
    stem_macs = stem_h * stem_w * width * 1 * 10 * 4
    layers.append({
        "name": "stem_conv", "type": "CONV_2D", "params": stem_params, "macs": stem_macs,
        "input_shape": [1, 1, in_h, in_w], "output_shape": [1, width, stem_h, stem_w],
    })
    total_params = stem_params + 2 * width  # + stem BatchNorm2d gamma/beta
    total_macs = stem_macs

    h, w = stem_h, stem_w
    for i in range(n_blocks):
        stride = 2 if i == 0 else 1
        out_h = conv_out(h, 3, stride, 1)
        out_w = conv_out(w, 3, stride, 1)

        dw_params = width * 3 * 3
        dw_macs = out_h * out_w * width * 3 * 3
        layers.append({
            "name": f"block{i+1}_depthwise", "type": "DEPTHWISE_CONV_2D",
            "params": dw_params, "macs": dw_macs,
            "input_shape": [1, width, h, w], "output_shape": [1, width, out_h, out_w],
        })

        pw_params = width * width * 1 * 1
        pw_macs = out_h * out_w * width * width
        layers.append({
            "name": f"block{i+1}_pointwise", "type": "CONV_2D",
            "params": pw_params, "macs": pw_macs,
            "input_shape": [1, width, out_h, out_w], "output_shape": [1, width, out_h, out_w],
        })

        total_params += dw_params + pw_params + 2 * 2 * width  # + 2x BatchNorm2d per block
        total_macs += dw_macs + pw_macs
        h, w = out_h, out_w

    pool_macs = h * w * width
    layers.append({
        "name": "avg_pool", "type": "AVERAGE_POOL_2D", "params": 0, "macs": pool_macs,
        "input_shape": [1, width, h, w], "output_shape": [1, width, 1, 1],
    })
    total_macs += pool_macs

    fc_params = width * n_classes + n_classes
    fc_macs = width * n_classes
    layers.append({
        "name": "fc", "type": "FULLY_CONNECTED", "params": fc_params, "macs": fc_macs,
        "input_shape": [1, width], "output_shape": [1, n_classes],
    })
    total_params += fc_params
    total_macs += fc_macs

    warnings = [
        "No exported quantized model (.tflite/.onnx) found under reference/tiny-kws/ -- "
        "total_macs and per-layer macs/params were derived analytically from "
        "src/model.py's DSCNN(width=160, n_blocks=4) definition, not read from an "
        "exported artifact. Cross-checked: analytical total_params "
        f"({total_params}) matches assets/metrics.json n_parameters ({reported_params}). "
        "dtype/quantized reflect the float32 training checkpoint. See "
        "docs/interface-requests.md.",
    ]
    if total_params != reported_params:
        warnings.append(
            f"MISMATCH: analytical total_params={total_params} != "
            f"assets/metrics.json n_parameters={reported_params}; architecture "
            "hyperparameters (width/n_blocks) may have changed."
        )

    return {
        "schema_version": "1.0.0",
        "tool": "profile_model",
        "source": "real",
        "run_id": new_run_id("profile_model"),
        "timestamp": utcnow_iso(),
        "model": {
            "name": "tiny_kws_ds_cnn",
            "architecture": "DS-CNN",
            "total_params": total_params,
            "total_macs": total_macs,
            "input_shape": [1, 1, in_h, in_w],
            "output_classes": n_classes,
            "dtype": "float32",
            "quantized": False,
            "quant_scheme": "none",
            "layers": layers,
            "unsupported_ops": [],
            "warnings": warnings,
        }
    }


def run_inspect_pipeline(repo_path: str, manifest_path: str,
                          corpus_dir: str = "") -> dict[str, Any]:
    """Return inspect_pipeline Output.pipeline from Person A's real manifest.

    The reference pipeline (reference/tiny-kws/src/common.py:LogMel) has no
    resample stage (input is already 16 kHz) and no pre-emphasis stage (not
    implemented at all), and computes STFT + mel filterbank in one fused
    torchaudio.transforms.MelSpectrogram call rather than separate framing/
    fft functions. contracts/inspect_pipeline.schema.json's stage name enum
    (resample, pre_emphasis, framing, fft, mel, log, normalize) assumes all
    seven are separable; only four of them are here. The other three are
    omitted rather than emitted with a null name (the schema enum doesn't
    allow that) — logged in docs/interface-requests.md.
    """
    manifest = _load_json(manifest_path) if manifest_path else _load_json("reference/pipeline_manifest.json")
    normalization = _load_json("reference/normalization.json")
    corpus_manifest = _load_json("reference/corpus_manifest.json")

    frontend = manifest["frontend"]
    stages_raw = manifest["stages"]

    def shape(stage_name: str) -> list[int]:
        return stages_raw[stage_name]["shape"]

    stages = [
        {
            "name": "fft", "order": 0,
            "input_shape": shape("post_input"), "output_shape": shape("post_power_spectrum"),
            "dtype": stages_raw["post_power_spectrum"]["dtype"],
            "constants": {
                "n_fft": frontend["n_fft"], "win_length": frontend["win_length"],
                "hop_length": frontend["hop_length"], "power": frontend["power"],
                "center": frontend["center"], "pad_mode": frontend["pad_mode"],
                "onesided": frontend["onesided"], "normalized": frontend["normalized"],
            },
            "source_file": "reference/tiny-kws/src/common.py",
            "function": "LogMel.forward (torchaudio.transforms.MelSpectrogram, pre-filterbank)",
        },
        {
            "name": "mel", "order": 1,
            "input_shape": shape("post_power_spectrum"), "output_shape": shape("post_mel"),
            "dtype": stages_raw["post_mel"]["dtype"],
            "constants": {
                "n_mels": frontend["n_mels"], "f_min": frontend["f_min"], "f_max": frontend["f_max"],
                "mel_scale": frontend["mel_scale"], "mel_norm": frontend["mel_norm"],
            },
            "source_file": "reference/tiny-kws/src/common.py",
            "function": "LogMel.forward (torchaudio.transforms.MelSpectrogram, filterbank)",
        },
        {
            "name": "log", "order": 2,
            "input_shape": shape("post_mel"), "output_shape": shape("post_log"),
            "dtype": stages_raw["post_log"]["dtype"],
            "constants": {"epsilon": frontend["log_epsilon"], "base": "natural"},
            "source_file": "reference/tiny-kws/src/common.py",
            "function": "LogMel.forward",
        },
        {
            "name": "normalize", "order": 3,
            "input_shape": shape("post_log"), "output_shape": shape("post_normalize"),
            "dtype": stages_raw["post_normalize"]["dtype"],
            "constants": {"mean": normalization["mean"], "std": normalization["std"]},
            "source_file": "reference/tiny-kws/src/common.py",
            "function": "normalize",
        },
    ]

    global_constants = {
        "sample_rate": manifest["input"]["sample_rate"],
        "n_fft": frontend["n_fft"],
        "win_length": frontend["win_length"],
        "hop_length": frontend["hop_length"],
        "n_mels": frontend["n_mels"],
        "n_frames": frontend["expected_frames"],
        "log_epsilon": frontend["log_epsilon"],
        "norm_mean": normalization["mean"],
        "norm_std": normalization["std"],
    }

    corpus_manifest_bytes = (_REPO_ROOT / "reference" / "corpus_manifest.json").read_bytes()
    corpus = {
        "n_samples": corpus_manifest["selection"]["count"],
        "corpus_id": f"{corpus_manifest['dataset']['name']}-{corpus_manifest['selection']['strategy']}"
                     f"-seed{corpus_manifest['selection']['seed']}",
        # No single aggregate hash in corpus_manifest.json (each sample has
        # its own sha256) — hash the manifest file itself as a stable proxy.
        "sha256": hashlib.sha256(corpus_manifest_bytes).hexdigest(),
    }

    return {
        "schema_version": "1.0.0",
        "tool": "inspect_pipeline",
        "source": "real",
        "run_id": new_run_id("inspect_pipeline"),
        "timestamp": utcnow_iso(),
        "pipeline": {
            "domain": "kws",
            "n_stages": len(stages),
            "stages": stages,
            "global_constants": global_constants,
            "corpus": corpus,
            "warnings": [
                "Real pipeline has no resample stage (input already 16 kHz) and no "
                "pre_emphasis stage (not implemented in reference/tiny-kws/src/common.py) "
                "-- only 4 of the schema's 7 possible stages are present "
                "(fft, mel, log, normalize). See docs/interface-requests.md.",
            ],
        }
    }


def run_run_parity_test(gate: str, attempt: int, corpus_dir: str,
                         ref_pipeline_path: str, impl_pipeline_path: str,
                         run_id: str = "") -> dict[str, Any]:
    """Return run_parity_test Output payload."""
    raise NotImplementedError(
        "waiting on Person A: implement run_parity_test in pipeline_adapter.py. "
        "Needs: compiled impl_pipeline_path binary or shared lib, corpus_dir of "
        ".wav files, and reference tensors from dump_reference.py. "
        "Must produce per-stage max_abs_diff, mean_abs_diff, cosine_similarity."
    )


def run_quantize_model(model_file: str, representative_data_dir: str,
                        n_calibration_samples: int = 100,
                        output_dir: str = "") -> dict[str, Any]:
    """Return quantize_model Output payload."""
    raise NotImplementedError(
        "waiting on Person A: implement quantize_model in pipeline_adapter.py. "
        "Needs: float32 SavedModel at model_file, representative .npy/.wav "
        "samples in representative_data_dir for PTQ calibration."
    )
