#!/usr/bin/env python3
"""
Code2Edge Host Differential Parity Engine.
Executes the native-compiled C preprocessing pipeline (src/pipeline/feature_extraction.c)
on host CPU and performs stage-by-stage numerical parity verification against immutable
frozen PyTorch reference artifacts (reference/artifacts/sample_0001/*.npy).

Stage Mapping:
  S0: Raw Audio Waveform       (post_input.npy)
  S1: STFT Power Spectrum      (post_power_spectrum.npy)
  S1_mel: Mel Energy Filterbank (post_mel.npy)
  S2: Log-Mel Compression      (post_log.npy)
  S3: Normalized Log-Mel       (post_normalize.npy)
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_c_dll import compile_c_dll  # noqa: E402

ARTIFACTS_DIR = REPO_ROOT / "reference" / "artifacts"
SMOKE_SAMPLE_DIR = ARTIFACTS_DIR / "sample_0001"
OUTPUT_REPORT = REPO_ROOT / "evidence" / "parity" / "host_parity_report.json"

# Stage tolerances for float32 C implementation vs PyTorch reference
STAGE_TOLERANCES = {
    "S0_raw_waveform": {
        "max_abs_diff": 1e-6,
        "mean_abs_diff": 1e-7,
        "min_cosine": 0.999999,
        "description": "Raw 16kHz PCM audio waveform [-1.0, 1.0]",
    },
    "S1_power_spectrum": {
        "max_abs_diff": 1e-2,
        "mean_abs_diff": 1e-4,
        "min_cosine": 0.999999,
        "description": "Periodic Hann STFT power spectrogram (201 bins x 101 frames)",
    },
    "S1_mel_energy": {
        "max_abs_diff": 1e-2,
        "mean_abs_diff": 2e-4,
        "min_cosine": 0.999999,
        "description": "64-band HTK Mel filterbank dot-product energy representation",
    },
    "S2_log_mel": {
        "max_abs_diff": 5e-3,
        "mean_abs_diff": 5e-4,
        "min_cosine": 0.999990,
        "description": "Log compression with eps=1e-6",
    },
    "S3_normalized_features": {
        "max_abs_diff": 1e-3,
        "mean_abs_diff": 1e-4,
        "min_cosine": 0.999990,
        "description": "Global mean/std normalized Log-Mel tensor",
    },
}


class NativeCPipeline:
    """Wrapper around compiled C feature extraction dynamic library."""

    def __init__(self, dll_path: Path | None = None):
        if dll_path is None or not dll_path.exists():
            dll_path = compile_c_dll()

        self.dll_path = dll_path
        self._dll = ctypes.CDLL(str(dll_path))

        # Setup C function signatures
        self._init = self._dll.feature_extraction_init
        self._init.argtypes = []
        self._init.restype = None

        self._stages_f32 = self._dll.feature_extraction_stages_f32
        self._stages_f32.argtypes = [
            ctypes.c_void_p,  # audio_f32_16k
            ctypes.c_void_p,  # s0_input_out
            ctypes.c_void_p,  # s1_power_out
            ctypes.c_void_p,  # s1_mel_out
            ctypes.c_void_p,  # s2_log_out
            ctypes.c_void_p,  # s3_norm_out
        ]
        self._stages_f32.restype = None

        self._checksum = self._dll.feature_extraction_compute_checksum
        self._checksum.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._checksum.restype = ctypes.c_uint32

        # Initialize
        self._init()

    def run_stages(self, audio_pcm_f32: np.ndarray) -> dict[str, np.ndarray]:
        """Runs the native C feature extraction stages on float32 audio input."""
        if audio_pcm_f32.shape != (16000,):
            raise ValueError(
                f"Expected audio shape (16000,), got {audio_pcm_f32.shape}"
            )

        audio_c = np.ascontiguousarray(audio_pcm_f32, dtype=np.float32)

        s0_out = np.zeros(16000, dtype=np.float32)
        s1_power_out = np.zeros((201, 101), dtype=np.float32)
        s1_mel_out = np.zeros((64, 101), dtype=np.float32)
        s2_log_out = np.zeros((64, 101), dtype=np.float32)
        s3_norm_out = np.zeros((64, 101), dtype=np.float32)

        self._stages_f32(
            audio_c.ctypes.data,
            s0_out.ctypes.data,
            s1_power_out.ctypes.data,
            s1_mel_out.ctypes.data,
            s2_log_out.ctypes.data,
            s3_norm_out.ctypes.data,
        )

        return {
            "S0_raw_waveform": s0_out,
            "S1_power_spectrum": s1_power_out,
            "S1_mel_energy": s1_mel_out,
            "S2_log_mel": s2_log_out,
            "S3_normalized_features": s3_norm_out,
        }

    def compute_checksum(self, features: np.ndarray) -> int:
        features_c = np.ascontiguousarray(features, dtype=np.float32)
        return int(self._checksum(features_c.ctypes.data, features_c.size))


def compare_stage_tensors(
    stage_name: str,
    ref_tensor: np.ndarray,
    impl_tensor: np.ndarray,
    tol: dict[str, Any],
) -> dict[str, Any]:
    """
    Performs rigorous differential comparison between ground truth and C output.
    Reports shape, dtype, max/mean absolute error, relative L2 error, cosine similarity,
    exact mismatch counts, and first divergent index.
    """
    ref_sq = ref_tensor.squeeze().astype(np.float64)
    impl_sq = impl_tensor.squeeze().astype(np.float64)

    # 1. Shape comparison
    shape_match = bool(ref_sq.shape == impl_sq.shape)
    dtype_match = bool(ref_tensor.dtype == impl_tensor.dtype)

    if not shape_match:
        return {
            "stage": stage_name,
            "status": "FAIL",
            "error": "SHAPE_MISMATCH",
            "shape_ref": list(ref_tensor.shape),
            "shape_impl": list(impl_tensor.shape),
            "dtype_ref": str(ref_tensor.dtype),
            "dtype_impl": str(impl_tensor.dtype),
        }

    # 2. Numerical differences
    diff = np.abs(ref_sq - impl_sq)
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))

    # Vector metrics
    a = ref_sq.ravel()
    b = impl_sq.ravel()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denom = norm_a * norm_b

    cosine_sim = float(np.dot(a, b) / denom) if denom > 0 else 1.0
    relative_l2 = float(np.linalg.norm(a - b) / norm_a) if norm_a > 0 else 0.0

    # Divergence analysis
    worst_flat_idx = int(np.argmax(diff))
    worst_multi_idx = [
        int(i) for i in np.unravel_index(worst_flat_idx, diff.shape)
    ]
    ref_val_at_worst = float(ref_sq[tuple(worst_multi_idx)])
    impl_val_at_worst = float(impl_sq[tuple(worst_multi_idx)])

    # First divergent index where abs error exceeds tolerance
    divergent_indices = np.argwhere(diff > tol["max_abs_diff"])
    mismatch_count = int(len(divergent_indices))
    first_divergent_idx = (
        [int(i) for i in divergent_indices[0]]
        if mismatch_count > 0
        else None
    )

    passed = (
        max_abs_diff <= tol["max_abs_diff"]
        and mean_abs_diff <= tol["mean_abs_diff"]
        and cosine_sim >= tol["min_cosine"]
        and shape_match
        and dtype_match
    )

    return {
        "stage": stage_name,
        "status": "PASS" if passed else "FAIL",
        "description": tol.get("description", ""),
        "shape_ref": list(ref_tensor.shape),
        "shape_impl": list(impl_tensor.shape),
        "dtype_ref": str(ref_tensor.dtype),
        "dtype_impl": str(impl_tensor.dtype),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_l2": relative_l2,
        "cosine_similarity": cosine_sim,
        "mismatch_count": mismatch_count,
        "tolerance": tol,
        "worst_index": worst_multi_idx,
        "ref_val_at_worst": ref_val_at_worst,
        "impl_val_at_worst": impl_val_at_worst,
        "first_divergent_index": first_divergent_idx,
    }


def evaluate_sample_parity(
    sample_dir: Path, c_pipeline: NativeCPipeline
) -> dict[str, Any]:
    """Runs host parity check for a single directory containing frozen .npy artifacts."""
    if not sample_dir.exists():
        raise FileNotFoundError(
            f"Reference artifact dir not found: {sample_dir}"
        )

    # 1. Load frozen PyTorch ground-truth tensors
    ref_input = np.load(sample_dir / "post_input.npy")
    ref_power = np.load(sample_dir / "post_power_spectrum.npy")
    ref_mel = np.load(sample_dir / "post_mel.npy")
    ref_log = np.load(sample_dir / "post_log.npy")
    ref_norm = np.load(sample_dir / "post_normalize.npy")

    # 2. Execute actual compiled C pipeline
    c_stages = c_pipeline.run_stages(ref_input)
    checksum = c_pipeline.compute_checksum(
        c_stages["S3_normalized_features"]
    )

    # 3. Stage-wise comparison
    stage_evals = [
        compare_stage_tensors(
            "S0_raw_waveform",
            ref_input,
            c_stages["S0_raw_waveform"],
            STAGE_TOLERANCES["S0_raw_waveform"],
        ),
        compare_stage_tensors(
            "S1_power_spectrum",
            ref_power,
            c_stages["S1_power_spectrum"],
            STAGE_TOLERANCES["S1_power_spectrum"],
        ),
        compare_stage_tensors(
            "S1_mel_energy",
            ref_mel,
            c_stages["S1_mel_energy"],
            STAGE_TOLERANCES["S1_mel_energy"],
        ),
        compare_stage_tensors(
            "S2_log_mel",
            ref_log,
            c_stages["S2_log_mel"],
            STAGE_TOLERANCES["S2_log_mel"],
        ),
        compare_stage_tensors(
            "S3_normalized_features",
            ref_norm,
            c_stages["S3_normalized_features"],
            STAGE_TOLERANCES["S3_normalized_features"],
        ),
    ]

    all_pass = all(s["status"] == "PASS" for s in stage_evals)
    first_divergent = next(
        (s["stage"] for s in stage_evals if s["status"] != "PASS"), None
    )

    return {
        "sample_id": sample_dir.name,
        "sample_path": str(sample_dir),
        "status": "PASS" if all_pass else "FAIL",
        "output_checksum_fnv1a": hex(checksum),
        "first_divergent_stage": first_divergent,
        "stages": stage_evals,
    }


def run_host_parity_suite(
    sample_dir: Path = SMOKE_SAMPLE_DIR,
    corpus_dir: Path | None = None,
    output_path: Path = OUTPUT_REPORT,
) -> dict[str, Any]:
    """
    Main entry point for host parity suite.
    Compiles native C pipeline, runs against frozen reference, and generates evidence report.
    """
    c_pipeline = NativeCPipeline()

    sample_dirs: list[Path] = []
    if corpus_dir and corpus_dir.exists():
        # Sweep corpus directory for sample subdirs containing post_input.npy
        for p in sorted(corpus_dir.iterdir()):
            if p.is_dir() and (p / "post_input.npy").exists():
                sample_dirs.append(p)
    else:
        sample_dirs.append(sample_dir)

    sample_results = []
    for sdir in sample_dirs:
        res = evaluate_sample_parity(sdir, c_pipeline)
        sample_results.append(res)

    all_passed = all(r["status"] == "PASS" for r in sample_results)
    is_smoke_only = len(sample_results) == 1 and sample_dirs[0] == SMOKE_SAMPLE_DIR

    # Aggregate summary metrics
    total_stages = sum(len(r["stages"]) for r in sample_results)
    passed_stages = sum(
        sum(1 for s in r["stages"] if s["status"] == "PASS")
        for r in sample_results
    )
    failed_stages = total_stages - passed_stages

    worst_max_abs = max(
        max(s.get("max_abs_diff", 0.0) for s in r["stages"])
        for r in sample_results
    )
    worst_mean_abs = max(
        max(s.get("mean_abs_diff", 0.0) for s in r["stages"])
        for r in sample_results
    )
    worst_cosine = min(
        min(s.get("cosine_similarity", 1.0) for s in r["stages"])
        for r in sample_results
    )

    report = {
        "schema_version": "1.0.0",
        "contract": "contracts/target/device-parity.json",
        "gate": "host_differential_parity",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "toolchain": {
            "c_source": "src/pipeline/feature_extraction.c",
            "c_header": "src/pipeline/feature_extraction.h",
            "compiled_library": str(c_pipeline.dll_path.relative_to(REPO_ROOT)),
        },
        "coverage": {
            "mode": "smoke_sample" if is_smoke_only else "corpus_sweep",
            "sample_count": len(sample_results),
            "samples_evaluated": [r["sample_id"] for r in sample_results],
            "full_corpus_covered": not is_smoke_only,
            "corpus_manifest_total_samples": 500,
            "parity_status_smoke": "PASS" if all_passed else "FAIL",
            "parity_status_full_corpus": "PENDING" if is_smoke_only else ("PASS" if all_passed else "FAIL"),
        },
        "summary": {
            "overall_status": "PASS" if all_passed else "FAIL",
            "total_stages_evaluated": total_stages,
            "passed_stages": passed_stages,
            "failed_stages": failed_stages,
            "worst_max_abs_diff": worst_max_abs,
            "worst_mean_abs_diff": worst_mean_abs,
            "worst_cosine_similarity": worst_cosine,
        },
        "sample_reports": sample_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Code2Edge Real Host Differential Parity Engine"
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=SMOKE_SAMPLE_DIR,
        help="Path to single frozen sample directory (default: reference/artifacts/sample_0001)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Path to corpus directory containing multiple sample subdirectories for full sweep",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_REPORT,
        help="Path to save output JSON report",
    )
    args = parser.parse_args()

    print("================================================================")
    print(" Code2Edge Stage-Wise Real Host Differential Parity Verification")
    print(" Native C Pipeline: src/pipeline/feature_extraction.c")
    print(" Ground Truth:      Immutable PyTorch Reference Artifacts (.npy)")
    print("================================================================")

    try:
        report = run_host_parity_suite(
            sample_dir=args.sample_dir,
            corpus_dir=args.corpus_dir,
            output_path=args.output,
        )
    except Exception as e:
        print(f"\n[ERROR] Host parity test execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    for s_rep in report["sample_reports"]:
        print(f"\n--- Sample: {s_rep['sample_id']} (Checksum: {s_rep['output_checksum_fnv1a']}) ---")
        for st in s_rep["stages"]:
            status_tag = f"[{st['status']:4s}]"
            print(
                f"{status_tag} Stage {st['stage']:24s} | "
                f"Max Abs: {st['max_abs_diff']:.6e} | "
                f"Mean Abs: {st['mean_abs_diff']:.6e} | "
                f"Rel L2: {st['relative_l2']:.6e} | "
                f"Cosine: {st['cosine_similarity']:.8f}"
            )
            if st["status"] != "PASS":
                print(
                    f"       -> FIRST DIVERGENT INDEX: {st['first_divergent_index']}, "
                    f"Ref: {st['ref_val_at_worst']:.6e}, Impl: {st['impl_val_at_worst']:.6e}"
                )

    print("\n----------------------------------------------------------------")
    print(
        f"Summary: {report['summary']['passed_stages']}/{report['summary']['total_stages_evaluated']} "
        f"stages passed. (Coverage: {report['coverage']['mode']}, {report['coverage']['sample_count']} sample)"
    )
    print(f"Smoke Parity Gate:       {report['coverage']['parity_status_smoke']}")
    print(f"Full Corpus Parity Gate: {report['coverage']['parity_status_full_corpus']}")
    print(f"Evidence Report:         {args.output}")
    print("================================================================")

    if report["summary"]["overall_status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
