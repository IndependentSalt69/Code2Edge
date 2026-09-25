"""
mcp_server/adapters/pipeline_adapter.py

Adapter for Person A's pipeline and parity outputs.

When CODE2EDGE_*_MODE=real, the MCP tools call functions here.
These stubs raise NotImplementedError with a clear message about what is needed
so the agent sees a structured ERROR rather than a silent fallback to mock.

Person A: implement the real functions here once pipeline/ is ready.
DO NOT write any files into src/pipeline/, reference/, or tools/.
"""
from __future__ import annotations

from typing import Any


def run_profile_model(repo_path: str, model_file: str,
                      labels_file: str = "", config_file: str = "") -> dict[str, Any]:
    """Return profile_model Output payload for the given repository."""
    raise NotImplementedError(
        "waiting on Person A: implement profile_model in pipeline_adapter.py. "
        "Needs: access to repo_path, model weights, and config to extract "
        "architecture, layer list, MACs, param count, dtype, quantization scheme."
    )


def run_inspect_pipeline(repo_path: str, manifest_path: str,
                          corpus_dir: str = "") -> dict[str, Any]:
    """Return inspect_pipeline Output payload."""
    raise NotImplementedError(
        "waiting on Person A: implement inspect_pipeline in pipeline_adapter.py. "
        "Needs: dump_reference.py manifest.json at manifest_path containing "
        "observed runtime constants, stage shapes, and corpus sha256."
    )


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
