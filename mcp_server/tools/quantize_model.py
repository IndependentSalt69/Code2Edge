"""mcp_server/tools/quantize_model.py"""
from __future__ import annotations

from mcp_server._mock_loader import load_mock
from mcp_server._mode import get_mode
from mcp_server._schema import validate_output
from mcp_server._ids import new_run_id, utcnow_iso

_TOOL = "quantize_model"


async def run(model_file: str, representative_data_dir: str,
              n_calibration_samples: int = 100,
              output_dir: str = "workflow/runs/quantized") -> dict:
    mode = get_mode(_TOOL)
    run_id = new_run_id(_TOOL)

    if mode == "mock":
        payload = load_mock(_TOOL, run_id=run_id)
        payload["quantization"]["output_path"] = f"{output_dir}/ds_cnn_kws_int8.tflite"
        payload["quantization"]["n_calibration_samples"] = n_calibration_samples
    else:
        from mcp_server.adapters import pipeline_adapter
        try:
            payload = pipeline_adapter.run_quantize_model(
                model_file=model_file,
                representative_data_dir=representative_data_dir,
                n_calibration_samples=n_calibration_samples,
                output_dir=output_dir,
            )
        except NotImplementedError as e:
            return {
                "schema_version": "1.0.0", "tool": _TOOL, "source": "real",
                "run_id": run_id, "timestamp": utcnow_iso(),
                "status": "ERROR", "error_message": str(e),
            }

    validate_output(_TOOL, payload)
    return payload
