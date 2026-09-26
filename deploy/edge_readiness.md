# Edge Readiness Report

Run: `run-7d942761`  Data: **real**

## Fit
- fits: **True**
- SRAM budget: 786.0 KB, used: 188.0 KB, headroom: 598.0 KB
- Flash budget: 2048.0 KB, model: 168.2 KB (real, documented in `src/pipeline/model_data.h`)
- Arena: 162.66 KB, feature buffer: 25.3 KB
- Unsupported ops: none
- Warnings: ["Model file 'model_data_int8.tflite' not found on disk; used the frozen int8 model size documented in src/pipeline/model_data.h (168.2 KB) instead."]

## Model
- DS-CNN, 119372 params, 58650560 MACs, float32
  - note: No exported quantized model (.tflite/.onnx) found under reference/tiny-kws/ -- total_macs and per-layer macs/params were derived analytically from src/model.py's DSCNN(width=160, n_blocks=4) definition, not read from an exported artifact. Cross-checked: analytical total_params (119372) matches assets/metrics.json n_parameters (119372). dtype/quantized reflect the float32 training checkpoint. See docs/interface-requests.md.

## Pipeline constants (4 stages, domain=kws)
- sample_rate: 16000
- n_fft: 400
- win_length: 400
- hop_length: 160
- n_mels: 64
- n_frames: 101
- log_epsilon: 1e-06
- norm_mean: -6.902360439300537
- norm_std: 4.81721305847168
- note: Real pipeline has no resample stage (input already 16 kHz) and no pre_emphasis stage (not implemented in reference/tiny-kws/src/common.py) -- only 4 of the schema's 7 possible stages are present (fft, mel, log, normalize). See docs/interface-requests.md.

## Risks
- Host parity is real and PASSING (500 samples, 4 stages, 2500/2500) -- see evidence/parity/host_parity_report.json.
- Device parity gate will use mock_scenario="pass" for this run: no board is physically connected in this session. Device parity real evidence exists but is thin (1 sample, 1 stage only).
- benchmark_target will also be mocked for this run for the same reason: no real DS-CNN physical latency benchmark exists yet, only an infra-validation run.
- A guide for completing the real device parity + benchmark steps on physical hardware will be produced alongside this run.
