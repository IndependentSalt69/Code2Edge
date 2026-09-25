# Code2Edge Workflow: From Research to Proven Edge Silicon

**Tagline:** Deployment with proof.  
**Core Innovation:** A Bob-native workflow that traces an ML repository's complete inference pipeline, generates a constrained-device implementation, and proves mathematical correctness through stage-wise differential parity before and after real hardware deployment.

---

## 1. End-to-End Workflow Stages

```text
[ Stage 1: Ingestion ]
    └── Frozen reference repository (reference/tiny-kws/)
    └── Pretrained weights (checkpoints/best.pt)

[ Stage 2: Host Parity Harness ]
    └── Non-invasive PyTorch forward hooks
    └── Capture intermediate tensors S0–S6 on host
    └── Dump golden test vectors to reference/golden/

[ Stage 3: Edge Generation (Person A) ]
    └── Generate CMSIS-DSP Mel feature extraction (src/pipeline/)
    └── Generate quantized int8 model array (src/pipeline/model_data.h)
    └── Verify host C parity vs PyTorch reference

[ Gate 1: Host Parity Gate (Mandatory) ]
    ├── PASS: Unblocks embedded compilation
    └── FAIL: Triggers automated Bob repair of generated C logic

[ Stage 4: Target Firmware Integration (Person B) ]
    └── Embed generated pipeline into STM32U585 firmware (src/firmware/)
    └── Compile via arduino-cli (arduino:zephyr:unoq)
    └── Flash to Arduino UNO Q STM32U585 MCU

[ Gate 2: On-Device Differential Parity (Person B) ]
    ├── Execute test corpus on physical hardware
    ├── Stream or compare on-device stage tensors vs golden host tensors
    ├── PASS: 100% classification agreement & within stage error tolerance
    └── FAIL: Pinpoint first divergent stage on hardware (S1..S6)

[ Stage 5: Target Benchmarking & Reporting ]
    └── Hardware cycle measurement (DWT_CYCCNT) -> inference latency (ms)
    └── Linker map analysis -> Flash & SRAM usage
    └── Produce predicted-vs-measured table in evidence/benchmarks/
```

---

## 2. Roles & Separation of Concerns

### Person A (Host Pipeline & Edge Code Generation)
- Owns reference ingestion and host parity harness (`src/inference/`, `src/parity/`).
- Owns edge C code generation for feature extraction and neural network (`src/pipeline/`).
- Enforces Host Parity Gate before handing off to Person B.

### Person B (Embedded Firmware, Target Toolchain & Verification)
- Owns STM32U585 firmware environment and toolchain integration (`src/firmware/`, `tools/target/`).
- Owns Day-1 dummy-model hardware proof and MPU ↔ MCU bridge verification.
- Owns on-device parity execution and authoritative hardware benchmarking (`contracts/target/`).
