# Person B Execution Plan: Embedded & Hardware Track

**Owner:** Person B (Hardware & Embedded Lead)  
**Target:** STM32U585 MCU on Arduino UNO Q  
**Document Status:** Actionable Engineering Execution Plan

---

## 1. Primary Mission

Prove that the keyword spotting model + preprocessing pipeline executes correctly on the **STM32U585** within its 786 KB SRAM / 2 MB Flash constraints, while capturing authoritative, reproducible latency and memory measurements.

---

## 2. Day-1 Priority: Dummy-Model Hardware Proof

> [!IMPORTANT]
> The primary Day-1 objective is to eliminate all firmware, toolchain, and communication uncertainty **BEFORE** Person A's generated C++ pipeline is delivered.

### Milestone 1.1: Dummy Model End-to-End Pipeline
- **Model:** Minimal 2-layer MLP or dummy Conv model (or sine-wave model) quantized to int8 / float32.
- **Workflow to Prove:**
  1. Compile firmware using `arduino-cli` with `arduino:zephyr:unoq`.
  2. Flash binary to STM32U585 MCU (via ADB or bootloader).
  3. Execute inference on a known dummy tensor.
  4. Transmit output over serial/UART.
  5. Read back and verify output deterministically across 50 iterations.
- **Success Criteria:** Zero crashes, deterministic numeric output, repeatable execution loop.

---

## 3. Sequential Dependency Gates

To ensure strict engineering integrity, Person B follows a gated pipeline:

```text
  [ Milestone 1: Toolchain & Dummy Model ]
                     ↓
  [ Milestone 2: MPU <-> MCU Bridge Verification ]
                     ↓
          Person A Generated Code
                     ↓
  [ Gate 3: Host Parity Check ] ------------------- FAIL ----> Bob Repair (Person A)
                     ↓ PASS
  [ Milestone 4: MCU Firmware Integration ]
                     ↓
  [ Gate 5: On-Device Differential Parity ] ------- FAIL ----> Bob Repair (Joint)
                     ↓ PASS
  [ Milestone 6: Authoritative Benchmark ]
                     ↓
  [ Final: Predicted vs Measured Report ]
```

### Gate Rules:
1. **No Hardware Deployment without Host Parity:** Person B will NOT build or flash Person A's generated code to target silicon until host-side parity passes with zero tolerance violations across stages S0–S6.
2. **First Divergence Isolation:** If on-device parity fails, Person B isolates whether the error originated from preprocessing (S1–S3) or neural weights/quantization (S4a–S5).

---

## 4. Phase-by-Phase Roadmap

### Phase 1: Environment & Toolchain Readiness (Immediate)
- [ ] Install `arduino:zephyr` core in `arduino-cli`.
- [ ] Verify `arm-zephyr-eabi-gcc` compiler presence and flags.
- [ ] Create `src/firmware/` skeleton with `CMakeLists.txt` or `.ino` sketch structure.
- [ ] Validate build of baseline empty sketch for Arduino UNO Q.

### Phase 2: MPU ↔ MCU Bridge Verification
- [ ] Implement `tools/target/bridge_test.py` on host / Linux MPU side.
- [ ] Implement echo handler in STM32 firmware.
- [ ] Verify loopback transmission of 100 8-byte frames without frame drops.
- [ ] Test 32 KB chunk transmission (simulating 1-second raw audio waveform).
- [ ] *Fallback Switch:* If UART bridge is unreliable, activate offline ROM fixture harness (`src/firmware/golden_fixtures.h`).

### Phase 3: Dummy Model Benchmark Tooling (`benchmark_target()`)
- [ ] Implement ARM Cortex-M33 DWT cycle counter in `src/firmware/timer.c`.
- [ ] Implement Flash and SRAM consumption extraction from GCC `.map` file (`tools/target/parse_map.py`).
- [ ] Implement `tools/target/benchmark_target.py` adhering to `contracts/target/benchmark-result.schema.json`.
- [ ] Validate schema output format with mock execution data.

### Phase 4: Integration of Person A Deliverables
- [ ] Receive generated preprocessing headers and model binary from Person A.
- [ ] Confirm host parity passed via `evidence/parity/host_parity_report.json`.
- [ ] Integrate into `src/firmware/` build.
- [ ] Measure tensor arena requirement and allocate statically.

### Phase 5: On-Device Parity & Final Benchmarking
- [ ] Run test corpus through device parity harness (`tools/target/run_device_parity.py`).
- [ ] Compare logits against reference Python tensors (`tests/parity/fixtures/`).
- [ ] Verify 100% classification agreement on test set.
- [ ] Execute `benchmark_target()` for 50 iterations; produce final `predicted_vs_measured.md` report.

---

## 5. Contingency & Fallback Matrix

| Failure Mode | Impact | Fallback Procedure |
|---|---|---|
| **No physical UNO Q board available** | Hardware execution blocked | Run on STM32U585 QEMU / Renode emulator or compile as native Cortex-M33 target; generate predicted memory budgets |
| **`arduino:zephyr` fails to flash via ADB** | Flashing blocked | Extract compiled ELF/BIN and flash via OpenOCD / SWD / DFU mode |
| **MPU UART bridge drops packets** | Parity data transfer corrupted | Switch to Offline Fixture Mode: compile golden test WAVs directly into STM32 Flash |
| **Tensor arena exceeds SRAM** | Out of memory | Prune TFLM ops resolver to `MicroMutableOpResolver`; reduce arena buffer to match activation high-water mark |
