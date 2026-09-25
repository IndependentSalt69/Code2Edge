# Code2Edge Target Benchmark Harness

A standalone, zero-heap CPU cycle benchmark harness for the **STMicroelectronics STM32U585** (ARM Cortex-M33) on the **Arduino UNO Q**.

---

## 1. Purpose & Scope

This harness provides the authoritative on-device timing and memory measurement infrastructure for the Code2Edge project:
- **Target Platform:** Arduino UNO Q (STM32U585 MCU subsystem)
- **FQBN:** `arduino:zephyr:unoq`
- **Measurement Engine:** Hardware cycle counter via ARM Cortex-M33 Data Watchpoint and Trace (`DWT_CYCCNT`)
- **Scope:** Measurement infrastructure only. Contains a deterministic validation arithmetic kernel to verify the timing machinery before plugging in the real Keyword Spotting (KWS) inference pipeline.

---

## 2. Hardware Cycle Timing Mechanism

### How DWT Cycle Counting Works
On the ARMv8-M Cortex-M33 core, the DWT unit provides a 32-bit hardware register incrementing every CPU clock cycle (160 MHz on STM32U585).

The harness initializes the counter via direct memory-mapped register access:
1. **`DEMCR` (Address `0xE000EDFC`):** Sets bit 24 (`TRCENA`) to enable global debug and trace peripherals.
2. **`DWT_CYCCNT` (Address `0xE0001004`):** Resets the cycle count register to `0`.
3. **`DWT_CTRL` (Address `0xE0001000`):** Sets bit 0 (`CYCCNTENA`) to enable active cycle counting.
4. **Active Verification:** The initialization routine reads the counter across a bounded loop of NOP instructions to ensure the register is actively incrementing before running any benchmark.

### Frequency Resolution
- **Core Clock:** 160 MHz ($160,000,000\text{ Hz}$).
- **Resolution:** $1\text{ cycle} = 6.25\text{ nanoseconds}$.
- **Microsecond Conversion:** $\text{Time } (\mu\text{s}) = \frac{\text{Cycles}}{160.0}$.

---

## 3. Benchmark API & Statistics

```cpp
void benchmark_target(
    const char* name,
    void (*fn)(),
    size_t iterations = 50,
    size_t warmup = 5
);
```

### Protocol & Guarantees
- **Warm-Up Runs:** Executes `warmup` unmeasured iterations (default: 5) to warm the instruction cache, branch predictors, and pipeline state.
- **Measured Iterations:** Executes `iterations` measured passes (default: 50), recording elapsed cycles into a pre-allocated static buffer.
- **Statistical Output:** Computes minimum, maximum, sample average ($\mu$), and sample standard deviation ($\sigma$):
  $$\sigma = \sqrt{\frac{\sum_{i=1}^{N} (c_i - \mu)^2}{N - 1}}$$
- **Zero Dynamic Allocation:** Uses static memory arrays (`s_cycles_record`) to prevent heap fragmentation or allocator latency interference during measurement.

---

## 4. Deterministic Validation Workload

The harness includes `deterministic_kernel()`, a small arithmetic benchmark kernel:
- Mixes and hashes 64 32-bit integers using linear congruential multiplier constants (`2654435761UL`), Golden Ratio offsets (`0x9E3779B9UL`), and non-linear bit rotations.
- Writes the final accumulator to `volatile uint32_t g_workload_checksum` to prevent compiler dead-code elimination.
- Verifies that the CPU pipeline executes arithmetic operations deterministically across repeated runs.

---

## 5. Build, Upload & Monitor Commands

### Compile
```bash
arduino-cli compile --fqbn arduino:zephyr:unoq tests/firmware/benchmark_harness
```

### Upload (via COM3)
```bash
arduino-cli upload -p COM3 --fqbn arduino:zephyr:unoq tests/firmware/benchmark_harness
```

### Monitor Serial Output (115200 baud)
```bash
arduino-cli monitor -p COM3 --config baudrate=115200
```

---

## 6. Expected Serial Output

```text
CODE2EDGE_BENCHMARK_START
================================================================
 Benchmark Workload: deterministic_kernel
 Target MCU:         STMicroelectronics STM32U585 (ARM Cortex-M33)
 Board FQBN:         arduino:zephyr:unoq
 CPU Clock:          160.0 MHz
 Iterations:         50 (Warmup: 5)
----------------------------------------------------------------
 Min Latency:        XXXX cycles (X.XXXX us)
 Max Latency:        XXXX cycles (X.XXXX us)
 Avg Latency:        XXXX.XX cycles (X.XXXX us)
 StdDev Latency:     X.XX cycles
 Kernel Checksum:    0xXXXXXXXX
================================================================
BENCHMARK_JSON={"name":"deterministic_kernel","iterations":50,"min_cycles":XXXX,"max_cycles":XXXX,"avg_cycles":XXXX.XX,"stddev_cycles":X.XX,"cpu_hz":160000000,"min_us":X.XXXX,"max_us":X.XXXX,"avg_us":X.XXXX,"checksum":XXXXXXXX}
CODE2EDGE_BENCHMARK_END
```

*Note: Sending character `'B'` or `'R'` over the serial connection triggers a fresh 50-iteration benchmark run.*

---

## 7. Memory Accounting & Map File Extraction

High-level compile summaries (e.g. preliminary test build values such as ~87 KB Flash / ~35 KB RAM) represent approximate binary sizing and must not be treated as authoritative final measurements. The actual compile result and runtime figures from the hardware run will become the recorded result.

Authoritative memory profiling extracts section-by-section data from the GCC linker map file:

### Linker Map Analysis
When compiling with `arduino-cli --verbose`, the build produces an ELF binary and `.map` file in the build cache:
- **Code Flash (`.text`):** Core application instructions, Zephyr kernel, and CMSIS math routines.
- **Constant Flash (`.rodata`):** Read-only lookup tables, Mel filterbank matrices, and neural network weights.
- **Initialized SRAM (`.data`):** Initialized variables copied from Flash to SRAM at reset.
- **Uninitialized SRAM (`.bss`):** Static tensor arena (`g_tensor_arena`), feature buffers, and runtime state.
- **Static Footprint Formula:**
  $$\text{Flash Used} = \text{sizeof}(.text) + \text{sizeof}(.rodata) + \text{sizeof}(.data)$$
  $$\text{SRAM Used} = \text{sizeof}(.data) + \text{sizeof}(.bss)$$

The `tools/target/parse_map.py` script automatically parses these sections and outputs metrics conforming to [`contracts/target/benchmark-result.schema.json`](../../../contracts/target/benchmark-result.schema.json).

---

## 8. Integration Roadmap

1. **Step 1 (Physical execution verified):** Validated DWT cycle counting and timing statistics on physical STM32U585 hardware ($990 \pm 0.00$ cycles, $6.1875\text{ }\mu\text{s}$ @ 160 MHz). Recorded in [`evidence/benchmarks/benchmark_harness_physical_run.json`](../../../evidence/benchmarks/benchmark_harness_physical_run.json).
2. **Step 2 (Next):** Wrap Person A's generated C feature extraction (`feature_extraction_run()`) in `benchmark_target("mel_spectrogram", ...)` to measure preprocessing latency.
3. **Step 3 (Final):** Wrap TFLite Micro / CMSIS-NN inference (`model_runner_invoke()`) in `benchmark_target("dscnn_inference", ...)` to produce authoritative latency tables.
