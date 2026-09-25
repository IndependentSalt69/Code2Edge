# Code2Edge Hardware Benchmark Evidence

This directory stores authoritative physical benchmark runs executed on the real **Arduino UNO Q (STM32U585)** hardware.

In accordance with Code2Edge architectural rules:
- All benchmark records strictly distinguish `MEASURED` vs `ESTIMATED` data.
- Compiler output sizing (approximate) is reported separately from linker-map section measurements.
- Benchmark harness infrastructure validation is explicitly segregated from downstream Keyword Spotting (KWS) model inference measurements.

---

## 1. Verified Benchmark Evidence Summary

| Run ID / Artifact | Workload Type | Run Type | CPU Cycles ($\mu \pm \sigma$) | Execution Time ($\mu$) | Static Flash | Static SRAM | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [`benchmark_harness_physical_run.json`](./benchmark_harness_physical_run.json) | Harness Validation (`deterministic_kernel`) | **MEASURED** | $990 \pm 0.00$ cycles | $6.1875\text{ }\mu\text{s}$ | $30,132\text{ B}$ ($29.43\text{ KB}$) | $8,264\text{ B}$ ($8.07\text{ KB}$) | **PASSED** (Checksum `0xBBF4ED4A`) |

> [!NOTE]
> **Measurement Infrastructure Validation:** The result above is the physical execution of the timing and measurement infrastructure (`deterministic_kernel`). It verifies zero-jitter DWT cycle counting and UART reporting on the ARM Cortex-M33 core. It is **NOT** the DS-CNN / KWS model latency.

---

## 2. Benchmark Harness Execution Details (STM32U585)

- **Target Device:** Arduino UNO Q — STM32U585 MCU Subsystem (ARM Cortex-M33)
- **Target Clock:** 160.0 MHz ($1\text{ cycle} = 6.25\text{ ns}$)
- **Measurement Engine:** Hardware Data Watchpoint and Trace (`DWT_CYCCNT`)
- **Sample Count:** 5 warmup iterations, 50 measured iterations
- **Observed Metrics:**
  - `min_cycles`: **990**
  - `max_cycles`: **990**
  - `avg_cycles`: **990.00**
  - `stddev_cycles`: **0.00** (Zero jitter across 50 iterations)
  - `avg_us`: **6.1875 µs**
  - `checksum`: `0xBBF4ED4A`

---

## 3. Authoritative Memory Breakdown

Memory metrics are parsed directly from the GCC linker map file using [`tools/target/parse_map.py`](../../tools/target/parse_map.py).

### Linker Map Section Sizing (Authoritative)
| Section | Type | Size (Bytes) | Size (KB) | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `.text` | Code Flash | 26,664 | 26.04 KB | Core executable instructions & Zephyr runtime |
| `.rodata` | Constant Flash | 968 | 0.95 KB | Read-only strings, format tables |
| `.llext.rodata.noreloc` | Zephyr LLEXT | 2,488 | 2.43 KB | Zephyr loadable extension metadata |
| `.data` | Initialized SRAM | 12 | 0.01 KB | Initialized global/static variables |
| `.bss` | Zero-Init SRAM | 8,252 | 8.06 KB | Static cycle recording buffer & runtime state |
| **Total App Flash** | **`.text + .rodata + .data + .llext`** | **30,132** | **29.43 KB** | **1.44% of 2 MB Flash (2,067,020 B Headroom)** |
| **Total App SRAM** | **`.data + .bss`** | **8,264** | **8.07 KB** | **1.03% of 786 KB SRAM (796,600 B Headroom)** |

### Compiler High-Level Sizing Summary
The `arduino-cli` compiler summary reports binary container sizes including Zephyr base images:
- **Program Storage Space:** 87,304 bytes (11% of 786,432 bytes virtual partition)
- **Dynamic Memory:** 35,928 bytes (13% of 262,144 bytes virtual partition)

---

## 4. Benchmark Execution Command Trace

```bash
# 1. Compile benchmark harness
arduino-cli compile --fqbn arduino:zephyr:unoq --build-path build/benchmark_harness tests/firmware/benchmark_harness

# 2. Extract exact linker section footprints
python tools/target/parse_map.py --map build/benchmark_harness/benchmark_harness.ino.map --json

# 3. Flash to STM32U585 and monitor
arduino-cli upload -p COM3 --fqbn arduino:zephyr:unoq tests/firmware/benchmark_harness
arduino-cli monitor -p COM3 --config baudrate=115200
```
