# Hardware & Embedded Specifications: STM32U585 on Arduino UNO Q

**Project:** Code2Edge — Deployment with proof.  
**Track:** Hardware & Embedded Systems (Person B)  
**Target Hardware:** STM32U585 MCU subsystem on Arduino UNO Q  
**Document Status:** Grounded Technical Audit & Specification (Pre-Implementation)

---

## 1. Target Hardware Architecture

### 1.1 Target Identity & Clarification
> **CRITICAL TARGET RULE:** Code2Edge targets the **STM32U585 MCU subsystem** of the **Arduino UNO Q**.  
> The **Arduino UNO R4** (which features a Renesas RA4M1 Cortex-M4 with 256 KB Flash / 32 KB SRAM) is **NOT** the target hardware. All previous references mentioning "UNO R4 / UNO Q" are erroneous and have been corrected to "Arduino UNO Q (STM32U585)".

The Arduino UNO Q is a heterogeneous dual-processor edge computing board:
1. **Host MPU (Edge Compute / Orchestrator):** Qualcomm Dragonwing QRB2210 (Quad-core ARM Cortex-A53, 4GB LPDDR4, running Linux). Connected via USB/ADB.
2. **Real-Time MCU (Ultra-Low-Power Target):** STMicroelectronics STM32U585 (ARM Cortex-M33 with TrustZone, FPU, and DSP extensions).

```text
+-------------------------------------------------------------------------------+
|                             Arduino UNO Q                                     |
|                                                                               |
|  +--------------------------------+       Internal UART       +-------------+ |
|  |     Qualcomm Dragonwing        | <-----------------------> |  STM32U585  | |
|  |           QRB2210              |    (/dev/ttyMSM0 / UART)  |  Cortex-M33 | |
|  |   (Quad Cortex-A53, Linux)     |                           |  @ 160 MHz  | |
|  |                                |                           |  2MB Flash  | |
|  |  - Python Parity Runner        |                           |  786KB SRAM | |
|  |  - Bridge Daemon (protocol.py) |                           |             | |
|  +--------------------------------+                           +-------------+ |
+-------------------------------------------------------------------------------+
```

### 1.2 STM32U585 Key Specifications
| Parameter | Value | Verification Status |
|---|---|---|
| **CPU Architecture** | ARM Cortex-M33 (ARMv8-M Mainline) | VERIFIED (STM32U585 datasheet) |
| **Max Clock Frequency** | 160 MHz | VERIFIED |
| **Hardware FPU** | Single-precision (FP32) | VERIFIED |
| **SIMD / DSP Extensions** | Armv8-M DSP instructions | VERIFIED |
| **Embedded Flash** | 2,048 KB (2 MB) dual-bank | VERIFIED |
| **Embedded SRAM** | 786 KB (including 64 KB CCM / SRAM1/2/3/4) | VERIFIED |
| **Security & Isolation** | ARM TrustZone, crypto accelerators | VERIFIED |
| **DMA** | Multi-channel GPDMA / LPDMA | VERIFIED |

---

## 2. Toolchain Audit & Decision

### 2.1 Environmental Audit Findings
An audit of the development environment and package indexes revealed:
1. **Installed Tool:** `arduino-cli` Version 1.4.1 exists at `C:\Users\MANAV\AppData\Local\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe`.
2. **Platform Index:** The official Arduino package repository defines:
   - Platform: `arduino:zephyr` (Package: `Arduino Q Boards`, version `1.0.0`)
   - Supported Boards: `Arduino UNO Q`, `Arduino Ventuno Q`
   - Tool Dependencies: `arduino:adb@32.0.0`, `zephyr:arm-zephyr-eabi@1.0.1`, `arduino:bin2uf2`, `arduino:bossac`, `arduino:dfu-util`.
3. **Firmware Core:** The Arduino UNO Q uses **Zephyr RTOS** underneath the Arduino core abstraction.
4. **Current Status in Environment:**
   - `arduino-cli`: VERIFIED INSTALLED.
   - `arduino:zephyr` core: VERIFIED AVAILABLE in index, NOT YET INSTALLED.
   - `arm-zephyr-eabi-gcc`: UNINSTALLED (bundled when core is installed).
   - Physical Board Detection: NOT VERIFIED (No board currently attached to host USB).

### 2.2 Toolchain Decision & Build Strategy
- **Primary Toolchain (Recommended):** `arduino-cli` with `arduino:zephyr:unoq` FQBN.
  - *Rationale:* Integrates the official board support package, handles MPU-to-MCU flashing through the vendor-approved pipeline, and links against the Cortex-M33 toolchain.
- **Secondary / Native Toolchain (Fallback):** Direct Zephyr RTOS build via `west` + CMake + Ninja targeting board `arduino_uno_q_stm32u585`.
- **Bare-Metal Toolchain (Tertiary Fallback):** STM32CubeIDE / GNU Arm Embedded Toolchain (`arm-none-eabi-gcc`) with standard CMSIS startup files.

---

## 3. Inference Runtime & Operator Viability

### 3.1 Candidate Runtimes
1. **TensorFlow Lite for Microcontrollers (TFLite Micro / TFLM):**
   - *Status:* VIABLE & RECOMMENDED.
   - *Operator Coverage for DS-CNN:* `CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`, `FULLY_CONNECTED`, `RELU`, `SOFTMAX`. All 6 operators are fully supported in TFLM reference and CMSIS-NN optimized kernels.
   - *Quantization:* Supports asymmetric int8 per-channel quantization.
2. **Bare-Metal C with CMSIS-NN:**
   - *Status:* VIABLE (Alternative / Co-design).
   - Direct calls to `arm_depthwise_conv_s8`, `arm_convolve_s8`, `arm_avgpool_s8`, `arm_fully_connected_s8`.
   - Bypasses TFLM interpreter overhead (~15 KB Flash savings), but requires custom execution graph sequencing.

### 3.2 CMSIS-NN Acceleration Readiness
- **Core Compatibility:** Cortex-M33 supports ARM DSP SIMD instructions.
- **Library Version:** CMSIS-NN 4.0.0+ contains optimized M33 kernels.
- **Verification Rule:** CMSIS-NN acceleration must not be claimed until cycle benchmarks confirm SIMD kernel dispatch rather than fallback reference code.

---

## 4. Memory Model & Static Budgeting

### 4.1 Memory Constraints
- **Flash Limit:** 2,048 KB (2.0 MB)
- **SRAM Limit:** 786 KB

### 4.2 Analytical Memory Budget (Estimated vs Measured)
> [!NOTE]
> All figures below are **ANALYTICAL ESTIMATES** based on DS-CNN architecture (119k parameters). Actual numbers will be populated once the model binary is linked.

| Memory Component | Float32 Baseline (Est.) | Int8 Quantized (Est.) | Actual Measured | Allocation Policy |
|---|---|---|---|---|
| **Model Weights (Flash)** | ~478 KB | ~120 KB | *Pending build* | Read-only Flash (`.rodata`) |
| **Runtime Code & Kernels (Flash)** | ~60–100 KB | ~80–120 KB | *Pending build* | Code Flash (`.text`) |
| **Total Flash Required** | ~578 KB | ~240 KB | *Pending build* | Max 2,048 KB (**~8.5x headroom**) |
| **Tensor Arena (SRAM)** | ~180 KB | ~45 KB | *Pending build* | Static continuous buffer |
| **Feature Buffer (SRAM)** | 25.8 KB (64x101x4B) | 6.5 KB (int8) | *Pending build* | Static buffer |
| **Stack & Static BSS (SRAM)** | ~16 KB | ~16 KB | *Pending build* | SRAM |
| **Total SRAM Required** | ~222 KB | ~68 KB | *Pending build* | Max 786 KB (**~11.5x headroom**) |

### 4.3 Memory Rules
1. **Zero Dynamic Allocation:** `malloc()`, `calloc()`, and `free()` are strictly forbidden in the inference and feature extraction paths.
2. **Static Arenas:** Tensor arena and feature buffers are declared globally in the `.bss` section with 16-byte alignment.
3. **No Heap Overhead:** TFLM `AllOpsResolver` should be pruned to `MicroMutableOpResolver` containing only the 6 required operators to minimize static memory.

---

## 5. MPU ↔ MCU Bridge Architecture

### 5.1 Physical Bus & Protocol
- **Transport:** Physical UART between Dragonwing QRB2210 MPU and STM32U585 MCU.
- **Device Node:** Linux MPU exposes serial port (typically `/dev/ttyMSM0` or `/dev/ttyHS0`).
- **Baud Rate:** 115,200 baud (smoke) / 921,600 baud (high-throughput parity streaming).
- **Packet Structure:** Framed binary protocol with start/stop sentinels and XOR checksum:
  ```text
  [0xAA] [CMD_TYPE: 1B] [PAYLOAD_LEN: 2B (BE)] [PAYLOAD: N Bytes] [CHECKSUM: 1B] [0x55]
  ```

### 5.2 Bridge Verification Test Plan
Before relying on the bridge for parity verification:
1. **Loopback Echo Test:** Send known array `[0x01, 0x02, 0x03, 0x04]` from Python on Linux MPU; verify MCU echoes back identical bytes.
2. **Full Audio Payload Test:** Stream 1-second 16 kHz audio buffer (16,000 samples = 32,000 bytes at int16) to MCU; verify SHA-256 match on MCU.
3. **Result Readback:** MCU executes inference and returns 12 logits (`12 * 4 = 48 bytes` float32) plus predicted class index.

### 5.3 Fallback Strategy: Offline Fixtures
If the UART bridge is unconfigured, electrically unstable, or introduces latency bottlenecks:
- **Strategy:** Embed golden audio fixtures and preprocessed features directly into MCU Flash header (`golden_fixtures.h`).
- **Execution:** MCU boots, executes inference across embedded test vectors in a loop, and reports metrics via standard serial console.

---

## 6. Measurement Methodology

### 6.1 Inference Latency
- **Timer:** High-resolution hardware cycle counter via ARM Cortex-M33 Data Watchpoint and Trace (`DWT->CYCCNT`) or SysTick.
- **Calculation:** $\text{Latency (ms)} = \frac{\text{Cycles}}{160,000,000} \times 1000$
- **Iterations:** Minimum 50 warm iterations; report min, max, average, and standard deviation.

### 6.2 Memory Footprint
- **Flash Usage:** Extracted from the GCC linker map file (`.text` + `.rodata` + `.data`).
- **SRAM Usage:** Extracted from linker map (`.data` + `.bss`) plus reported tensor arena utilization via `arena_used_bytes()`.

---

## 7. Technical Risks & Status Summary

| Risk Item | Impact | Status | Mitigation Plan |
|---|---|---|---|
| **Board Availability** | Cannot run on physical silicon | NOT VERIFIED | Build & test on QEMU / offline fixture harness first |
| **UNO Q Zephyr Core** | Core installation or compiler mismatch | PLANNED | Install `arduino:zephyr` core via `arduino-cli` |
| **MPU/MCU Bridge Stability** | Cannot stream live audio to MCU | UNKNOWN | Implement Day-1 UART loopback; use offline header fixtures as fallback |
| **CMSIS-NN Compatibility** | Compiler flags fail on Cortex-M33 | PLANNED | Validate standard TFLM reference kernels first, then enable CMSIS-NN |
| **Flash Overrun** | MCU memory exhaustion | MINIMAL (2MB Flash vs ~240KB req.) | Monitored via map file CI checks |
