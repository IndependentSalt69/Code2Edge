# STM32U585 Firmware Integration (`src/firmware/`)

This directory contains the firmware source code for the **STMicroelectronics STM32U585** MCU subsystem on the **Arduino UNO Q**.

## Target Architecture

- **MCU:** STM32U585 (ARM Cortex-M33 @ 160 MHz)
- **Board:** Arduino UNO Q
- **Toolchain:** `arduino-cli` with `arduino:zephyr:unoq` FQBN (Zephyr RTOS core)
- **Inference Runtime:** TensorFlow Lite for Microcontrollers (TFLM) with CMSIS-NN kernels

## Planned Structure

```text
src/firmware/
├── app.ino                # Arduino sketch entrypoint (setup, loop, UART polling)
├── model_runner.h / .cpp  # TFLite Micro interpreter, tensor arena, invocation logic
├── feature_runner.h / .c  # Preprocessing wrapper invoking generated CMSIS-DSP code
├── timer.h / .c           # ARM Cortex-M33 DWT cycle counter (DWT->CYCCNT)
├── uart_bridge.h / .c     # Framed binary protocol handler for MPU communication
└── golden_fixtures.h      # Embedded test vectors for offline fallback execution
```

## Static Memory Rules

- **Zero Dynamic Allocation:** `malloc()` and `free()` must never be called during inference.
- **Tensor Arena:** Declared globally in `.bss` with 16-byte alignment (`constexpr int kTensorArenaSize = 64 * 1024;`).
- **Feature Buffer:** Ping-pong audio buffer allocated statically.
