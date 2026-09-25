# UNO Q RouterBridge RPC Smoke Test

Diagnostic firmware and verification procedure for the **Arduino UNO Q** inter-processor communication bridge (Linux MPU ↔ STM32U585 MCU).

---

## 1. Purpose & Overview

This test verifies the complete hardware and transport path between the Qualcomm Dragonwing QRB2210 Linux subsystem and the STM32U585 MCU subsystem:
- **MPU Subsystem:** Qualcomm Dragonwing QRB2210 (Linux)
- **MCU Subsystem:** STMicroelectronics STM32U585 (ARM Cortex-M33)
- **Hardware Transport:** Internal UART `/dev/ttyHS1` managed by `arduino-router` daemon
- **Linux IPC Endpoint:** UNIX domain socket at `/var/run/arduino-router.sock`
- **Firmware Library:** `Arduino_RouterBridge` (RPC provider on STM32U585)

The firmware registers a remote procedure call `code2edge_ping` which returns the integer `42`.

---

## 2. Target Details & Commands

- **Board:** Arduino UNO Q
- **Target MCU:** STM32U585
- **FQBN:** `arduino:zephyr:unoq`

### Build Command
```bash
arduino-cli compile --fqbn arduino:zephyr:unoq .
```

### Upload Command
```bash
arduino-cli upload -p COM3 --fqbn arduino:zephyr:unoq .
```

### Serial Monitor Command
```bash
arduino-cli monitor -p COM3 --config baudrate=115200
```

### Expected MCU Serial Output
```text
CODE2EDGE_BRIDGE_TEST_START
BRIDGE_BEGIN_OK
BRIDGE_PROVIDE_OK
```

---

## 3. Linux RPC Verification Procedure

From the host development PC, access the Linux MPU shell via ADB:
```bash
adb shell
```

Execute the raw RPC client snippet against `/var/run/arduino-router.sock`:
```bash
python3 -c 'import socket; s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.settimeout(3); s.connect("/var/run/arduino-router.sock"); s.sendall(bytes.fromhex("940001ae636f646532656467655f70696e6790")); print("Response:", s.recv(1024).hex())'
```

### Verified Response
```text
Response: 940101c02a
```

### Packet Breakdown
- `940101c02a` is the msgpack/framed RPC response payload from `arduino-router`.
- The terminal byte `2a` (hexadecimal `0x2a`) equals decimal `42`, corresponding to the return value of `code2edge_ping()`.

---

## 4. Verified Hardware Evidence

The following events were directly observed and verified on physical hardware:
- `arduino-cli` detected `Arduino UNO Q` as `arduino:zephyr:unoq` on port `COM3`.
- STM32U585 microcontroller was successfully flashed with `router_bridge_smoke.ino`.
- Serial output confirmed `BRIDGE_BEGIN_OK`.
- Serial output confirmed `BRIDGE_PROVIDE_OK`.
- Linux-side RPC invocation across `/var/run/arduino-router.sock` returned `940101c02a` (result value `42`).

---

## 5. Scope & Isolation

This directory is an isolated diagnostic smoke-test artifact. It does not contain TensorFlow Lite Micro, Keyword Spotting models, audio feature extraction, or performance benchmark harnesses.
