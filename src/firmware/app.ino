/*
 * Code2Edge STM32U585 Firmware Entrypoint (Arduino UNO Q)
 *
 * Target: Arduino UNO Q (STM32U585 MCU subsystem)
 * Core:   arduino:zephyr (1.0.0)
 *
 * Day-1 Objective:
 * Minimal boot, serial heartbeat, and deterministic dummy inference proof.
 */

#include <Arduino.h>

#define SERIAL_BAUD_RATE 115200
#define PROTOCOL_START_BYTE 0xAA
#define PROTOCOL_END_BYTE   0x55

// Global static tensor arena (aligned to 16 bytes, zero dynamic allocation)
alignas(16) static uint8_t g_tensor_arena[32 * 1024];

// Dummy inference counter
static volatile uint32_t g_inference_count = 0;

void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    while (!Serial && millis() < 2000) {
        // Wait for serial or timeout
    }

    Serial.println(F("{\"event\":\"boot\",\"target\":\"arduino_uno_q_stm32u585\",\"status\":\"ready\"}"));
}

void loop() {
    // Check for incoming bridge command from Dragonwing MPU
    if (Serial.available() > 0) {
        uint8_t b = Serial.read();
        if (b == 'P' || b == PROTOCOL_START_BYTE) {
            // Execute dummy benchmark inference iteration
            unsigned long t_start = micros();
            
            // Dummy arithmetic simulating MAC operations in tensor arena
            volatile uint32_t dummy_sum = 0;
            for (size_t i = 0; i < 1000; ++i) {
                g_tensor_arena[i % sizeof(g_tensor_arena)] = (uint8_t)(i ^ 0x5A);
                dummy_sum += g_tensor_arena[i % sizeof(g_tensor_arena)];
            }
            
            unsigned long t_elapsed_us = micros() - t_start;
            g_inference_count++;

            // Return deterministic JSON metric line
            Serial.print(F("{\"event\":\"inference_done\",\"iter\":"));
            Serial.print(g_inference_count);
            Serial.print(F(",\"latency_us\":"));
            Serial.print(t_elapsed_us);
            Serial.print(F(",\"sum\":"));
            Serial.print(dummy_sum);
            Serial.println(F("}"));
        }
    }
    delay(10);
}
