#include <Arduino_RouterBridge.h>

int code2edge_ping() {
    return 42;
}

void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("CODE2EDGE_BRIDGE_TEST_START");

    bool bridge_ok = Bridge.begin();

    if (bridge_ok) {
        Serial.println("BRIDGE_BEGIN_OK");
    } else {
        Serial.println("BRIDGE_BEGIN_FAIL");
    }

    bool provide_ok = Bridge.provide("code2edge_ping", code2edge_ping);

    if (provide_ok) {
        Serial.println("BRIDGE_PROVIDE_OK");
    } else {
        Serial.println("BRIDGE_PROVIDE_FAIL");
    }
}

void loop() {
    delay(100);
}
