#include <Arduino.h>

const int PIN_ECG_OUT = A0;
const int PIN_LO_PLUS = 2;
const int PIN_LO_MINUS = 3;

const unsigned long SAMPLE_INTERVAL_MICROS = 2778;
unsigned long nextSampleMicros = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ;
  }

  pinMode(PIN_LO_PLUS, INPUT);
  pinMode(PIN_LO_MINUS, INPUT);
  pinMode(PIN_ECG_OUT, INPUT);

  nextSampleMicros = micros() + SAMPLE_INTERVAL_MICROS;
}

void loop() {
  unsigned long currentMicros = micros();

  if ((long)(currentMicros - nextSampleMicros) >= 0) {
    nextSampleMicros += SAMPLE_INTERVAL_MICROS;

    int lo_plus = digitalRead(PIN_LO_PLUS);
    int lo_minus = digitalRead(PIN_LO_MINUS);
    int leads_off = (lo_plus == 1 || lo_minus == 1) ? 1 : 0;

    int ecg_raw = analogRead(PIN_ECG_OUT);

    long ppg_ir = 0;
    long ppg_red = 0;

    Serial.print(ecg_raw);
    Serial.print(',');
    Serial.print(leads_off);
    Serial.print(',');
    Serial.print(ppg_ir);
    Serial.print(',');
    Serial.println(ppg_red);
  }
}
