#pragma once

// from kv to float
float e4m3_f(uint8_t);
float e5m2_f(uint8_t);
float e8m7_f(uint16_t);
double e4m3_d(uint8_t);
double e5m2_d(uint8_t);
double e8m7_d(uint16_t);
// from float to kv
uint8_t conv_e4m3(float);
uint8_t conv_e5m2(float);
uint16_t conv_e8m7(float);
