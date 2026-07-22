#pragma once
// types for cb param names
typedef std::unordered_map<unsigned short, const char *> NvCBParamNames;
struct NvCBParams {
  NvCBParamNames *bank0 = nullptr;
  int cnp_off = 0; // pascal & maxwell 0x1840, 0x1860 on more new SMs
  NvCBParamNames *cnp = nullptr;
};