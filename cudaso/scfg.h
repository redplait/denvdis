#pragma once

#define WITH_CEREAL

#ifdef WITH_CEREAL
#include "cereal/cereal.hpp"

constexpr const char *json_deb = "cudadeb.json";
constexpr const char *json_cuda = "cuda.json";
constexpr const char *cuda_so1_path = "/usr/lib/x86_64-linux-gnu/libcuda.so.1";

#endif

// for nvPTX hooking
#define HOOK_nvPTX

// dump tables names
#define HAS_DUIDS