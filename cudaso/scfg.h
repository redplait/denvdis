#pragma once

#define WITH_CEREAL

#ifdef WITH_CEREAL
#include "cereal/cereal.hpp"

constexpr const char *json_deb = "cudadeb.json";
constexpr const char *json_cuda = "cuda.json";

#endif

// for nvPTX hooking
#define HOOK_nvPTX