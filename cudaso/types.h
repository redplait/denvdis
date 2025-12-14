#pragma once

#ifdef _MSC_VER
#include <windows.h>
typedef unsigned __int64 a64;
typedef __int64 sa64;
#else
#include <cstdint>

typedef unsigned char *PBYTE;
typedef uint64_t a64;
typedef int64_t sa64;
typedef uint32_t DWORD;
typedef uint32_t *PDWORD;
typedef long LONG;
typedef unsigned long ULONG;
#endif
