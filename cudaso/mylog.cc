#include <stdint.h>
#include <stdio.h>
#include <mutex>
#include "trace_fmt.h"

static dbg_trace old_handler = nullptr;
static uint64_t logger_addr;

#define STORED_MASKS   0x60

static unsigned char hex_masks[STORED_MASKS];
static std::mutex mtx;
static FILE *s_fp = nullptr;

// void 

// patcher
int patch_dbg(uint64_t addr, FILE *fp, const unsigned char *mask, size_t mask_size) {
  for ( int i = 0; i < std::min(mask_size, sizeof(hex_masks)); i++ ) {
    hex_masks[i] = ( mask[i] & 2 ) ? 1 : 0;
  }
  logger_addr = addr;
  old_handler = *(dbg_trace *)addr;
  s_fp = fp;
  return 1;
};