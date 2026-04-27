#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>

/* simple decryptor of nvidia-smi logs
   to produce logs set env vars
__NVML_DBG_FILE=log
__NVML_DBG_LVL=DEBUG
__RM_ENABLE_VERBOSE_OUTPUT=1
HWLOC_COMPONENTS_VERBOSE=1
HWLOC_GROUPING_VERBOSE-1
   in libnvidia-ml.so.1 there is string to build directory /home/huyn/work/p4/sw/tools/unix/hosts/Linux-x86/targets
   huyn, yeah, my ass
 */

static uint32_t
 key1 = 0xD3DAECB8,
 key2 = 0x1D4D4848,
 key3 = 0xAA7B8E81,
 key4 = 0x23CC0EC3,
 key5 = 0x7645F3ED,
 key6 = 0xE44A4F49;

static uint32_t next(void) {
  uint32_t v1; // eax
  uint32_t v0 = key1 ^ ((uint32_t)key1 >> 2);
  key1 = key2;
  key2 = key3;
  v1 = key4;
  key4 = key5;
  key3 = v1;
  key5 ^= (2 * v0) ^ v0 ^ (16 * key5);
  key6 += 0x587C5;
  return (uint32_t)(key6 + key5);
}

int main(int argc, char **argv) {
  if ( argc != 2 ) {
    fprintf(stderr, "where is log?\n");
    return 1;
  }
  FILE *fp = fopen(argv[1], "rb");
  if ( fp == NULL ) {
   fprintf(stderr, "cannot open %s, error %d (%s)\n", argv[1], errno, strerror(errno));
   return 2;
  }
  while( !feof(fp) ) {
    auto c = fgetc(fp);
    c -= next();
    putc(c, stdout);
  }
  fclose(fp);
}