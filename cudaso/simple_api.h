#pragma once

#if __cplusplus
extern "C" {
#endif
struct dbg_patch {
  const char *name;
  int what;
};

void check_cuda(const char *fname, FILE *fp);
void check_patch(const char *fname, FILE *fp, const struct dbg_patch *, int);
// couple of cautions
// 1) you can't close fp
// 2) masks is bit mask - 1 to add tracepoint, 2 to hexdump
int set_logger(const char *fname, FILE *fp, const unsigned char *mask, size_t mask_size);
int reset_logger();

#if __cplusplus
};
#endif