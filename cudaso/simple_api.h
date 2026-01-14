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

#if __cplusplus
};
#endif