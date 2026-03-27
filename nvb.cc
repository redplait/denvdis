#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
// tool to dump llvm builtins BC from libnvrtc-builtins.so.XXX
// rdi - size_t *, rsi - arch
typedef const unsigned char *(*Hack)(size_t *, int);
typedef const char *(*Hdr)(size_t *);

int dump(int arch, const unsigned char *bc, size_t sz) {
  std::string fn = std::to_string(arch);
  fn += ".bc";
  auto cstr = fn.c_str();
  FILE *fp = fopen(cstr, "wb");
  if ( !fp ) {
    fprintf(stderr, "cannot create %s, error %d (%s)\n", cstr, errno, strerror(errno));
    return errno;
  }
  fwrite(bc, 1, sz, fp);
  fclose(fp);
  return 0;
}

int dump_hdr(const char *bc, size_t sz) {
  const char *hdr_fn = "BuiltinHeader.h";
  FILE *fp = fopen(hdr_fn, "wb");
  if ( !fp ) {
    fprintf(stderr, "cannot create %s, error %d (%s)\n", hdr_fn, errno, strerror(errno));
    return errno;
  }
  fwrite(bc, 1, sz, fp);
  fclose(fp);
  return 0;

}

int main(int argc, char **argv) {
  if ( argc != 2 ) {
    fprintf(stderr, "where is patch to libnvrtc-builtins.so ?\n");
    return 1;
  }
  auto dl = dlopen(argv[1], RTLD_NOW);
  if ( !dl ) {
    fprintf(stderr, "cant open %s: %s\n", argv[1], dlerror());
    exit(EXIT_FAILURE);
  }
  size_t sz = 0;
  Hdr gh = (Hdr)dlsym(dl, "getBuiltinHeader");
  if ( !gh ) {
    fprintf(stderr, "cant find hdr: %s\n", dlerror());
  } else {
    auto res = gh(&sz);
    if ( res ) dump_hdr(res, sz);
    else fprintf(stderr, "cant get hdr\n");
  }
  Hack h = (Hack)dlsym(dl, "getArchBuiltins");
  if ( !h ) {
    dlclose(dl);
    fprintf(stderr, "cant hack: %s\n", dlerror());
    exit(EXIT_FAILURE);
  }
  // brute-force
  for ( int arch = 1; arch <= 0x80; arch++ ) {
    sz = 0;
    auto res = h(&sz, arch);
    if ( res ) dump(arch, res, sz);
  }
  dlclose(dl);
  return 0;
}
