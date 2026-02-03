#include "decuda.h"
#include "de_bg.h"
#include "de_cupti.h"
#include <unistd.h>

int opt_v = 0,
    opt_d = 0,
    opt_t = 0;

void usage(const char *prog)
{
  printf("%s usage: [options] libcubin.so\n", prog);
  printf("Options:\n");
  printf("-d - show disasm\n");
  printf("-t - dump symbols\n");
  printf("-v - verbose mode\n");
  exit(6);
}

template <typename T>
void process(T *dc) {
  dc->read();
  if ( opt_t ) dc->dump_syms();
  dc->dump_res();
}

int main(int argc, char **argv) {
  int c;
  int do_dbg = 0;
  int do_cupti = 0;
  while(1) {
    c = getopt(argc, argv, "CDdtv");
    if ( c == -1 ) break;
    switch(c) {
      case 'd': opt_d = 1; break;
      case 'C': do_cupti = 1; break;
      case 'D': do_dbg = 1; break;
      case 't': opt_t = 1; break;
      case 'v': opt_v = 1; break;
      default: usage(argv[0]);
    }
  }
  if ( argc == optind ) {
    usage(argv[0]);
    return 6;
  }
  if ( do_dbg || do_cupti ) {
    ELFIO::elfio *rdr = new ELFIO::elfio;
    if ( !rdr->load(argv[optind]) ) {
      delete rdr;
      fprintf(stderr, "cannot load ELF %s\n", argv[optind]);
      return 2;
    }
    if ( do_cupti ) {
      de_cupti dg(rdr);
      process(&dg);
    } else {
      de_bg dg(rdr);
      process(&dg);
    }
  } else {
    decuda *dc = get_decuda(argv[optind]);
    if ( !dc ) return 2;
    process(dc);
    delete dc;
  }
}