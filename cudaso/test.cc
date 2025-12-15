#include "decuda.h"
#include <unistd.h>

int opt_v = 0,
    opt_d = 0,
    opt_t = 0;

void usage(const char *prog)
{
  printf("%s usage: [options] libcubin.so\n", prog);
  printf("Options:\n");
  printf("-t - dump symbols\n");
  printf("-v - verbose mode\n");
  exit(6);
 
}

int main(int argc, char **argv) {
  int c;
  while(1) {
    c = getopt(argc, argv, "dtv");
    if ( c == -1 ) break;
    switch(c) {
      case 'd': opt_d = 1; break;
      case 't': opt_t = 1; break;
      case 'v': opt_v = 1; break;
      default: usage(argv[0]);
    }
  }
  if ( argc == optind ) {
    usage(argv[0]);
    return 6;
  }
  decuda *dc = get_decuda(argv[optind]);
  if ( !dc ) return 2;
  dc->read();
  if ( opt_t ) dc->dump_syms();
  dc->dump_res();
  delete dc;
}