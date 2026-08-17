#include "ptx_types.h"
#include <iostream>
#include <unistd.h>

int opt_v = 0;

void dump_res(ParseRes *pr) {
  if ( !pr->forms.empty() ) {
    for ( auto &f: pr->forms ) {
      printf(" line %d:", f->ln);
      if ( f->fmt ) printf(" %s", f->fmt);
      if ( f->ops ) printf(" %s", f->ops);
      fputc('\n', stdout);
    }
  }
  if ( !pr->types.empty() ) {
    printf("--- types %ld:\n", pr->types.size());
    for ( auto &name: pr->types )
      printf(" %.*s\n", name.size(), name.data());
  }
  if ( !pr->attrs.empty() ) {
    printf("--- attrs %ld:\n", pr->attrs.size());
    for ( auto ap: pr->attrs ) {
      if ( ap.first != -1 ) {
        int maj = ap.first >> 3;
        int min = ap.first & 7;
        printf(" %d:%d", maj, min);
      }
      printf(" %.*s\n", ap.second.size(), ap.second.data());
    }
  }
}

void usage(const char *prog)
{
  printf("usage: %s [options]\n", prog);
  printf("Options:\n");
  printf(" -v - verbose mode\n");
  exit(6);
}

int main(int argc, char **argv)
{
  int c;
  while(1) {
    c = getopt(argc, argv, "v");
    if ( c == -1 ) break;
    switch(c) {
      case 'v': opt_v = 1; break;
      default: usage(argv[0]);
    }
  }
  PTXParser p(nullptr);
  while( !std::cin.eof() ) {
    std::string str;
    std::getline(std::cin, str);
    if ( str.empty() ) continue;
    auto res = p.parse(str, opt_v);
    p.dump(stdout);
    if ( res ) {
      dump_res(res);
      delete res;
    }
  }
}