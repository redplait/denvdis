#include "ptx_types.h"
#include <iostream>
#include <unistd.h>

int opt_v = 0;

void dump_res(ParseRes *pr, const std::unordered_map<int, PTXDot> *rem) {
  if ( rem ) {
    if ( rem->empty() ) return;
    printf("--- rem attrs %ld:\n", rem->size());
    for ( auto &name: *rem ) {
      auto rlen = name.second.second.size();
      printf(" col %d %.*s len %d\n", name.second.first, rlen, name.second.second.data(), rlen);
    }
  }
  if ( pr->forms.empty() ) return;
  int latch = 0;
  for ( auto &f: pr->forms ) {
    if ( !latch++ ) printf("--> %s\n", f->name);
    printf(" line %d:", f->ln);
    if ( f->fmt ) printf(" %s", f->fmt);
    if ( f->ops ) printf(" %s", f->ops);
    fputc('\n', stdout);
  }
  if ( !pr->types.empty() ) {
    printf("--- types %ld:\n", pr->types.size());
    for ( auto &name: pr->types )
      printf(" %.*s\n", name.size(), name.data());
  }
  if ( !pr->attrs.empty() ) {
    printf("--- attrs %ld:\n", pr->attrs.size());
    for ( auto ap: pr->attrs ) {
      if ( ap.first >= 0 ) {
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
  printf(" -d - debug mode\n");
  printf(" -t - apply number of operands\n");
  printf(" -v - verbose mode\n");
  exit(6);
}

int main(int argc, char **argv)
{
  int c, opt_t = 0, opt_r = 0;
  while(1) {
    c = getopt(argc, argv, "drtv");
    if ( c == -1 ) break;
    switch(c) {
      case 'r': opt_r = 1; break;
      case 't': opt_t = 1; break;
      case 'v': opt_v |= 1; break;
      case 'd': opt_v |= 2; break;
      default: usage(argv[0]);
    }
  }
  PTXParser p(nullptr);
  while( !std::cin.eof() ) {
    std::string str;
    std::getline(std::cin, str);
    if ( str.empty() ) continue;
    auto res = p.parse(str, opt_t, opt_v);
    p.dump(stdout);
    if ( res ) {
      dump_res(res, opt_r ? &p.rem_attrs() : nullptr);
      delete res;
    }
  }
}