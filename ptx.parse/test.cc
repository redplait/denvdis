#include "ptx_types.h"
#include <iostream>

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
    printf("types %ld:\n", pr->types.size());
    for ( auto &name: pr->types )
      printf(" %.*s\n", name.size(), name.data());
  }
  if ( !pr->attrs.empty() ) {
    printf("attrs %ld:\n", pr->attrs.size());
  }
}

int main() {
  PTXParser p(nullptr);
  while( !std::cin.eof() ) {
    std::string str;
    std::getline(std::cin, str);
    if ( str.empty() ) continue;
    auto res = p.parse(str, 1);
    p.dump(stdout);
    if ( res ) {
      dump_res(res);
      delete res;
    }
  }
}