#include "ptx_types.h"
#include <iostream>

int main() {
  PTXParser p(nullptr);
  while( !std::cin.eof() ) {
    std::string str;
    std::getline(std::cin, str);
    if ( str.empty() ) continue;
    auto res = p.parse(str, 1);
    p.dump(stdout);
  }
}