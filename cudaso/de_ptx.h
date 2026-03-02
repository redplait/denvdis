#pragma once
#include "decuda_base.h"

struct de_res {
  int what; // 0 - num, 1 - '-', 2 - decrypted string
  int num;
  std::string dec;
};

using res_map = std::map<uint64_t, de_res>;

// dirty hack to extract encrypted tables from ptxas
class de_ptx: public decuda_base {
 public:
  de_ptx(ELFIO::elfio *rdr):
     decuda_base(rdr)
   {
   }
   void dump_res() const {}
 protected:
   virtual int _read();
   void hack_ctor(uint64_t, const char *fname);
   int hack(diter &, res_map &);
   int check(de_res &, uint64_t off);
   int dump_deres(const char *fname, const res_map &);
};