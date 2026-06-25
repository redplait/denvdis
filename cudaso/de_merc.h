#pragma once
#include "decuda_base.h"

// dirty hack to extract encrypted tables from ptxas
class de_merc: public decuda_base {
 public:
  de_merc(ELFIO::elfio *rdr):
     decuda_base(rdr)
   {
   }
   void dump_res() const {}
   template <typename S>
   struct lat_res {
    int what; // 0 - num, 1 - '-', 2 - decrypted string
    int num;
    S str;
   };
   typedef std::map<uint64_t, lat_res<std::string> > dec_map;
   typedef std::map<uint64_t, lat_res<std::string_view> > opt_map;
  protected:
   virtual int _read();
   template <typename T>
   int hack(uint64_t addr);
   template <typename T>
   int _hack(diter &, std::map<uint64_t, T> &);
   template <typename T>
   void dump(const T&);
   template <typename T>
   int check(T &, uint64_t);
};