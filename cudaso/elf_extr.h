#pragma once
#include "decuda_base.h"
#include "bm_search.h"

extern int opt_d;

class elf_extr: public decuda_base {
 public:
  elf_extr(ELFIO::elfio *rdr):
     decuda_base(rdr)
   {
     m_idx = -1;
   }
   virtual int _read() override {
     if ( !s_rodata.has_value() ) return 0;
     unsigned long from = 0x134C440;
     const unsigned char *ptr = (const unsigned char *)sdata(s_rodata, from);
     if ( !ptr ) return 0;
     auto *s = s_rodata.value();
     size_t len = s->get_size();
     const unsigned char * end = len + (const unsigned char *)s->get_data();
     const unsigned char sign[4] = { 0x7F, 'E', 'L', 'F' };
     bm_search bm(sign, sizeof(sign));
     const unsigned char *prev = nullptr;
     int cnt = 0;
     for ( ptr = bm.search(ptr, end - ptr); ptr; ptr = bm.search(ptr + 4, end - ptr - 4) )
     {
       if ( prev ) {
         dump(ptr, ptr - prev); ++cnt;
       }
       prev = ptr;
     }
     if ( prev ) {
       dump(prev, end - prev); ++cnt;
     }
     return cnt;
   }
 protected:
   int dump(const unsigned char *ptr, size_t len) {
     std::string fname = std::to_string(++m_idx);
     fname += ".elf";
     FILE *fp = fopen(fname.c_str(), "wn");
     if ( !fp ) {
       fprintf(stderr, "cannot create %s, error %d (%s)]n", fname.c_str(), errno, strerror(errno));
       return 0;
     }
     int res = 1;
     if ( 1 != fwrite(ptr, len, 1, fp) ) {
       fprintf(stderr, "cannot write %s, error %d (%s)]n", fname.c_str(), errno, strerror(errno));
       res = 0;
     }
     fclose(fp);
     return res;
   }
   int m_idx;
};