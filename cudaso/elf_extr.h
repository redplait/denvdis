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
     auto *s = s_rodata.value();
     const unsigned char *start = (const unsigned char *)s->get_data();
     const unsigned char *ptr = (const unsigned char *)start + from - s->get_address();
     if ( !ptr ) return 0;
     const unsigned char *end = (const unsigned char *)start + s->get_size();
     const unsigned char *prev = nullptr;
     int cnt = 0;
     for ( ptr = search(ptr, end); ptr; ptr = search(ptr + sign_len, end) )
     {
       if ( prev ) {
         dump(prev, s, ptr - prev); ++cnt;
       }
       prev = ptr;
     }
     if ( prev ) {
       dump(prev, s, end - prev); ++cnt;
     }
     return cnt;
   }
 protected:
  const unsigned char sign[4] = { 0x7F, 'E', 'L', 'F' };
   const int sign_len = 4;
   const unsigned char *search(const unsigned char *from, const unsigned char *upto) {
     while ( from < upto - sign_len ) {
       if ( from[0] != 0x7f ) { from++; continue; }
       if ( from[1] == 'E' && from[2] == 'L' && from[3] == 'F' ) return from;
       ++from;
     }
     return nullptr;
   }
   int dump(const unsigned char *ptr, const ELFIO::section *s, size_t len) {
     if ( !len ) return 0;
     auto s_start = s->get_address();
     auto diff = ptr - (const unsigned char *)s->get_data();
     char buf[50];
     sprintf(buf, ".%X-%d", s_start + diff, len);
     std::string fname = std::to_string(++m_idx);
     fname += buf;
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