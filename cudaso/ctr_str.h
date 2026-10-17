#pragma once
#include "decuda_base.h"

class ctr_str: public decuda_base {
 public:
  ctr_str(ELFIO::elfio *rdr):
     decuda_base(rdr)
   {
   }
   int _read() override {
    if ( !s_rodata.has_value() ) return 0;
    for ( ELFIO::Elf_Half i = 0; i < n_sec; ++i ) {
      ELFIO::section *sec = m_rdr->sections[i];
      auto st = sec->get_type();
      // fill some sections
      if ( st == ELFIO::SHT_PROGBITS ) {
        if ( sec->get_name() == ".ctors" ) { s_ctors = sec; continue; }
      } else if ( st == ELFIO::SHT_INIT_ARRAY && sec->get_size() ) {
        s_init_array = sec;
        continue;
      }
    }
    if ( s_ctors.has_value() )
      return proc_section(s_ctors.value());
    else if ( s_init_array.has_value() )
      return proc_section(s_init_array.value() );
    else
      return 0;
   }
 protected:
   int proc_section(ELFIO::section *);
   const char *look_like_str(unsigned long);
   int process_item(diter &, unsigned long, bool second = false);
   std::unordered_set<unsigned long> m_visited;
   int m_idx;
   bool m_latch;
   std::optional<ELFIO::section *> s_ctors, s_init_array;
};
