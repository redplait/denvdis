#pragma once
#include "decuda_base.h"

struct bg_api {
  uint64_t addr = 0;
  uint64_t sub = 0;
  std::string name;
};

class de_bg: public decuda_base {
 public:
  de_bg(ELFIO::elfio *rdr):
     decuda_base(rdr)
   {
   }
   void dump_res() const;
   inline uint64_t dbg_root() const {
     if ( m_log_root ) return m_log_root;
     if ( m_bg_log ) return (m_bg_log - 0x68);
     return 0;
   }
   inline uint64_t bg_log() const { return m_bg_log; }
   int verify(FILE *, rtmem_storage &, int hook);
 protected:
   virtual int _read() override;
   int looks_name(uint64_t, std::string &) const;
   int try_api(uint64_t, std::vector<elf_reloc>::iterator &);
   int try_hack_api(std::vector<elf_reloc>::iterator &);
   int try_one_api(diter &, uint64_t, std::string &);
   int extract_name(diter &, uint64_t, std::string &);
   // verify methods
   int vrf_api(FILE *, uint64_t delta, rtmem_storage &);
   int vrf_log(FILE *, uint64_t delta, rtmem_storage &);
   void dump_logh(FILE *, const my_phdr *, int idx, uint64_t addr, rtmem_storage &);
   // output data
   uint64_t m_api = 0;
   uint64_t m_state = 0;
   uint64_t m_bg_log = 0;
   uint64_t m_log_root = 0;
   std::vector<bg_api> m_apis;
};