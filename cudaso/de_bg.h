#pragma once
#include "decuda_base.h"

struct bg_api {
  uint64_t addr = 0;
  uint64_t sub = 0;
  std::string name;
};

struct diter;

class de_bg: public decuda_base {
 public:
  de_bg(ELFIO::elfio *rdr):
     decuda_base(rdr)
   {
   }
   void dump_res() const;
 protected:
   virtual int _read() override;
   int looks_name(uint64_t, std::string &) const;
   int try_api(uint64_t, std::vector<elf_reloc>::iterator &);
   int try_hack_api(std::vector<elf_reloc>::iterator &);
   int try_one_api(diter &, uint64_t, std::string &);
   int extract_name(diter &, uint64_t, std::string &);
   // output data
   uint64_t m_api = 0;
   uint64_t m_state = 0;
   uint64_t m_bg_log = 0;
   std::vector<bg_api> m_apis;
};