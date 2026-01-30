#pragma once
#include "decuda_base.h"

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
   // output data
   uint64_t m_api = 0;
};