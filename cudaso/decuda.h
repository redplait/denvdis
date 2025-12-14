#pragma once

#include "elfio/elfio.hpp"
#include "types.h"
#include <optional>
#include <memory>
#include <map>
#include <unordered_set>

// symbols
struct elf_symbol {
  std::string name;
  ELFIO::Elf64_Addr addr;
  ELFIO::Elf_Xword idx = 0;
  ELFIO::Elf_Xword size = 0;
  ELFIO::Elf_Half section;
  unsigned char bind = 0,
                type = 0,
                other = 0;
};

// relocs - stored in sorted vector
struct elf_reloc
{
  uint64_t offset;
  ELFIO::Elf_Xword info;
  int64_t add;
  int is_rela;
};

class decuda {
 public:
   decuda(ELFIO::elfio *rdr):
     m_rdr(rdr)
   {
   }
   ~decuda() {
     if ( m_rdr ) delete m_rdr;
   }
   int read();
 protected:
   int read_syms(ELFIO::section *);
   int read_rels(std::unordered_set<ELFIO::Elf_Half> &, int);
   ELFIO::elfio *m_rdr;
   std::vector<elf_reloc> m_relocs;
   std::map<std::string, elf_symbol> m_syms;
   // sections
   ELFIO::Elf_Half n_sec = 0;
   std::optional<ELFIO::section *> s_text, s_rodata, s_bss, s_data, s_data_rel;
   // output data
};

decuda *get_decuda(const char *);
