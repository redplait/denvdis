#pragma once

#include "elfio/elfio.hpp"
#include "types.h"
#include <optional>
#include <memory>
#include <map>
#include <unordered_set>

class rtmem_storage;

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

struct one_intf {
  unsigned char uuid[16];
  uint64_t addr;
  int size = 0;
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
   void dump_syms() const;
   void dump_res() const;
   void verify(FILE *) const;
 protected:
   bool in_sec(std::optional<ELFIO::section *> &s, uint64_t addr) const {
     if ( !s.has_value() ) return false;
     return addr >= (*s)->get_address() && addr < ((*s)->get_address() + (*s)->get_size());
   }
   template <typename T>
   bool read(ELFIO::section *, uint64_t off, T &);
   uint32_t read_size(uint64_t);
   uint32_t read_size(ELFIO::section *, uint64_t off);
   int read_syms(ELFIO::section *);
   int read_rels(std::unordered_set<ELFIO::Elf_Half> &, int);
   int find_intf_tab();
   int resolve_indirects();
   int resolve_api_gate(ptrdiff_t);
   // verifier methods
   template <typename T>
   bool dump_xxx(FILE *fp, const char *, int64_t delta) const;
   bool dump_str_with_len(FILE *fp, const char *addr, const char *len, int64_t delta) const;
   void dump_bss_publics(FILE *fp, int64_t delta) const;
   void check_addr(FILE *fp, uint64_t, int64_t delta, const char *pfx, rtmem_storage &) const;
   bool is_32;
   ELFIO::elfio *m_rdr;
   std::vector<elf_reloc> m_relocs;
   std::map<std::string, elf_symbol> m_syms;
   // sections
   ELFIO::Elf_Half n_sec = 0;
   std::optional<ELFIO::section *> s_text, s_rodata, s_bss, s_data, s_data_rel;
   // output data
   uint64_t m_api_gate = 0;
   uint64_t m_api_data = 0;
   uint64_t m_intf_tab = 0;
   std::vector<one_intf> m_intfs;
   // key is name from m_syms so we can use string_view
   // value is pair<ptr, original function>
   std::map<std::string_view, std::pair<uint64_t, uint64_t> > m_forwards;
};

decuda *get_decuda(const char *);
