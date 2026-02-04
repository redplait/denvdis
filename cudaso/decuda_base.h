#pragma once

#include "elfio/elfio.hpp"
#include "types.h"
#include <optional>
#include <memory>
#include <map>
#include <dlfcn.h>
#include "rtmem.h"
#include <unordered_set>
#include "interval_tree.hpp"

using ITree = lib_interval_tree::interval_tree_t<ptrdiff_t, lib_interval_tree::right_open>;

extern int opt_d;

template <typename T>
bool read_mem(const my_phdr *p, uint64_t addr, T &res ) {
  if ( addr < p->addr || addr + sizeof(T) >= (p->addr + p->memsz) ) return false;
  res = *(const T *)(addr);
  return true;
}

struct auto_dlclose {
  explicit auto_dlclose(void *v) : handle(v) {}
  ~auto_dlclose() {
    if ( handle != NULL ) dlclose(handle);
  }
  void *handle;
};

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

struct diter;

class decuda_base {
 public:
   decuda_base(ELFIO::elfio *rdr):
     m_rdr(rdr)
   {
   }
   ~decuda_base() {
     if ( m_rdr ) delete m_rdr;
   }
   int read();
   void dump_syms() const;
 protected:
   virtual int _read() = 0;
   bool in_sec(const std::optional<ELFIO::section *> &s, uint64_t addr) const {
     if ( !s.has_value() ) return false;
     return addr >= (*s)->get_address() && addr < ((*s)->get_address() + (*s)->get_size());
   }
   uint64_t send(const ELFIO::section *s) const {
     auto sa = s->get_address();
     return sa + s->get_size();
   }
   const char *sdata(const ELFIO::section *s, uint64_t off) const {
     auto sa = s->get_address();
     if ( off < sa || off >= sa + s->get_size() ) return nullptr;
     return (const char *)(s->get_data() + off - sa);
   }
   template <typename T>
   bool read(ELFIO::section *s, uint64_t off, T &res) {
     auto sa = s->get_address();
     if ( off < sa || off + sizeof(T) >= sa + s->get_size() ) return false;
     const T *ptr = (const T *)(s->get_data() + off - sa);
     res = *ptr;
     return true;
   }
   uint32_t read_size(ELFIO::section *, uint64_t off);
   uint64_t read_ptr(ELFIO::section *, uint64_t off);
   int cmp_str(ELFIO::section *, uint64_t off, const char *);
   int read_syms(ELFIO::section *);
   int read_rels(std::unordered_set<ELFIO::Elf_Half> &, int);
   // data
   bool is_32;
   ELFIO::elfio *m_rdr;
   std::vector<elf_reloc> m_relocs;
   std::map<std::string, elf_symbol> m_syms;
   // sections
   ELFIO::Elf_Half n_sec = 0;
   std::optional<ELFIO::section *> s_text, s_rodata, s_bss, s_data, s_data_rel;
};