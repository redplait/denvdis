#pragma once
// cubin elf boring logic - header-only

#include "nv_rend.h"
#include "elfio/elfio.hpp"

using namespace ELFIO;

struct asymbol
{
  std::string name;
  Elf64_Addr addr;
  Elf_Xword idx = 0;
  Elf_Xword size = 0;
  Elf_Half section;
  unsigned char bind = 0,
                type = 0,
                other = 0;
};

template <typename T>
class CElf: public T {
 public:
     int open(const char *fname, int opc = 0) {
     if ( !reader.load(fname) ) {
       fprintf(stderr, "cannot load\n");
       return 0;
     }
     if ( reader.get_machine() != 190 ) {
       fprintf(stderr, "not CUBIN\n");
       return 0;
     }
     // try load smXX
     int sm = (reader.get_flags() >> 0x10) & 0xff;
     if ( !sm ) sm = (reader.get_flags() >> 8) & 0xff;
     auto smi = NV_renderer::s_sms.find(sm);
     if ( smi == NV_renderer::s_sms.end() ) {
       fprintf(stderr, "unknown SM %X\n", sm);
       return 0;
     }
     // check SM_DIR env
     std::string sm_name;
     char *sm_dir = getenv("SM_DIR");
     if ( sm_dir ) {
      sm_name = sm_dir;
      if ( !sm_name.ends_with("/") ) sm_name += '/';
     } else {
      sm_name = "./";
     }
     sm_name += smi->second.second ? smi->second.second : smi->second.first;
     sm_name += ".so";
     if ( opc ) printf(".target sm_%d\n", sm);
     else printf("load %s\n", sm_name.c_str());
     return NV_renderer::load(sm_name);
   }
 protected:
   template <typename F>
   int _read_symbols(int opt, F &&f) {
     section *sym_sec = nullptr;
     for ( Elf_Half i = 0; i < n_sec; ++i )
     {
       section* sec = reader.sections[i];
       if ( sec->get_type() == SHT_SYMTAB ) { sym_sec = sec; break; }
     }
     if ( !sym_sec ) return 0;
     // read symtab
     symbol_section_accessor symbols( reader, sym_sec );
     Elf_Xword sym_no = symbols.get_symbols_num();
     if ( !sym_no )
     {
       fprintf(this->m_out, "no symbols\n");
       return 0;
     }
     if ( opt ) {
       fprintf(this->m_out, "%ld symbols\n", sym_no);
     }
     m_syms.reserve(sym_no);
     for ( Elf_Xword i = 0; i < sym_no; ++i )
     {
        asymbol sym;
        sym.idx = i;
        symbols.get_symbol( i, sym.name, sym.addr, sym.size, sym.bind, sym.type, sym.section, sym.other );
        if ( opt ) {
          if ( sym.type != STT_SECTION )
            fprintf(this->m_out, " [%ld] %lX sec %d type %d %s\n", i, sym.addr, sym.section, sym.type, sym.name.c_str());
        }
        f(sym);
     }
     return 1;
   }
   int fill_rels() {
     for ( Elf_Half i = 0; i < n_sec; ++i ) {
       section *sec = reader.sections[i];
       auto st = sec->get_type();
       if ( st == SHT_REL || st == SHT_RELA ) {
         auto slink = sec->get_info();
         section *ls = reader.sections[slink];
#ifdef DEBUG
 fprintf(this->m_out, "link %d %s\n", slink, ls->get_name().c_str());
#endif
         auto st2 = ls->get_type();
         if ( st2 == SHT_NOBITS || !ls->get_size() ) continue;
         if ( strncmp(ls->get_name().c_str(), ".text.", 6) ) continue;
         // yup, this is our client
         const_relocation_section_accessor rsa(reader, sec);
         auto n = rsa.get_entries_num();
         SRels srels;
         for ( Elf_Xword ri = 0; ri < n; ri++ ) {
           Elf64_Addr addr;
           Elf_Word sym;
           unsigned type;
           Elf_Sxword add;
           if ( rsa.get_entry(ri, addr, sym, type, add) ) {
             srels[addr] = { type, sym };
           }
         }
#ifdef DEBUG
      fprintf(this->m_out, "store %ld relocs for section %d %s\n", srels.size(), slink, ls->get_name().c_str());
#endif
         if ( !srels.empty() )
         {
           auto prev = m_srels.find(slink);
           if ( prev == m_srels.end() )
             m_srels[slink] = std::move(srels);
           else {
             SRels &old = prev->second;
             for ( auto &p: srels ) old[p.first] = std::move(p.second);
           }
         }
       }
     }
     return !m_srels.empty();
   }
   void dump_csym(const asymbol *as) const {
    if ( as->bind == STB_GLOBAL )
      fprintf(this->m_out, "\t.global %s\n", as->name.c_str());
    if ( as->type == STT_OBJECT )
      fprintf(this->m_out, "\t.type %s,@object\n", as->name.c_str());
    else if ( as->type == STT_FUNC )
      fprintf(this->m_out, "\t.type %s,@function\n", as->name.c_str());
    if ( as->size )
      fprintf(this->m_out, "\t.size %lX\n", as->size);
    if ( as->other ) {
      fprintf(this->m_out, "\t.other %s, @\"", as->name.c_str());
      char upE = as->other & 0xE0;
      char lo2 = as->other & 3;
      int idx = 0;
    // as far I understod order is not matters
#define _DA(c) { if ( idx ) fputc(' ', this->m_out); fprintf(this->m_out, "%s", c); idx++; }
      if ( as->other & 0x10 ) _DA("STO_CUDA_ENTRY")
      if ( as->other & 0x4 )  _DA("STO_CUDA_MANAGED")
      if ( as->other & 0x8 )  _DA("STO_CUDA_OBSCURE")
      if ( upE == 0x20 ) _DA("STO_CUDA_GLOBAL")
      if ( upE == 0x40 ) _DA("STO_CUDA_SHARED")
      if ( upE == 0x60 ) _DA("STO_CUDA_LOCAL")
      if ( upE == 0x80 ) _DA("STO_CUDA_CONSTANT")
      if ( upE == 0xa0 ) _DA("STO_CUDA_RESERVED_SHARED")
      if ( lo2 ) {
        if ( lo2 == 1 ) _DA("STV_INTERNAL")
        else if ( lo2 == 2 ) _DA("STV_HIDDEN")
        else if ( lo2 == 3 ) _DA("STV_PROTECTED")
      } else
       _DA("STV_DEFAULT")
      fprintf(this->m_out, "\"\n");
#undef _DA
    }
   }
   // relocs. key - offset
   typedef std::map<unsigned long, NV_renderer::NV_rel> SRels;
   // key - section index
   std::unordered_map<int, SRels> m_srels;

   Elf_Half n_sec = 0;
   elfio reader;
   // symbols
   std::vector<asymbol> m_syms;
};