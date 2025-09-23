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

struct one_indirect_branch {
 uint32_t addr;
 std::list<uint32_t> labels;
};

template <typename T>
class CElf: public T {
 public:
     int open(elfio *r, const char *fname, int opc = 0) {
     m_reader = r;
     if ( !m_reader->load(fname) ) {
       T::Err("cannot load\n");
       return 0;
     }
     if ( m_reader->get_machine() != 190 ) {
       T::Err("not CUBIN\n");
       return 0;
     }
     // try load smXX
     m_sm = (m_reader->get_flags() >> 0x10) & 0xff;
     if ( !m_sm ) m_sm = (m_reader->get_flags() >> 8) & 0xff;
     auto smi = NV_renderer::s_sms.find(m_sm);
     if ( smi == NV_renderer::s_sms.end() ) {
       T::Err("unknown SM %X\n", m_sm);
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
     m_sm_name = smi->second.second ? smi->second.second : smi->second.first;
     sm_name += m_sm_name;
     sm_name += ".so";
     if ( opc ) printf(".target sm_%d\n", m_sm);
     else printf("load %s\n", sm_name.c_str());
     return NV_renderer::load(sm_name);
   }
 protected:
   // key - section index, value - cb num (like 3 for .nv.constant3)
   bool gather_cbsections(std::map<Elf_Half, int> &out_res) {
     constexpr const char *spfx = ".nv.constant";
     constexpr size_t spfx_len = strlen(spfx);
     for ( Elf_Half i = 0; i < n_sec; ++i )
     {
       section* sec = m_reader->sections[i];
       if ( sec->get_type() != SHT_PROGBITS ) continue;
       if ( sec->get_info() ) continue; // like .nv.constant0.XX has info as reference to code section
       auto sname = sec->get_name();
       if ( !sname.starts_with(spfx) ) continue;
       // read cb index
       auto d = sname.data() + spfx_len;
       int idx = atoi(d);
       if ( idx ) out_res[i] = idx;
     }
     return !out_res.empty();
   }
   void fill_eaddrs(NV_labels *l, int ltype, const char *data, int alen) {
    for ( const char *bcurr = data + 4; data + 4 + alen - bcurr >= 0x4; bcurr += 0x4 )
    {
      uint32_t addr = *(uint32_t *)(bcurr);
      // there can be several labels for some addr, so add only if not exists yet
      auto ri = l->find(addr);
      if ( ri == l->end() )
        (*l)[addr] = ltype;
    }
   }
   // read EIATTR_INDIRECT_BRANCH_TARGETS and run callback C for each record
   template <typename C>
   int parse_branch_targets(const char *start, uint32_t len, C &&cb) {
     int res = 0;
     auto end = start + len;
     for ( auto curr = start; curr < end; ) {
       one_indirect_branch ib;
       // record like
       // 0 - 32bit address
       // 4 & 6 - unknown 16bit words
       // 8 - 32bit count
       // 0xc - ... - list of 32bit labels, count items
       // so minimal size should be 0xc
       if ( end - start < 0xc ) return 0;
       ib.addr = *(uint32_t *)(curr);
       uint32_t cnt = *(uint32_t *)(curr + 0x8);
       curr += 0xc;
       // read labels
       for ( uint32_t i = 0; i < cnt && curr < end; i++, curr += 4 ) ib.labels.push_back( *(uint32_t *)(curr) );
       // call cb
       cb(ib);
       res++;
     }
     return res;
   }
   template <typename F>
   int _read_symbols(int opt, F &&f) {
     section *sym_sec = nullptr;
     for ( Elf_Half i = 0; i < n_sec; ++i )
     {
       section* sec = m_reader->sections[i];
       if ( sec->get_type() == SHT_SYMTAB ) { sym_sec = sec; break; }
     }
     if ( !sym_sec ) return 0;
     // read symtab
     symbol_section_accessor symbols( *m_reader, sym_sec );
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
       section *sec = m_reader->sections[i];
       auto st = sec->get_type();
       if ( st == SHT_REL || st == SHT_RELA ) {
         auto slink = sec->get_info();
         section *ls = m_reader->sections[slink];
#ifdef DEBUG
 fprintf(this->m_out, "link %d %s\n", slink, ls->get_name().c_str());
#endif
         auto st2 = ls->get_type();
         if ( st2 == SHT_NOBITS || !ls->get_size() ) continue;
         if ( strncmp(ls->get_name().c_str(), ".text.", 6) ) continue;
         // yup, this is our client
         const_relocation_section_accessor rsa( *m_reader, sec);
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
   elfio *m_reader = nullptr;
   // SM number & name
   int m_sm = 0;
   const char *m_sm_name = nullptr;
   // symbols
   std::vector<asymbol> m_syms;
};