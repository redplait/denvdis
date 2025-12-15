#include "decuda.h"
#include "bm_search.h"
#include <algorithm>

extern int opt_d;

void decuda::dump_syms() const {
  for ( auto &s: m_syms )
    printf("%lX %s %d sec %d\n", s.second.addr, s.first.c_str(), s.second.type, s.second.section);
}

int decuda::read_rels(std::unordered_set<ELFIO::Elf_Half> &r, int is_ra) {
  if ( r.empty() ) return 0;
  for ( auto ri: r ) {
    ELFIO::const_relocation_section_accessor rsa( *m_rdr, m_rdr->sections[ri]);
    auto n = rsa.get_entries_num();
    for ( ELFIO::Elf_Xword ri = 0; ri < n; ri++ ) {
      ELFIO::Elf64_Addr addr;
      ELFIO::Elf_Word sym;
      unsigned type;
      ELFIO::Elf_Sxword add;
      if ( rsa.get_entry(ri, addr, sym, type, add) ) {
        m_relocs.push_back( { addr, type, add, is_ra } );
      }
    }
  }
  return 1;
}

int decuda::read_syms(ELFIO::section *s) {
  ELFIO::symbol_section_accessor symbols( *m_rdr, s );
  ELFIO::Elf_Xword sym_no = symbols.get_symbols_num();
  if ( !sym_no )
  {
     fprintf(stderr, "no symbols\n");
     return 0;
  }
  for ( ELFIO::Elf_Xword i = 0; i < sym_no; ++i )
  {
    elf_symbol sym;
    sym.idx = i;
    symbols.get_symbol( i, sym.name, sym.addr, sym.size, sym.bind, sym.type, sym.section, sym.other );
    if ( sym.type == ELFIO::STT_SECTION ) continue;
    if ( sym.section == ELFIO::SHN_UNDEF ) continue;
      m_syms[sym.name] = std::move(sym);
  }
   return 1;
}

int decuda::read() {
  is_32 = m_rdr->get_class() == ELFIO::ELFCLASS32;
  // enum sections
  std::unordered_set<ELFIO::Elf_Half> s_rel, s_rela;
  n_sec = m_rdr->sections.size();
  for ( ELFIO::Elf_Half i = 0; i < n_sec; ++i ) {
    ELFIO::section *sec = m_rdr->sections[i];
    auto st = sec->get_type();
    if ( ELFIO::SHT_REL == st ) { s_rel.insert(i); continue; }
    if ( ELFIO::SHT_RELA == st ) { s_rela.insert(i); continue; }
    if ( st == ELFIO::SHT_SYMTAB || st == ELFIO::SHT_DYNSYM ) {
      if ( !read_syms(sec) ) return 0;
      continue;
    }
    // fill some sections
    if ( st == ELFIO::SHT_PROGBITS ) {
      if ( sec->get_name() == ".text" ) { s_text = sec; continue; }
      if ( sec->get_name() == ".rodata" ) { s_rodata = sec; continue; }
      if ( sec->get_name() == ".data" ) { s_data = sec; continue; }
      if ( sec->get_name() == ".data.rel.ro" ) { s_data_rel = sec; continue; }
    }
    if ( st == ELFIO::SHT_NOBITS ) {
      if ( sec->get_name() == ".bss" ) { s_bss = sec; continue; }
    }
  }
  // process relocs
  read_rels(s_rel, 0);
  read_rels(s_rela, 1);
  std::sort(m_relocs.begin(), m_relocs.end(), [](const elf_reloc &a, elf_reloc &b) { return a.offset < b.offset; });
  find_intf_tab();
  return 1;
}

const unsigned char first_intf[16] = {
 0x2C, 0x8E, 0x0A, 0xD8, 0x07, 0x10, 0xAB, 0x4E, 0x90, 0xDD, 0x54, 0x71, 0x9F, 0xE5, 0xF7, 0x4B };

int decuda::find_intf_tab() {
 // try to find first_intf in .rodata section
 if ( !s_rodata.has_value() ) return 0;
 if ( !s_data_rel.has_value() ) return 0;
 bm_search bm(first_intf, 16);
 const char *rs = (*s_rodata)->get_data();
 auto fres = bm.search((const unsigned char*)rs, (*s_rodata)->get_size());
 if ( !fres ) {
   printf("cannot find\n");
   return 0;
 }
 // cool. now read relocs in .data.rel.ro
 auto rstart = (*s_rodata)->get_address();
 auto addr = rstart + fres - (const unsigned char*)rs;
 if ( opt_d ) printf("addr %lX\n", addr);
 auto start = (*s_data_rel)->get_address();
 auto riter = std::lower_bound(m_relocs.begin(), m_relocs.end(), (ptrdiff_t)start,
  [](auto &what, ptrdiff_t off) { return what.offset < off; });
 if ( riter == m_relocs.end() ) {
   printf("cannot find relocs for .data.rel.ro\n");
   return 0;
 }
 auto end = start + (*s_data_rel)->get_size();
 const char *drs = (*s_data_rel)->get_data();
 for ( ; riter != m_relocs.end(); ++riter ) {
   if ( riter->offset >= end ) break;
   // read value at riter->offset
   auto r_addr = drs + (riter->offset - start);
   if ( addr == *(uint64_t *)r_addr ) {
     m_intf_tab = riter->offset;
     break;
   }
 }
 return m_intf_tab != 0;
}

void decuda::dump_res() const {
  if ( m_intf_tab )
    printf("intf_tab: %lX\n", m_intf_tab);
}

decuda *get_decuda(const char *fname) {
  ELFIO::elfio *rdr = new ELFIO::elfio;
  if ( !rdr->load(fname) ) {
    delete rdr;
    fprintf(stderr, "cannot load ELF %s\n", fname);
    return nullptr;
  }
  return new decuda(rdr);
}
