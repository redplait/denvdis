#include "decuda.h"

void decuda_base::dump_syms() const {
  for ( auto &s: m_syms )
    printf("%lX %s %d sec %d\n", s.second.addr, s.first.c_str(), s.second.type, s.second.section);
}

int decuda_base::read_rels(std::unordered_set<ELFIO::Elf_Half> &r, int is_ra) {
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

int decuda_base::read_syms(ELFIO::section *s) {
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
    if ( sym.type == ELFIO::STT_SECTION || sym.type == ELFIO::STT_FILE ) continue;
    if ( sym.section == ELFIO::SHN_UNDEF ) continue;
      m_syms[sym.name] = std::move(sym);
  }
  return 1;
}

uint32_t decuda_base::read_size(ELFIO::section *s, uint64_t off) {
  auto rs = s->get_data() + (off - s->get_address());
  return *(uint32_t *)rs;
}

uint64_t decuda_base::read_ptr(ELFIO::section *s, uint64_t off) {
  auto rs = s->get_data() + (off - s->get_address());
  return *(uint64_t *)rs;
}

int decuda_base::read() {
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
  return _read();
}