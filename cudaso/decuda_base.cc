#include "decuda.h"
#include "ahocor.h"

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

int decuda_base::cmp_str(ELFIO::section *s, uint64_t off, const char *what) {
  auto rs = s->get_data() + (off - s->get_address());
  auto len = strlen(what);
  // check if string fit into section
  if ( rs + len >= s->get_data() + s->get_size() ) return 0;
  return !strcmp(rs, what);
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

int decuda_base::process_tlg(const char **names, size_t n_size, Tlg &res) {
  if ( !n_size || !s_rodata.has_value() || !s_data.has_value() ) return 0;
  std::vector<uint64_t> s_offs(n_size);
  // make string trie from names
  trie s_trie;
  for ( size_t i = 0; i < n_size; ++i ) {
    auto len = strlen(names[i]);
    s_trie.insert_iter(names[i], names[i] + 1 + len);
  }
  // try to search strings
  auto sro = s_rodata.value();
  auto start = sro->get_data();
  auto ro_off = sro->get_address();
  auto sres = s_trie.parse_text(start, start + sro->get_size());
  int added = 0;
  for ( auto citer = sres.cbegin(); citer != sres.end(); ++citer ) {
    auto off = citer->get_start();
    auto idx = citer->get_index();
    s_offs[idx] = ro_off + off;
    added |= 1;
  }
  if ( !added ) return 0;
  // second lookup - addresses in data section
  res.resize(n_size);
  auto dat = s_data.value();
  auto dstart = dat->get_data();
  auto d_off = dat->get_address();
  trie d_trie;
  for ( size_t i = 0; i < n_size; i++ )
  {
    if ( !s_offs[i] ) continue;
    char buf[8];
    *(uint64_t *)buf = s_offs[i];
    d_trie.insert_iter(buf, buf + 8, i);
  }
  sres = d_trie.parse_text(dstart, dstart + dat->get_size());
  added = 0;
  for ( auto citer = sres.cbegin(); citer != sres.end(); ++citer ) {
    auto off = citer->get_start();
    auto idx = citer->get_index();
    res[idx] = { names[idx], d_off + off };
    added++;
  }
  if ( !added ) res.clear();
  return added;
}

void decuda_base::dump_tlg(const Tlg &res) const {
  if ( res.empty() ) return;
  printf("Tlg: %d\n", res.size());
  for ( size_t i = 0; i < res.size(); ++i ) {
    auto &curr = res.at(i);
    if ( !curr.name || !curr.addr ) continue;
    printf(" [%d] %s - %lX\n", i, curr.name, curr.addr);
  }
}