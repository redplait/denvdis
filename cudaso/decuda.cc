#include "decuda.h"
#include "bm_search.h"
#include <algorithm>
#include "x64arch.h"
#include "rtmem.h"
#include <dlfcn.h>

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
    if ( sym.type == ELFIO::STT_SECTION || sym.type == ELFIO::STT_FILE ) continue;
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
  resolve_indirects();
  return 1;
}

template <typename T>
bool decuda::read(ELFIO::section *s, uint64_t off, T &res) {
  auto sa = s->get_address();
  if ( off < sa || off + sizeof(T) >= sa + s->get_size() ) return false;
  const T *ptr = (const T *)(s->get_data() + off - sa);
  res = *ptr;
  return true;
}

template <typename S>
void try_indirect(diter *di, S &&s) {
/* simple FSM: 0 - wait for cmp 0x321CBA00
 * 1 - wait for jz/jnz
 * 2 - wait for jmp/call [data] <- call clojure s for it
 */
  int state = 0;
  while( 1 ) {
    if ( !di->next() ) break;
    di->dasm(state);
    if ( !state ) {
      if ( di->is_imm(UD_Icmp, 1) && di->ud_obj.operand[1].lval.sdword == 0x321CBA00 ) {
        state = 1;
        continue;
      }
    }
    if ( 1 == state ) {
      if ( di->is_jxx_jimm(UD_Ijz) ) {
        state = 2;
        continue;
      }
    }
    if ( 2 == state ) {
      if ( di->is_mrip(0, UD_Icall, UD_Ijmp) ) {
        s(di->get_jmp(0));
        return;
      }
    }
    if ( di->is_end() ) break;
  }
}

int decuda::resolve_indirects()
{
  if ( !s_text.has_value() ) return 0;
  // addresses cache for weak symbols/synonyms
  std::unordered_set<uint64_t> cache;
  // enum symbols
  diter di(*s_text);
  for ( auto &s: m_syms ) {
    auto ip = cache.find(s.second.addr);
    if ( ip != cache.end() ) continue;
    if ( !in_sec(s_text, s.second.addr) ) continue;
    if ( s.second.type != ELFIO::STT_FUNC ) continue;
    // put address in cache
    cache.insert(s.second.addr);
    if ( !di.setup(s.second.addr) ) continue;
    try_indirect(&di, [&](uint64_t addr) {
      uint64_t val = 0;
      if ( in_sec(s_data, addr) && read(*s_data, addr, val) ) m_forwards[s.first] = { addr, val };
    });
  }
  return !m_forwards.empty();
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
 auto prev = riter->offset;
 if ( !m_intf_tab || ++riter == m_relocs.end() ) return 0;
 // dirty hack - read till relocs are adjacent
 one_intf curr;
 int state = 1;
 memcpy(&curr.uuid, first_intf, 16);
 auto cnt = drs + (riter->offset - start);
 for ( ; riter != m_relocs.end(); ++riter, cnt += 8 ) {
   if ( opt_d ) printf("state %d off %lX\n", state, riter->offset);
   if ( riter->offset >= end ) break;
   if ( riter->offset != 8 + prev ) break;
   if ( 1 == state ) { // read addr and size
     curr.addr = *(uint64_t *)cnt;
     if ( !curr.addr ) break;
     curr.size = read_size(curr.addr);
   } else { // read uuid
    uint64_t uid_addr = *(uint64_t *)cnt;
// printf("uid_addr %lX\n", uid_addr);
    if ( !uid_addr ) break;
    if ( !in_sec(s_rodata, uid_addr) ) break;
    auto ua = rs + uid_addr - rstart;
    memcpy(&curr.uuid, (const unsigned char *)ua, 16);
   }
   // for next cycle
   if ( 2 == ++state ) {
     m_intfs.push_back(curr);
     memset(&curr, 0, sizeof(curr));
     state = 0;
   }
   prev = riter->offset;
 }
 return !m_intfs.empty();
}

uint32_t decuda::read_size(ELFIO::section *s, uint64_t off) {
  auto rs = s->get_data() + (off - s->get_address());
  return *(uint32_t *)rs;
}

uint32_t decuda::read_size(uint64_t off) {
  if ( in_sec(s_data, off) ) return read_size(*s_data, off);
  if ( in_sec(s_data_rel, off) ) return read_size(*s_data_rel, off);
  return 0;
}

void decuda::dump_res() const {
  if ( m_intf_tab ) {
    printf("intf_tab: %lX\n", m_intf_tab);
    for ( auto &oi: m_intfs ) {
      // dump UUID
      printf("%8.8X-%4.4hX-%4.4hX-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X",
       *(uint32_t *)(oi.uuid), *(unsigned short *)(oi.uuid + 4), *(unsigned short *)(oi.uuid + 6),
       oi.uuid[8], oi.uuid[9], oi.uuid[10], oi.uuid[11], oi.uuid[12], oi.uuid[13], oi.uuid[14], oi.uuid[15]);
      printf(" %lX", oi.addr);
      if ( oi.size ) printf(" size %X\n", oi.size);
      else printf("\n");
    }
  }
  if ( !m_forwards.empty() ) {
    printf("%ld forwards:\n", m_forwards.size());
    for ( auto &fi: m_forwards ) {
      printf("%.*s: %lX %lX\n", fi.first.size(), fi.first.data(), fi.second.first, fi.second.second);
    }
  }
}

// verify methods
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

void decuda::verify(FILE *out_fp) const {
 auto first_sym = m_syms.cbegin();
 // get delta
 const char *fname = first_sym->first.c_str();
 auto dh = dlopen("libcuda.so.1", 2);
 if ( !dh ) {
   fprintf(out_fp, "cannot load libcuda, %s\n", dlerror());
   return;
 }
 auto_dlclose dummy(dh);
 uint64_t real_addr = (uint64_t)dlsym(dh, fname);
 if ( !real_addr ) {
   fprintf(out_fp, "cannot find address of %s, (%s)\n", fname, dlerror());
   return;
 }
 auto delta = real_addr - first_sym->second.addr;
 fprintf(out_fp, "real_addr %lX, delta %lX\n", real_addr, delta);
 rtmem_storage rs;
 if ( !rs.read() ) {
   fprintf(out_fp, "cannot read addresses, delta %lX\n", delta);
   return;
 }
 const my_phdr *curr = nullptr;
 // enum and dump
 for ( auto &fi: m_forwards ) {
   auto addr = delta + fi.second.first;
   if ( !curr ) curr = rs.check(addr);
   if ( !curr ) {
     fprintf(out_fp, "cannot resolve module for addr %lX (%lX)\n", addr, fi.second.first);
     continue;
   }
   // read ptr at addr - must be fi.second.second + delta
   uint64_t read_addr = 0;
   if ( !read_mem(curr, addr, read_addr) ) {
     fprintf(out_fp, "read %.*s at %lX failed\n", fi.first.size(), fi.first.data(), addr);
     continue;
   }
   if ( opt_d )
     fprintf(out_fp, "%.*s: %lX\n", fi.first.size(), fi.first.data(), read_addr);
   if ( delta + fi.second.second == read_addr ) continue;
   // report
   auto adr_name = rs.find(read_addr);
   if ( adr_name ) {
     fprintf(out_fp, "patched %.*s (%lX) - %s\n", fi.first.size(), fi.first.data(), read_addr, adr_name->c_str());
   } else {
     fprintf(out_fp, "patched %.*s (%lX)\n", fi.first.size(), fi.first.data(), read_addr);
   }
 }
 dump_bss_publics(out_fp, delta);
}

template <typename T>
void s_print(FILE *, T);

template <>
void s_print(FILE *fp, uint32_t v) {
  fprintf(fp, "%X", v);
}

template <>
void s_print(FILE *fp, uint64_t v) {
  fprintf(fp, "%lX", v);
}

template <>
void s_print(FILE *fp, pid_t v) {
  fprintf(fp, "%d", v);
}

bool decuda::dump_str_with_len(FILE *fp, const char *addr_name, const char *len_name, int64_t delta) const {
  auto si_addr = m_syms.find(addr_name);
  if ( si_addr == m_syms.end() ) return false;
  auto si_len = m_syms.find(len_name);
  if ( si_len == m_syms.end() ) return false;
  // check length - 8 bytes
  size_t len = *(size_t *)(delta + si_len->second.addr);
  if ( !len ) return true;
  const char *addr = *(char **)(delta + si_addr->second.addr);
  if ( !addr ) return true;
  fprintf(fp, "%s at %p len %X: %.*s\n", addr_name, addr, len, len, addr);
  return true;
}

template <typename T>
bool decuda::dump_xxx(FILE *fp, const char *pubname, int64_t delta) const {
  auto si = m_syms.find(pubname);
  if ( si == m_syms.end() ) return false;
  T *addr = (T*)(delta + si->second.addr);
  fprintf(fp, "%s at %p: ", pubname, addr);
  s_print(fp, *addr);
  fprintf(fp, "\n");
  return true;
}

void decuda::dump_bss_publics(FILE *fp, int64_t delta) const {
 if ( !s_bss.has_value() ) return;
 /* some other interesting data (mostly from .bss):
     cudbgUseExternalDebugger - 4 bytes
     cudbgReportedDriverInternalErrorCode - 8 bytes
     cudbgRpcEnabled - 4
     cudbgResumeForAttachDetach - 4
     cudbgDebuggerInitialized - 4
     cudbgDebuggerCapabilities - 4
     cudbgAttachHandlerAvailable - 4
     cudbgApiClientRevision - 4
     cudbgSessionId - 4
     cudbgApiClientPid - 4
     cudbgEnablePreemptionDebugging - 4
     cudbgEnableLaunchBlocking - 4
     cudbgReportedDriverApiErrorFuncNameAddr - 8
     cudbgReportedDriverApiErrorFuncNameSize - 8
     cudbgReportedDriverApiErrorCode - 8
     cudbgReportDriverApiErrorFlags - 4
     cudbgEnableIntegratedMemcheck - 4
     cudbgDetachSuspendedDevicesMask - 4
     cudbgInjectionPath - 0x1000
   from 13.1 sdk
     cudbgIpcFlag - 4 bytes
     cudbgReportedDriverApiErrorStringAddr - 8
     cudbgReportedDriverApiErrorStringSize - 8
     cudbgReportedDriverApiErrorNameAddr - 8
     cudbgReportedDriverApiErrorNameSize - 8
     cudbgReportedDriverApiErrorSource - 8
  */
  dump_xxx<uint32_t>(fp, "cudbgUseExternalDebugger", delta);
  dump_xxx<uint64_t>(fp, "cudbgReportedDriverInternalErrorCode", delta);
  dump_xxx<uint32_t>(fp, "cudbgRpcEnabled", delta);
  dump_xxx<uint32_t>(fp, "cudbgResumeForAttachDetach", delta);
  dump_xxx<uint32_t>(fp, "cudbgDebuggerInitialized", delta);
  dump_xxx<uint32_t>(fp, "cudbgDebuggerCapabilities", delta);
  dump_xxx<uint32_t>(fp, "cudbgAttachHandlerAvailable", delta);
  dump_xxx<uint32_t>(fp, "cudbgApiClientRevision", delta);
  dump_xxx<pid_t>(fp, "cudbgSessionId", delta);
  dump_xxx<pid_t>(fp, "cudbgApiClientPid", delta);
  dump_xxx<uint32_t>(fp, "cudbgEnablePreemptionDebugging", delta);
  dump_xxx<uint32_t>(fp, "cudbgEnableLaunchBlocking", delta);
  dump_xxx<uint64_t>(fp, "cudbgReportedDriverApiErrorCode", delta);
  dump_xxx<uint32_t>(fp, "cudbgReportDriverApiErrorFlags", delta);
  dump_xxx<uint32_t>(fp, "cudbgEnableIntegratedMemcheck", delta);
  dump_xxx<uint32_t>(fp, "cudbgDetachSuspendedDevicesMask", delta);
  dump_xxx<uint32_t>(fp, "cudbgIpcFlag", delta);
  dump_xxx<uint64_t>(fp, "cudbgReportedDriverApiErrorSource", delta);
  dump_str_with_len(fp, "cudbgReportedDriverApiErrorFuncNameAddr", "cudbgReportedDriverApiErrorFuncNameSize", delta);
  dump_str_with_len(fp, "cudbgReportedDriverApiErrorStringAddr", "cudbgReportedDriverApiErrorStringSize", delta);
  dump_str_with_len(fp, "cudbgReportedDriverApiErrorNameAddr", "cudbgReportedDriverApiErrorNameSize", delta);
}

/*
 * simple API
 */
decuda *get_decuda(const char *fname) {
  ELFIO::elfio *rdr = new ELFIO::elfio;
  if ( !rdr->load(fname) ) {
    delete rdr;
    fprintf(stderr, "cannot load ELF %s\n", fname);
    return nullptr;
  }
  return new decuda(rdr);
}

void check_cuda(const char *fname, FILE *fp) {
  auto obj = get_decuda(fname);
  if ( obj->read() )
   obj->verify(fp);
  else
   fprintf(fp, "cannot read %s\n", fname);
  delete obj;
}