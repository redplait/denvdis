#include "de_cupti.h"
#include "x64arch.h"
#include <queue>
#include "simple_api.h"

extern int opt_d;

void de_cupti::dump_res() const {
  if ( m_cupti_root )
    printf("cupti_root: %lX\n", m_cupti_root);
  if ( m_dbg_root )
    printf("dbg_root: %lX\n", m_dbg_root);
  for ( const auto &r: m_items ) {
    printf(" %lX - %lX", r.addr, r.value);
    if ( r.ind ) printf(" %lX\n", r.ind);
    else printf("\n");
  }
}

static const char *s_ext = "InitializeInjectionNvtxExtension";
static const char *s_marker = "Cupti_Public";

int de_cupti::try_ext(uint64_t off) {
  diter di(*s_text);
  if ( !di.setup(off) ) return 0;
  for ( int i = 0; i < 10; ++i ) {
    if ( !di.next() ) break;
    di.dasm();
    // check lea reg, [rip + data]
    if ( di.is_lea() && di.is_r1() ) {
      auto res = di.get_jmp(1);
      if ( !in_sec(s_data, res) ) continue;
      // check marker
      uint64_t mark_addr = read_ptr(s_data.value(), res);
      if ( !in_sec(s_rodata, mark_addr) ) break;
      if ( cmp_str(s_rodata.value(), mark_addr, s_marker) ) {
        m_cupti_root = res;
        return 1;
      }
    }
    if ( di.is_end() ) break;
  }
  return 0;
}

int de_cupti::_read() {
  // 1) find cupti_root from InitializeInjectionNvtxExtension
  auto si = m_syms.find(s_ext);
  if ( si == m_syms.end() ) {
    fprintf(stderr, "cannot get entry %s\n", s_ext);
    return 0;
  }
  auto api_addr = si->second.addr;
  if ( !in_sec(s_text, api_addr) ) {
    fprintf(stderr, "entry %s not in text section\n", s_ext);
    return 0;
  }
  if ( !try_ext(api_addr) ) return 0;
  // read function pointers
  auto ri = std::lower_bound(m_relocs.begin(), m_relocs.end(), (ptrdiff_t)m_cupti_root,
             [](auto &what, ptrdiff_t off) { return what.offset < off; });
  if ( ri == m_relocs.end() ) return 0;
  auto dend = send(s_data.value());
  for ( ; ri != m_relocs.end() && ri->offset < dend; ++ri ) {
    uint64_t val = 0;
    val = read_ptr(s_data.value(), ri->offset);
    // check if what we read inside data section - then stop
    if ( in_sec(s_data, val) ) break;
    if ( !in_sec(s_text, val) ) continue;
    m_items.push_back( { ri->offset, val } );
  }
  return !m_items.empty();
}