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
  return try_ext(api_addr);
}