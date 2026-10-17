#include "ctr_str.h"
#include "x64arch.h"

int g_opt_r = 0;

const char *ctr_str::look_like_str(unsigned long sa) {
  if ( !in_sec(s_rodata, sa) ) return nullptr;
  const unsigned char *s = (const unsigned char *)sdata(s_rodata, sa);
  if ( !s ) return nullptr;
  if ( !s[0] ) return nullptr;
  if ( !isalnum(s[0]) || s[0] > 0x7f || !s[1] ) return nullptr;
  // check first 6 symbols
  for ( int i = 1; i < 7; ++i ) {
    if ( !s[i] ) break;
    if ( s[i] > 0x7f ) return nullptr;
  }
  return (const char *)s;
}

int ctr_str::proc_section(ELFIO::section *s) {
  int res = 0;
  diter di(*s_text);
  auto s_end = s->get_size() / 8;
  const unsigned long *s_body = (const unsigned long *)s->get_data();
  for ( m_idx = 0; m_idx < s_end; ++m_idx ) {
    if ( !s_body[m_idx] ) continue;
// printf("%lX %lX - %lX\n", s_body[m_idx], s_text.value()->get_address(), s_text.value()->get_size());
    if ( !in_sec(s_text, s_body[m_idx]) ) continue;
    m_latch = false;
    res += process_item(di, s_body[m_idx]);
  }
  return res;
}

int ctr_str::process_item(diter &di, unsigned long ca, bool second) {
  auto vi = m_visited.find(ca);
  if ( vi != m_visited.end() ) return 0;
  m_visited.insert(ca);
  if ( !di.setup(ca) ) return 0;
  int dumped = 0;
  while(1) {
    if ( !di.next() ) break;
    if ( di.is_jxx_jimm(UD_Ijmp) && g_opt_r && !second ) {
      auto what = di.get_jmp(0);
      return process_item(di, what, true);
    }
    if ( di.is_end() ) break;
    if ( di.is_lea() && di.is_r1() ) {
      auto what = di.get_jmp(1);
      auto s = look_like_str(what);
      if ( s ) {
        if ( !m_latch ) {
          printf("[%d] at %lX:\n", m_idx, ca);
          m_latch = true;
        }
        printf(" %s\n", s);
        dumped++;
      }
    }
  }
  return dumped;
}