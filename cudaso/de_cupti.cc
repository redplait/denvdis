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

int de_cupti::fsm_log(diter &di, uint64_t off, uint64_t &res) {
  if ( !di.setup(off) ) return 0;
 /* prolog looks like
   mov     eax, cs:dbg_root <- state 0, 32 bit reg
   ...
   cmp     eax, 2 ; state 1
   jz do_log ; state 2, change address
 do_log:
   mov     rax, cs:off_4178D0 ; state 3
   jmp     rax
 */
  int state = 0;
  ud_type s1_reg = UD_NONE;
  used_regs<uint64_t> regs;
  while(1) {
    if ( !di.next() ) break;
    di.dasm(state);
    if ( !state ) {
      if ( di.is_mov32r() && di.is_r1() ) {
        // check that mem in .data section
        auto addr = di.get_jmp(1);
        if ( in_sec(s_data, addr) ) {
          if ( !m_dbg_root ) m_dbg_root = addr;
          else if ( m_dbg_root != addr ) continue;
          s1_reg = di.ud_obj.operand[0].base;
          state = 1;
          continue;
        }
      }
    }
    if ( 1 == state ) {
      if ( di.is_cmp_rimm(s1_reg) ) {
        state = 2;
        continue;
      }
    }
    if ( state > 1 ) {
      if ( di.is_jxx_jimm(UD_Ijz) ) {
        auto addr = di.get_jmp(0);
        if ( state == 2 && !di.setup(addr) ) break;
        state = 3;
        continue;
      }
      if ( di.is_mov64r() && di.is_r1() ) {
        // check that mem in .data section
        auto addr = di.get_jmp(1);
        if ( in_sec(s_data, addr) ) {
          regs.add(di.ud_obj.operand[0].base, addr);
          continue;
        }
      }
      if ( di.is_jmp_reg() ) {
        if ( regs.asgn(di.ud_obj.operand[0].base, res) ) return 1;
        break;
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
  diter di(s_text.value());
  for ( ; ri != m_relocs.end() && ri->offset < dend; ++ri ) {
    uint64_t val = 0;
    val = read_ptr(s_data.value(), ri->offset);
    // check if what we read inside data section - then stop
    if ( in_sec(s_data, val) ) break;
    if ( !in_sec(s_text, val) ) continue;
    if ( m_items.size() < 2 )
      m_items.push_back( { ri->offset, val } );
    else {
      uint64_t res = 0;
      fsm_log(di, val, res);
      m_items.push_back( { ri->offset, val, res } );
    }
  }
  return !m_items.empty();
}