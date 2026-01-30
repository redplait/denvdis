#include "de_bg.h"
#include "x64arch.h"
#include <queue>

using AddrIdent = std::pair<uint64_t, int>;
using Q2 = std::queue<AddrIdent>;

void de_bg::dump_res() const {
  if ( m_api )
    printf("api %lX\n", m_api);
}

static const char *s_api = "GetCUDADebuggerAPI";

int de_bg::_read() {
  auto si = m_syms.find(s_api);
  if ( si == m_syms.end() ) {
    fprintf(stderr, "cannot get entry %s\n", s_api);
    return 0;
  }
  auto api_addr = si->second.addr;
  if ( !in_sec(s_text, api_addr) ) {
    fprintf(stderr, "entry %s not in text section\n", s_api);
    return 0;
  }
  std::vector<elf_reloc>::iterator ri;
  try_api(api_addr, ri);
}

int de_bg::looks_name(uint64_t off, std::string &res) const {
  if ( !in_sec(s_rodata, off) ) return 0;
  auto data = sdata(s_rodata.value(), off);
  if ( !data ) return 0;
  for( int idx = 0; off < send(s_rodata.value()); idx++, data++ ) {
    char c = *data;
    if ( !c ) {
      if ( idx > 2 ) return 1;
      return 0;
    }
    if ( idx ) {
      if ( !isalnum(c) && c != '_' ) return 0;
    } else {
      if ( !isalpha(c) && c != '_' ) return 0;
    }
    res.push_back(c);
  }
  return 0;
}

int de_bg::try_api(uint64_t addr, std::vector<elf_reloc>::iterator &ri) {
  Q2 q;
  q.push({ addr, 0});
  diter di(*s_text);
  while( !q.empty() ) {
    auto addr = q.front();
    q.pop();
    if ( !di.setup(addr.first) ) continue;
    while( 1 ) {
      if ( !di.next() ) break;
      // check jmp
      if ( di.is_jxx_jimm(UD_Ijmp) ) {
        // add to queue
        auto jv = di.get_jmp(0);
        q.push({jv, addr.second});
        break;
      }
      // call
      if ( di.is_jxx_jimm(UD_Icall) ) {
        // add to queue
        auto jv = di.get_jmp(0);
        if ( !addr.second )
          q.push({jv, addr.second + 1});
        continue;
      }
      di.dasm();
      // lea [rip + data]
      if ( di.is_r1() && di.ud_obj.mnemonic == UD_Ilea ) {
        auto addr = di.get_jmp(1);
        if ( in_sec(s_data, addr) ) {
          ri = std::lower_bound(m_relocs.begin(), m_relocs.end(), (ptrdiff_t)addr,
             [](auto &what, ptrdiff_t off) { return what.offset < off; });
          if ( ri == m_relocs.end() ) continue;
          m_api = addr;
          goto out;
        }
      }
      if ( di.is_end() ) break;
    }
  }
out:
  return (m_api != 0);
}