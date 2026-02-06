#include "de_bg.h"
#include "x64arch.h"
#include <queue>
#include "simple_api.h"
#include <cuda-gdb/cudadebugger.h>

extern int opt_d;

using AddrIdent = std::pair<uint64_t, int>;
using Q2 = std::queue<AddrIdent>;

void de_bg::dump_res() const {
  dump_tlg(m_tlg);
  if ( m_api )
    printf("api %lX\n", m_api);
  if ( m_state )
    printf("state %lX\n", m_state);
  if ( m_bg_log )
    printf("bg_log %lX\n", m_bg_log);
  if ( m_log_root )
    printf("log_root %lX\n", m_log_root);
  int idx = -1;
  for ( auto &api: m_apis ) {
    idx++;
    printf("[%d] %lX %lX", idx, api.addr, api.sub);
    if ( !api.name.empty() ) printf(" %s\n", api.name.c_str());
    else printf("\n");
  }
}

static const char *s_tlg[] = {
"dbg_cudbg_api",
"dbg_da_workload",
"dbg_da_thd",
"dbg_da_tgt",
"dbg_da_step",
"dbg_da_stack",
"dbg_da_suspend",
"dbg_da_spatcher",
"dbg_da_runctl",
"dbg_da_reg",
"dbg_profile",
"dbg_da_mod",
"dbg_da_mem",
"dbg_da_hal",
"dbg_da_gpu",
"dbg_da_init",
"dbg_da_func",
"dbg_da_etbl",
"dbg_da_evt",
"dbg_da_elf",
"dbg_da_dev",
"dbg_da_ctx",
"dbg_da_coredump",
"dbg_da_cache",
"dbg_da_bp",
"dbg_da",
};

static const char *s_api = "GetCUDADebuggerAPI";

int de_bg::_read() {
  process_tlg(s_tlg, sizeof(s_tlg) / sizeof(s_tlg[0]), m_tlg);
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
  try_hack_api(ri);
  return !m_apis.empty();
}

int de_bg::looks_name(uint64_t off, std::string &res) const {
  res.clear();
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

int de_bg::try_hack_api(std::vector<elf_reloc>::iterator &ri)
{
  uint64_t prev = 0;
  int res = 0;
  auto dend = send(s_data.value());
  diter di(*s_text);
  while( ri != m_relocs.end() && ri->offset < dend ) {
    if ( prev ) {
      if ( prev + sizeof(uint64_t) != ri->offset ) break;
    }
    prev = ri->offset;
    uint64_t val = 0;
    if ( !read(s_data.value(), ri->offset, val) ) break;
    if ( !in_sec(s_text, val) ) break;
    std::string name;
    try_one_api(di, val, name);
    if ( opt_d ) printf("val %lX\n", val);
    m_apis.push_back({ri->offset, val, name});
    res++;
    ri++;
  }
  return res;
}

int de_bg::try_one_api(diter &di, uint64_t off, std::string &res) {
 /* there can be 2 type of prologues
    1) check dbg_st and jmp to thunk like
     mov     rax, cs:dbg_st - state 0
     test rax, rax - state 1
     ...
     jmp thunk
    2) almost the same but
     call xxx - state 1 - collect them and process
  */
  if ( !di.setup(off) ) return 0;
  std::unordered_set<uint64_t> coll;
  used_regs<uint64_t> regs;
  int state = 0;
  while(1) {
    if ( !di.next() ) break;
    di.dasm(state);
    if ( !state ) {
      if ( di.is_mov64() && di.is_r1() ) {
        auto what = di.get_jmp(1);
        if ( in_sec(s_bss, what) ) {
          if ( !m_state ) {
             regs.add(di.ud_obj.operand[0].base, what);
             m_state = what;
             state = 1;
             continue;
          } else if ( m_state == what ) {
             regs.add(di.ud_obj.operand[0].base, what);
             state = 1;
             continue;
          }
        }
      }
    }
    if ( 1 == state ) {
      if ( di.is_test_rr() && di.ud_obj.operand[0].base == di.ud_obj.operand[1].base && regs.exists(di.ud_obj.operand[0].base) ) {
        if ( regs.exists(di.ud_obj.operand[0].base) ) {
          state = 2;
          continue;
        }
      }
    }
    if ( 2 == state ) {
      // check jmp and exit from while loop
      if ( di.is_jxx_jimm(UD_Ijmp) ) {
        coll.insert(di.get_jmp(0));
        break;
      }
      // check call and add to coll
      if ( di.is_jxx_jimm(UD_Icall) ) {
        coll.insert(di.get_jmp(0));
      }
    }
    if ( di.is_end() ) break;
  }
  if ( coll.empty() ) return 0;
  for ( auto addr: coll )
   if ( extract_name(di, addr, res) ) return 1;
  return 0;
}

int de_bg::extract_name(diter &di, uint64_t off, std::string &res) {
  // typical prolog looks like
  // mov     rax, 0FF43AA8B00000001h - imm64 - state 0
  // ...
  // lea reg, [rip + rodata] <- res, state 1
  // mov reg_log, [rip + data] - state 2
  // test reg_log, reg_log - state 3
  // call reg_log - state 4
  // -- 13.1 has log_root and checking of log function looks like
  // lea base_reg, [rip + data] - state 5
  // mov reg, [base_reg + off] - back to state 3
  if ( !di.setup(off) ) return 0;
  used_regs<uint64_t> regs;
  int state = 0;
  for ( int i = 0; i < 40; i++ ) {
    if ( !di.next() ) break;
    di.dasm(state);
    // 0: wait for mov reg, imm64
    if ( !state ) {
      if ( di.is_mov_rimm() && di.ud_obj.operand[1].size == 64 ) {
        state = 1;
        continue;
      }
    }
    // 1: lea from ro_data
    if ( !state || 1 == state ) {
      if ( di.is_lea() && di.is_r1() ) {
        if ( looks_name(di.get_jmp(1), res) ) {
          state = 2;
          if ( m_bg_log ) return 1;
          continue;
        }
      }
    }
    // 5: mov reg, [reg + off]
    if ( 5 == state ) {
      if ( di.is_movr() && !di.is_r1() ) {
        uint64_t base = 0;
        if ( regs.asgn(di.ud_obj.operand[1].base, base) ) {
          auto log_off = base + di.ud_obj.operand[1].lval.sdword;
          if ( in_sec(s_data, log_off) ) {
            regs.add(di.ud_obj.operand[0].base, log_off);
            state = 3;
            continue;
          }
        }
      }
    }
    // 2: mov reg, [data]
    if ( 2 == state ) {
      if ( di.is_mov64() && di.is_r1() && di.ud_obj.operand[0].size == 64 ) {
        auto addr = di.get_jmp(1);
        if ( in_sec(s_data, addr) ) {
          regs.add(di.ud_obj.operand[0].base, addr);
          state = 3;
          continue;
        }
      }
      // check 13.1
      if ( di.is_lea() && di.is_r1() ) {
        auto off = di.get_jmp(1);
        if ( !in_sec(s_data, off) ) continue;
        if ( !m_log_root ) m_log_root = off;
        regs.add(di.ud_obj.operand[0].base, off);
        state = 5;
        continue;
      }
    }
    // 3: test reg, reg
    if ( 3 == state ) {
      if ( di.is_test_rr() && di.ud_obj.operand[0].base == di.ud_obj.operand[1].base && regs.exists(di.ud_obj.operand[0].base) ) {
         state = 4;
         continue;
      }
    }
    // 4: call reg
    if ( 4 == state ) {
      if ( di.is_call_reg() ) {
        if ( regs.asgn(di.ud_obj.operand[0].base, m_bg_log) ) return 1;
        break;
      }
    }
    if ( di.is_end() ) return 0;
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

void de_bg::dump_logh(FILE *fp, const my_phdr *curr, int idx, uint64_t addr, rtmem_storage &rs) {
  uint64_t val = 0;
  if ( !read_mem(curr, addr, val) ) {
    fprintf(fp, "cannot read log_root+%X at %lX\n", idx, addr);
    return;
  }
  if ( !val ) {
    fprintf(fp, "log_root+%X at %lX: %lX\n", idx, addr, val);
    return;
  }
  auto who = rs.check(val);
  if ( who ) {
    fprintf(fp, "log_root+%X at %lX patched by %s: %lX\n", idx, addr, who->name_ref->c_str(), val);
  } else {
    fprintf(fp, "log_root+%X at %lX patched: %lX\n", idx, addr, val);
  }
}

int de_bg::vrf_log(FILE *fp, uint64_t delta, rtmem_storage &rs) {
  auto root = dbg_root();
  if ( !root ) return 0;
/* layout under 10 looks like
  0 - dword
  8 - dword
 10 - func ptr
 18 - dword
 28 till 230 - func ptrs
 */
  auto curr = rs.check(root + delta);
  if ( !curr ) {
    fprintf(fp, "dbg_root in unknown section, %lX\n", root);
    return 0;
  }
  // dump flags
  const static int flags_off[] = { 0, 8, 0x18, 0x20 };
  for ( auto i: flags_off ) {
    int flag = 0;
    auto addr = root + delta + i;
    if ( read_mem(curr, addr, flag) ) {
      fprintf(fp, "flag at %X: %X\n", i, flag);
    }
  }
  // dump handlers
  dump_logh(fp, curr, 0x10, root + delta + 0x10, rs);
  for ( int i = 0x28; i < 0x230; i += 8 ) {
    dump_logh(fp, curr, i, root + delta + i, rs);
  }
  return 1;
}

int de_bg::vrf_api(FILE *fp, uint64_t delta, rtmem_storage &rs) {
  if ( !m_api ) return 0;
  if ( m_state ) {
    auto addr = delta + m_state;
    auto curr = rs.check(addr);
    if ( curr ) {
      uint64_t state = 0;
      if ( read_mem(curr, addr, state) )
        fprintf(fp, "state at %lX: %lX\n", addr, state);
    }
  }
  int idx = -1;
  auto curr = rs.check(m_api + delta);
  if ( !curr ) {
    fprintf(fp, "api in unknown section, %lX\n", m_api);
    return 0;
  }
  for ( auto const &a: m_apis ) {
    ++idx;
    auto addr = delta + a.addr;
    uint64_t cb = 0;
    if ( !read_mem(curr, addr, cb) ) {
      fprintf(fp, "cannot read cb[%d] at %lX\n", idx, addr);
    } else {
      if ( cb == a.sub + delta ) continue;
      // something patched
      auto who = rs.check(cb);
      if ( who ) {
        fprintf(fp, "[%d] at %lX %s patched by %s\n", idx, addr, a.name.c_str(), who->name_ref->c_str());
      } else {
        fprintf(fp, "[%d] at %lX %s patched: %lX\n", idx, addr, a.name.c_str(), cb);
      }
    }
  }
  return 1;
}

// GetCUDADebuggerAPI store ptr to CUDBGAPI into [rcx], so prototype like
//                     rdi  rsi  rdx  rcx
typedef int (*und_api)(int, int, int, CUDBGAPI *api);

int patch_dbg_trace(FILE *fp, uint64_t addr);

int de_bg::verify(FILE *fp, rtmem_storage &rs, int hook, char tlg) {
  // extract delta
  auto si = m_syms.find(s_api);
  if ( si == m_syms.end() ) {
    fprintf(fp, "cannot get entry %s\n", s_api); fflush(fp);
    return 0;
  }
  auto dh = dlopen("libcudadebugger.so.1", 2);
  if ( !dh ) {
    fprintf(fp, "cannot load libcudadebugger, %s\n", dlerror()); fflush(fp);
    return 0;
  }
  auto_dlclose dummy(dh);
  uint64_t real_addr = (uint64_t)dlsym(dh, s_api);
  if ( !real_addr ) {
    fprintf(fp, "cannot find address of %s, (%s)\n", s_api, dlerror()); fflush(fp);
    return 0;
  }
  auto delta = real_addr - si->second.addr;
  fprintf(fp, "delta %lX\n", delta);
  CUDBGAPI api = nullptr;
  if ( hook ) {
    und_api hack = (und_api)real_addr;
    hack(0, 0, 0, &api);
    auto res = api->initialize();
    fprintf(fp, "init res: %X\n", res);
  }
  vrf_api(fp, delta, rs);
  vrf_log(fp, delta, rs);
  fflush(fp);
  if ( tlg ) patch_tlg(delta, tlg);
  if ( !hook ) return 1;
  auto addr_log = bg_log();
  if ( !addr_log ) return 0;
  int res = patch_dbg_trace(fp, addr_log + delta);
  // test logger
  if ( hook > 1 ) api->acknowledgeEvents42();
  return res;
}

int de_bg::patch_tlg(uint64_t delta, char value) {
  int res = 0;
  for ( auto &t: m_tlg ) {
    if ( !t.addr ) continue;
    char *addr = (char *)(t.addr + delta);
    // +0x8 - 1
    // +0xa - value
    *(addr + 8) = 1;
    *(addr + 0xa) = value;
    res++;
  }
  return res;
}

// simple api
int check_dbg(const char *fname, FILE *fp, int hook, char tlg_value) {
  if ( !fp ) fp = stdout;
  // read modules
  rtmem_storage rs;
  if ( !rs.read() ) {
    fprintf(fp, "cannot enum modules\n"); fflush(fp);
    return 0;
  }
  // check if we under debugger
  std::regex test_rx("libcudadebugger\\.so");
  if ( !rs.check_re(test_rx) ) {
    fprintf(fp, "libcudadebugger not detected\n"); fflush(fp);
    return 0;
  }
  // hack de_bg
  ELFIO::elfio *rdr = new ELFIO::elfio();
  if ( !rdr->load(fname) ) {
    fprintf(fp, "cannot load %s\n", fname); fflush(fp);
    delete rdr;
    return 0;
  }
  de_bg mod(rdr);
  if ( !mod.read() ) {
    fprintf(fp, "cannot hack %s\n", fname); fflush(fp);
    return 0;
  }
  return mod.verify(fp, rs, hook, tlg_value);
}