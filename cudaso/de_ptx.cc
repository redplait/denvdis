#include "de_ptx.h"
#include "x64arch.h"
#include <queue>
#include <regex>

extern int opt_d;

int de_ptx::dump_cicc(const char *fname, const cicc_names &cn) {
  FILE *fp = fopen(fname, "w");
  if ( !fp ) {
    fprintf(stderr, "cannot create %s, error %d (%s)\n", fname, errno, strerror(errno));
    return 0;
  }
  for ( const auto ci: cn ) {
    fprintf(fp,"%d %.*s\n", ci.second, ci.first.size(), ci.first.data());
  }
  fclose(fp);
  return 1;
}

int de_ptx::dump_deres(const char *fname, const res_map &rm) {
  FILE *fp = fopen(fname, "w");
  if ( !fp ) {
    fprintf(stderr, "cannot open %s, error %d (%s)\n", fname, errno, strerror(errno));
    return 0;
  }
  auto first = rm.cbegin();
  for ( auto fi = first; fi != rm.cend(); ++fi ) {
    fprintf(fp, "%lX: ", fi->first - first->first);
    if ( fi->second.what == 1 )
      fprintf(fp, "-\n");
    else if ( fi->second.what == 2 )
      fprintf(fp, "%s\n", fi->second.dec.c_str());
    else
      fprintf(fp, "%d\n", fi->second.num);
  }
  fclose(fp);
  return 1;
}

int de_ptx::dump_objoff(const char *fname, const res_map &rm) {
  FILE *fp = fopen(fname, "w");
  if ( !fp ) {
    fprintf(stderr, "cannot open %s, error %d (%s)\n", fname, errno, strerror(errno));
    return 0;
  }
  for ( auto fi = rm.begin(); fi != rm.cend(); ++fi ) {
    fprintf(fp, "%lX: ", fi->first);
    if ( fi->second.what == 1 )
      fprintf(fp, "-\n");
    else if ( fi->second.what == 2 )
      fprintf(fp, "%s\n", fi->second.dec.c_str());
    else
      fprintf(fp, "%d\n", fi->second.num);
  }
  fclose(fp);
  return 1;
}

void de_ptx::hack_cicc_intr(uint64_t off, const char *fname) {
  diter di(*s_text);
  if ( !di.setup(off) ) return;
  cicc_names res;
  if ( hack_cicc(di, res) ) dump_cicc(fname, res);
}

void de_ptx::hack_ctor(uint64_t off, const char *fname, int in_bss) {
  diter di(*s_text);
  if ( !di.setup(off) ) return;
  res_map res;
  if ( hack(di, res, in_bss) ) {
    if ( in_bss )
      dump_deres(fname, res);
    else
      dump_objoff(fname, res);
  }
}

void de_ptx::hack_ops(uint64_t off, uint64_t croot, const char *fname) {
  diter di(*s_text);
  if ( !di.setup(off) ) return;
  res_map res;
  if ( hack_ops(di, croot, res) ) dump_deres(fname, res);
}

void de_ptx::hack_sp(uint64_t off, const char *fname) {
  diter di(*s_text);
  if ( !di.setup(off) ) return;
  res_map res;
  hack_sp(di, res);
  if ( !res.empty() ) dump_deres(fname, res);
}

int de_ptx::check(lat_res &r, uint64_t off) {
  if ( !in_sec(s_rodata, off) ) return 0;
  auto s = sdata(s_rodata.value(), off);
  if ( !s ) return 0;
  if ( s[0] == '-' && !s[1] ) {
    r.what = 1;
    return 1;
  }
  // decrypt
  r.what = 2;
  while( *s ) {
    auto c = *s;
    if ( c >= 0x41 ) {
      auto mask = c & 0xDF;
      auto si = mask - 0x41;
      if ( si <= 0xc ) {
       c = c + 0xd;
      } else {
        si = mask - 0x4e;
        if ( si < 0x0d ) {
         c = c - 0xd;
        }
      }
    }
    r.dec.push_back(c);
    ++s;
  }
  return 1;
}

extern int opt_d;

void report(diter &di, const char *pfx) {
  printf("%s at %lX\n", pfx, ud_insn_off(&di.ud_obj));
}

template <typename T>
int de_ptx::hack_cicc(diter &di, T &cn) {
 // states:
 // 0 - initial
 // 1 - rsi got string
 // 2 - call
 // 3 - mov [rax+8], const, reset to 0
 using KT = typename T::key_type;
 KT name, prev;
 int state = 0;
 while(1) {
    if ( !di.next() ) break;
    di.dasm(state);
    if ( di.ud_obj.mnemonic == UD_Icall ) { prev = name; state = 2; continue; }
    if ( di.is_lea() && di.is_r1() && di.ud_obj.operand[0].base == UD_R_RSI ) {
      auto res = di.get_addr(1);
      if ( !in_sec(s_rodata, res) ) continue;
      name = {};
      if ( read_str(s_rodata.value(), res, name) ) state = 1;
    } else if ( di.ud_obj.mnemonic == UD_Imov && di.ud_obj.operand[1].type == UD_OP_IMM &&
             di.ud_obj.operand[0].type == UD_OP_MEM && di.ud_obj.operand[0].lval.sdword == 8 ) {
      cn[prev] = di.ud_obj.operand[1].lval.sdword;
      prev = {};
    }
    if ( di.is_end() ) break;
 }
 return !cn.empty();
}

int de_ptx::hack(diter &di, res_map &rm, int in_bss) {
  used_regs<uint64_t> regs;
  while(1) {
    if ( !di.next() ) break;
    di.dasm();
    // luckily there is only limited set of instructions
    if ( di.ud_obj.mnemonic == UD_Ipush || di.ud_obj.mnemonic == UD_Ipop )
      continue;
    if ( di.is_movrr() && (di.ud_obj.operand[0].base == UD_R_RBP) ) continue;
    // lea
    if ( di.is_lea() && di.is_r1() ) {
      auto res = di.get_addr(1);
      if ( !in_sec(s_rodata, res) ) { report(di, "not in rodata"); return 0; }
      regs.add(di.ud_obj.operand[0].base, res);
      continue;
    }
    if ( !in_bss && di.is_lea() ) continue;
    if ( !in_bss && di.ud_obj.mnemonic == UD_Imov && di.ud_obj.operand[0].type == UD_OP_MEM ) {
      auto res = di.ud_obj.operand[0].lval.uqword;
      if ( di.ud_obj.operand[1].type == UD_OP_IMM ) {
        auto val = di.ud_obj.operand[1].lval.sdword;
        if ( in_sec(s_rodata, val) ) {
          lat_res what{ 0};
          if ( check(what, val) ) {
            rm[res] = what;
            continue;
          }
        }
        lat_res what{ 0, val };
        rm[res] = what;
        continue;
      }
      // mov [rip], reg
      if ( di.ud_obj.operand[1].type == UD_OP_REG ) {
        uint64_t val = 0;
        if ( !regs.asgn(di.ud_obj.operand[1].base, val) ) {
          report(di, "bad asgn");
          continue;
        }
        lat_res what{ 0};
        if ( check(what, val) ) rm[res] = what;
        continue;
      }
    }
    // mov [rip], imm
    if ( in_bss && di.is_mrip(0) && di.ud_obj.mnemonic == UD_Imov ) {
      auto res = di.get_addr(0);
      if ( !in_sec(s_bss, res) ) { report(di, "not in bss"); return 0; }
      if ( di.ud_obj.operand[1].type == UD_OP_IMM ) {
        auto val = di.ud_obj.operand[1].lval.sdword;
        if ( in_sec(s_rodata, val) ) {
          lat_res what{ 0};
          if ( check(what, val) ) {
            rm[res] = what;
            continue;
          }
        }
        lat_res what{ 0, val };
        rm[res] = what;
        continue;
      }
      // mov [rip], reg
      if ( di.ud_obj.operand[1].type == UD_OP_REG ) {
        uint64_t val = 0;
        if ( !regs.asgn(di.ud_obj.operand[1].base, val) ) {
          report(di, "bad asgn");
          return 0;
        }
        lat_res what{ 0};
        if ( check(what, val) ) rm[res] = what;
        continue;
      }
    }
    if ( di.is_end() ) return 1;
    if ( !in_bss && di.is_mov_rimm() ) continue;
    report(di, "unknown instr");
    if ( !in_bss ) return 1;
    break;
  }
  return 0;
}

int de_ptx::hack_ops(diter &di, uint64_t croot, res_map &rm) {
  used_regs<uint64_t> regs;
  while(1) {
    if ( !di.next() ) break;
    di.dasm();
    // lea
    if ( di.is_lea() && di.is_r1() ) {
      auto res = di.get_addr(1);
      if ( res <= croot || !in_sec(s_rodata, res) ) { regs.erase(di.ud_obj.operand[0].base); continue; }
      regs.add(di.ud_obj.operand[0].base, res);
      continue;
    }
    if ( di.is_end() || di.ud_obj.mnemonic == UD_Imovsq ) break;
    // mov [rbx + xx], imm/reg
    if ( di.is_mmem(UD_R_RBX) ) {
      auto off = di.ud_obj.operand[0].lval.udword;
      if ( !off ) continue;
      if ( di.ud_obj.operand[1].type == UD_OP_IMM ) {
        auto val = di.ud_obj.operand[1].lval.sdword;
        lat_res what{ 0, val };
        rm[off] = what;
        continue;
      } else if ( di.ud_obj.operand[1].type == UD_OP_REG ) {
        uint64_t r_val;
        if ( !regs.asgn(di.ud_obj.operand[1].base, r_val) ) continue;
        lat_res what{ 0};
        if ( check(what, r_val) ) {
          rm[off] = what;
          continue;
        }
      }
    }
  }
  return !rm.empty();
}

int de_ptx::hack_sp(diter &di, res_map &rm) {
  used_regs<uint64_t> regs;
  while(1) {
    if ( !di.next() ) break;
    di.dasm();
    // lea
    if ( di.is_lea() && di.is_r1() ) {
      auto res = di.get_addr(1);
      if ( !in_sec(s_rodata, res) ) { report(di, "not in rodata"); return 0; }
      regs.add(di.ud_obj.operand[0].base, res);
      continue;
    }
    // mov [rsp + off], imm
    if ( di.is_rsp() && di.ud_obj.mnemonic == UD_Imov ) {
      auto res = di.ud_obj.operand[0].lval.sdword;
      if ( di.ud_obj.operand[1].type == UD_OP_IMM ) {
        auto val = di.ud_obj.operand[1].lval.sdword;
        if ( in_sec(s_rodata, val) ) {
          lat_res what{ 0};
          if ( check(what, val) ) {
            rm[res] = what;
            continue;
          }
        }
        lat_res what{ 0, val };
        rm[res] = what;
        continue;
      }
      // mov [rsp + off], reg
      if ( di.ud_obj.operand[1].type == UD_OP_REG ) {
        uint64_t val = 0;
        if ( !regs.asgn(di.ud_obj.operand[1].base, val) ) {
          report(di, "bad asgn");
          return 0;
        }
        lat_res what{ 0};
        if ( check(what, val) ) rm[res] = what;
        continue;
      }
    }
    if ( di.is_end() ) return 1;
    report(di, "unknown instr");
//    printf("base %X\n", di.ud_obj.operand[0].base);
    if ( di.is_jxx(UD_Ijz) ) break;
  }
  return 0;
}

// check if operand 0 is OP_MEM sp based and offset fit in 0x40 - 0x50
static int in_sr(diter &di) {
  if ( di.ud_obj.operand[0].type != UD_OP_MEM ) return -1;
  if ( di.ud_obj.operand[0].base != UD_R_RSP )  return -1;
  auto off = di.ud_obj.operand[0].lval.sdword;
  if ( off < 0x40 ) return -1;
  if ( off >= 0x50 ) return -1;
  return off - 0x40;
}

// check if this is mov [rsp + 0x50], dword - for main big registration
static int is_add_mask(diter &di) {
  if ( di.ud_obj.operand[0].type != UD_OP_MEM ) return 0;
  if ( di.ud_obj.operand[0].base != UD_R_RSP )  return 0;
  auto off = di.ud_obj.operand[0].lval.sdword;
  if ( off < 0x50 ) return 0;
  if ( off >= 0x54 ) return 0;
  int res = 0x10 + off - 0x50;
//  printf("add_mask %X\n", res);
  return res;
}

// check if operand 0 is OP_MEM sp based and offset fit in 0x20 - 0x30 - for tables func
static int in_sr20(diter &di) {
  if ( di.ud_obj.operand[0].type != UD_OP_MEM ) return -1;
  if ( di.ud_obj.operand[0].base != UD_R_RSP )  return -1;
  auto off = di.ud_obj.operand[0].lval.sdword;
  if ( off < 0x20 ) return -1;
  if ( off >= 0x30 ) return -1;
  return off - 0x20;
}

// check if this is mov [rsp + 0x30], dword - for single registration
static int is_add_mask10(diter &di) {
  if ( di.ud_obj.operand[0].type != UD_OP_MEM ) return 0;
  if ( di.ud_obj.operand[0].base != UD_R_RSP )  return 0;
  auto off = di.ud_obj.operand[0].lval.sdword;
  if ( off < 0x30 ) return 0;
  if ( off >= 0x34 ) return 0;
  int res = 0x10 + off - 0x30;
//  printf("add_mask10 %X\n", res);
  return res;
}

template <typename G, typename T, typename A>
int de_ptx::cmn_ptx_op(diter &di, ptx_op &curr, G &regs, T t, A add) {
  // or reg, imm
  if ( di.is_or_rimm() ) {
   auto tgt = di.normalize_reg(di.ud_obj.operand[0].base, di.ud_obj.operand[0].size);
   regs.add(tgt, di.ud_obj.operand[1].lval.udword);
   return 1;
  }
  // mov reg, imm
  if ( di.is_mov_rimm(UD_R_R8D) ) {
    curr.idx = di.ud_obj.operand[1].lval.sdword;
    return 1;
  }
  // mov [mem], imm/reg
  if ( di.ud_obj.mnemonic == UD_Imov ) {
    int off = add(di);
    if ( off ) {
      if ( opt_d ) printf("add mask[%d], sizr %d, %X\n", off, di.ud_obj.operand[0].size,
       32 == di.ud_obj.operand[0].size ? di.ud_obj.operand[1].lval.udword : di.ud_obj.operand[1].lval.ubyte);
      // check zeroing
      if ( 32 == di.ud_obj.operand[0].size ) {
        curr.store_add_mask(di.ud_obj.operand[1].lval.udword);
        return 1;
      }
      curr.st[off] = di.ud_obj.operand[1].lval.ubyte;
      return 1;
    }
    off = t(di);
    if ( off >= 0 ) {
      if ( di.ud_obj.operand[1].type == UD_OP_REG ) {
        auto src = di.normalize_reg(di.ud_obj.operand[1].base, di.ud_obj.operand[1].size);
        uint32_t rval = 0;
        if ( regs.asgn(src, rval) ) curr.st[off] = rval & 0xff;
      } else
        curr.st[off] = di.ud_obj.operand[1].lval.ubyte;
      return 1;
    }
  }
  // or [mem], imm/reg
  if ( di.ud_obj.mnemonic == UD_Ior ) {
    int off = t(di);
    if ( off >= 0 ) {
      if ( di.ud_obj.operand[1].type == UD_OP_REG ) {
        auto src = di.normalize_reg(di.ud_obj.operand[1].base, di.ud_obj.operand[1].size);
        uint32_t rval = 0;
        if ( regs.asgn(src, rval) ) curr.st[off] |= rval & 0xff;
      } else
        curr.st[off] |= di.ud_obj.operand[1].lval.ubyte;
      return 1;
    }
  }
  return 0;
}

void de_ptx::gather_string(diter &di, de_ptx::ptx_op &curr) {
  auto saddr = di.get_addr(1);
  if ( !saddr ) return;
  auto s = sdata(s_rodata, saddr);
  if ( !s ) return;
  if ( di.ud_obj.operand[0].base == UD_R_RCX ) curr.cx = s;
  else if ( di.ud_obj.operand[0].base == UD_R_RDX ) curr.dx = s;
  else if ( di.ud_obj.operand[0].base == UD_R_RSI ) curr.si = s;
}

int de_ptx::process_one_ptx_op(diter &di, std::list<ptx_op> &res) {
  ptx_op curr;
  used_regs<uint32_t> regs;
  while(1) {
    if ( !di.next() ) return 0;
    di.dasm();
    if ( di.is_end() ) return 0;
    // check lea reg
    if ( di.is_lea() && di.is_r1() ) {
      gather_string(di, curr);
      continue;
    }
    if ( cmn_ptx_op(di, curr, regs, in_sr20, is_add_mask10) ) continue;
    // check call
    if ( di.is_call_jimm() ) {
      res.push_back(curr);
      return 1;
    }
  }
  return 0;
}

int de_ptx::hack_ptx_ops(uint64_t start, uint64_t end, uint64_t reg_call, uint64_t ops_tab, uint64_t ops_tab_end) {
  diter di(*s_text);
  std::list<ptx_op> res;
  ptx_op curr;
  if ( !di.setup(start) ) return 0;
  used_regs<uint32_t> regs;
  do {
    if ( !di.next() ) break;
    di.dasm();
    // check lea reg
    if ( di.is_lea() && di.is_r1() ) {
      gather_string(di, curr);
      continue;
    }
    if ( di.ud_obj.mnemonic == UD_Imovaps ) {
      if ( in_sr(di) >= 0 ) curr.re_st16();
      continue;
    }
    // check call
    if ( di.is_call_jimm() ) {
      auto saddr = di.get_addr(0);
      if ( saddr != reg_call ) continue;
      res.push_back(curr);
      regs.clear();
      continue;
    }
    if ( cmn_ptx_op(di, curr, regs, in_sr, is_add_mask) ) continue;
  } while( di.pc() < end );
  if ( res.empty() ) return 0;
  // read ops_tab
  if ( ops_tab && s_data.has_value() ) {
    for ( uint64_t ti = ops_tab; ti < ops_tab_end; ti += sizeof(uint64_t) ) {
       auto t_addr = read_ptr(s_data.value(), ti);
       if ( !t_addr ) continue;
       if ( !di.setup(t_addr) ) continue;
       process_one_ptx_op(di, res);
    }
  }
  res.sort([](ptx_op &a, ptx_op &b) { return a.idx < b.idx; });
  dump_ptx_ops(res);
  return 1;
}

void de_ptx::dump_ptx_ops(std::list<ptx_op> &lops) const {
  for ( auto &op: lops ) {
//    printf("%d %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X ", op.idx,
    printf("%d %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X ", op.idx,
     op.st[0], op.st[1], op.st[2], op.st[3], op.st[4], op.st[5], op.st[6], op.st[7],
     op.st[8], op.st[9], op.st[10], op.st[11], op.st[12], op.st[13], op.st[14], op.st[15]
     ,op.st[16], op.st[17], op.st[18], op.st[19]
    );
    printf("%s\t%s\t%s\n", op.dx, op.cx, op.si);
  }
}

static int dump_kw_types(std::vector<de_ptx::kw_type> &res) {
  if ( res.empty() ) return 0;
  std::sort(res.begin(), res.end(), [](const de_ptx::kw_type &a, const de_ptx::kw_type &b) { return a.first < b.first; });
  for ( auto &kw: res ) {
    printf("%X\t%s\n", kw.first, kw.second);
  }
  return 1;
}

int de_ptx::hack_intr(uint64_t start, uint64_t reg_call) {
  diter di(*s_text);
  std::vector<kw_type> res;
  kw_type curr;
  if ( !di.setup(start) ) return 0;
  while(1) {
    if ( !di.next() ) break;
    di.dasm();
    // lea rsi
    if ( di.is_lea() && di.is_r1() && di.ud_obj.operand[0].base == UD_R_RSI ) {
      auto saddr = di.get_addr(1);
      if ( !saddr ) continue;
      curr.second = sdata(s_rodata, saddr);
      continue;
    }
    // mov edx, imm
    if ( di.is_mov_rimm(UD_R_EDX) ) {
      curr.first = di.ud_obj.operand[1].lval.udword;
      continue;
    }
    // call/jmp
    if ( di.is_jxx_jimm(UD_Icall, UD_Ijmp) ) {
      auto ja = di.get_addr(0);
      if ( ja == reg_call && curr.second ) {
        res.push_back(curr);
        curr.first = 0;
        curr.second = nullptr;
      }
    }
    // we can have final jmp so checking for end must be last
    if ( di.is_end() ) break;
  }
  return dump_kw_types(res);
}

int de_ptx::hack_ptx_kws(uint64_t start) {
  diter di(*s_text);
  std::vector<kw_type> res;
  if ( !di.setup(start) ) return 0;
  used_regs<const char *> regs;
  while(1) {
    if ( !di.next() ) break;
    di.dasm();
    if ( di.is_end() ) break;
    // lea reg, rip
    if ( di.is_lea() && di.is_r1() ) {
      auto saddr = di.get_addr(1);
      if ( !saddr ) { regs.erase(di.ud_obj.operand[0].base); continue; }
      auto s = sdata(s_rodata, saddr);
      if ( !s ) { regs.erase(di.ud_obj.operand[0].base); continue; }
      regs.add(di.ud_obj.operand[0].base, s);
      continue;
    }
    if ( di.is_lea() ) {
       regs.erase(di.ud_obj.operand[0].base); continue;
    }
    // mov [mem], reg
    if ( di.ud_obj.mnemonic == UD_Imov && di.ud_obj.operand[1].type == UD_OP_REG &&
         di.ud_obj.operand[0].type == UD_OP_MEM && di.ud_obj.operand[0].base == UD_R_RBX ) {
      const char *st = nullptr;
      if ( regs.asgn(di.ud_obj.operand[1].base, st) ) res.push_back( { di.ud_obj.operand[0].lval.udword, st } );
    }
  }
  return dump_kw_types(res);
}

int de_ptx::hack_dumpers(diter &di, uint64_t reg_func, dump_map &res) {
  std::string name; // rsi
  uint64_t val = 0; // rdx
  while(1) {
    if ( !di.next() ) break;
    di.dasm();
    if ( di.is_lea() && di.is_r1() ) {
      if ( di.ud_obj.operand[0].base == UD_R_RSI ) {
        auto saddr = di.get_addr(1);
        if ( !saddr ) continue;
        name = sdata(s_rodata, saddr);
      } else if ( di.ud_obj.operand[0].base == UD_R_RDX ) {
        auto saddr = di.get_addr(1);
        if ( !saddr ) continue;
        if ( !in_sec(s_text, saddr) ) continue;
        val = saddr;
      }
      continue;
    }
    if ( di.is_jxx_jimm(UD_Icall, UD_Ijmp) ) {
      auto ja = di.get_addr(0);
      if ( ja == reg_func && !name.empty() && val ) {
        res[name] = { val, {} };
        name.clear();
        val = 0;
      }
    }
    // must be last
    if ( di.is_end() ) break;
  }
  return !res.empty();
}

void de_ptx::dump_dumpers(const dump_map &res) const {
  for( auto &pair: res ) {
    printf("%X %s\n", pair.second.first, pair.first.c_str());
    for ( auto off: pair.second.second ) printf("  %X\n", off);
  }
}

int de_ptx::collect(diter &di, one_dump &res) {
  // to keep block addresses ordered
  std::priority_queue<ptrdiff_t, std::vector<ptrdiff_t>, std::greater<ptrdiff_t> > addr_list;
  ITree covered;
  addr_list.push(res.first);
  std::unordered_set<uint64_t> added;
  ud_type base = UD_NONE;
  while( !addr_list.empty() ) {
    auto addr = addr_list.top();
    addr_list.pop();
    // check if we already processed it
    auto visited = covered.overlap_find( { addr, addr + 1 } );
    if ( visited != covered.end() ) continue;
    // setup
    if ( !di.setup(addr) ) continue;
    while(1) {
      if ( !di.next() ) break;
      di.dasm();
      // check if we have base
      if ( base == UD_NONE && di.is_movrr(UD_R_RSI) ) {
         base = di.ud_obj.operand[0].base;
         continue;
      }
      // collect jmps
      if ( di.is_jmp() && (di.ud_obj.operand[0].type == UD_OP_JIMM) ) {
        // add to queue
        auto jv = di.get_jmp(0);
        addr_list.push(jv);
      }
      if ( di.is_end() ) break;
      if ( base == UD_NONE ) continue;
      // collect all lea reg, [base + off]
      if ( di.is_lea() && di.ud_obj.operand[1].base == base ) {
        auto off = di.ud_obj.operand[1].lval.udword;
        auto vi = added.find(off);
        if ( vi == added.end() ) {
          res.second.push_back(off);
          added.insert(off);
        }
        continue;
      }
    }
    // add covered area
    if ( di.total )
      covered.insert_overlap( { addr, addr + di.total } );
  }
  return !res.second.empty();
}

void de_ptx::hack_dumpers(uint64_t start, uint64_t reg_func) {
  diter di(*s_text);
  if ( !di.setup(start) ) return;
  dump_map res;
  if ( hack_dumpers(di, reg_func, res) ) {
    for ( auto &rpair: res ) {
      if ( !di.setup(rpair.second.first) ) {
        fprintf(stderr, "cannot setup %X for %s\n", rpair.second.first, rpair.first.c_str());
        continue;
      }
      collect(di, rpair.second);
    }
    dump_dumpers(res);
  }
}

int de_ptx::extr_sw_str(diter &di, size_t idx, std::regex &re) {
  // states:
  // mov edi, imm - size
  // then lea rsi, [.rodata]
  int size = 0;
  while(1) {
    if ( !di.next() ) break;
    di.dasm();
    if ( !size ) {
      if ( di.is_mov_rimm(UD_R_EDI) ) size = di.ud_obj.operand[1].lval.sdword;
      continue;
    }
    if ( di.is_lea(UD_R_RSI) && di.is_r1() ) {
      auto saddr = di.get_addr(1);
      std::string_view res;
      if ( read_str(s_rodata.value(), saddr, res) ) {
        std::match_results<std::string_view::const_iterator> matches;
        printf("|%d %X", idx, size);
        if ( std::regex_search(res.begin(), res.end(), matches, re) ) {
          std::string_view mname{matches[1].first, matches[1].second};
          auto crc = adler32((const uint8_t*)mname.data(), mname.size());
          printf(" %X %.*s| %.*s\n", crc, mname.size(), mname.data(), res.size(), res.data());
        } else
          printf("| %.*s\n", res.size(), res.data());
        return 1;
      }
    }
  }
  return 0;
}

int de_ptx::extr_sw_strs(uint64_t start, size_t count) {
  diter di(*s_text);
  std::regex pattern(R"(\(.*\)\s+\b(\S+)\s*)");
  int32_t *tab = (int32_t *)sdata(s_rodata, start);
  if ( !tab ) return 0;
  int res = 0;
  for ( size_t idx = 0; idx < count; ++idx ) {
    auto addr = start + tab[idx];
    if ( !di.setup(addr) ) continue;
// printf("%d %p\n", idx, addr);
    res += extr_sw_str(di, idx, pattern);
  }
  return res;
}

int de_ptx::_read() {
  if ( !s_bss.has_value() || !s_text.has_value() || !s_rodata.has_value() ) return 0;
  // ptxas V13.1.80 md5 f38e5732c94163b96cf797eef252b4cb
  // string from get_pseudo_prototype switch case, size 1078
//  extr_sw_strs(0x1E66DB8, 1078);
//  hack_ptx_ops(0xC2341C, 0xC3C014, 0xC210C0, 0x2971260 + 8, 0x2971AD0);
 // yet another couple of latency tables, totally identical to c17.txt
  // hack_ops(0x7357B7, 0x1E01480, "ops1.txt");
  // hack_ops(0xEE17A7, 0x1E01480, "ops2.txt");
 // extract dumpers
//  hack_dumpers(0xE32420, 0x1E4A20);
//  hack_ptx_kws(0x391FA0);
 // extract ptx_intr.txt
//  hack_intr(0xE2F4A5, 0x1E4A20);
  // cicc 13.1 - md5 f3638b32a8740eda5e8cd5e5fe9decfb
  // hack_cicc_intr(0xA8BD00, "intr.txt");
  // for 12.8 md5 14dc7bbb0bafae1313489c389e9486eb - NPDOHYX
//  hack_ctor(0x582500, "c15.txt");
//  hack_ctor(0x598620, "c17.txt");
    hack_ctor(0xEE17A7, "c17_2.txt", 0);
    hack_ctor(0x7357B7, "c17_l.txt", 0);
//  hack_ctor(0x59D4A0, "c18.txt");
//  hack_ctor(0x41A8E0, "c5.txt");
//  hack_sp(0x11BCB51, "c1.txt");
//  hack_sp(0xA6C239, "c2.txt");
  // offsets from V13.1.80
  // md5: f38e5732c94163b96cf797eef252b4cb
//  hack_ctor(0x1BAC80, "c4.txt");
//  hack_ctor(0x1D0D70, "c5.txt");
//  hack_ctor(0x1D3250, "c8.txt");
  return 1;
}