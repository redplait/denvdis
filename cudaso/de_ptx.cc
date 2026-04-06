#include "de_ptx.h"
#include "x64arch.h"

extern int opt_d;

int de_ptx::dump_cicc(const char *fname, const cicc_names &cn) {
  FILE *fp = fopen(fname, "w");
  if ( !fp ) {
    fprintf(stderr, "cannot create %s, error %d (%s)\n", fname, errno, strerror(errno));
    return 0;
  }
  for ( const auto ci: cn ) {
    fprintf(fp,"%d %s\n", ci.second, ci.first.c_str());
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

void de_ptx::hack_cicc_intr(uint64_t off, const char *fname) {
  diter di(*s_text);
  if ( !di.setup(off) ) return;
  cicc_names res;
  if ( hack_cicc(di, res) ) dump_cicc(fname, res);
}

void de_ptx::hack_ctor(uint64_t off, const char *fname) {
  diter di(*s_text);
  if ( !di.setup(off) ) return;
  res_map res;
  if ( hack(di, res) ) dump_deres(fname, res);
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

static void report(diter &di, const char *pfx) {
  printf("%s at %lX\n", pfx, ud_insn_off(&di.ud_obj));
}

int de_ptx::hack_cicc(diter &di, cicc_names &cn) {
 // states:
 // 0 - initial
 // 1 - rsi got string
 // 2 - call
 // 3 - mov [rax+8], const, reset to 0
 std::string name, prev;
 int state = 0;
 while(1) {
    if ( !di.next() ) break;
    di.dasm(state);
    if ( di.ud_obj.mnemonic == UD_Icall ) { prev = name; state = 2; continue; }
    if ( di.is_lea() && di.is_r1() && di.ud_obj.operand[0].base == UD_R_RSI ) {
      auto res = di.get_addr(1);
      if ( !in_sec(s_rodata, res) ) continue;
      name.clear();
      if ( read_str(s_rodata.value(), res, name) ) state = 1;
    } else if ( di.ud_obj.mnemonic == UD_Imov && di.ud_obj.operand[1].type == UD_OP_IMM &&
             di.ud_obj.operand[0].type == UD_OP_MEM && di.ud_obj.operand[0].lval.sdword == 8 ) {
      cn[prev] = di.ud_obj.operand[1].lval.sdword;
      prev.clear();
    }
    if ( di.is_end() ) break;
 }
 return !cn.empty();
}

int de_ptx::hack(diter &di, res_map &rm) {
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
    // mov [rip], imm
    if ( di.is_mrip(0) && di.ud_obj.mnemonic == UD_Imov ) {
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
    report(di, "unknown instr");
    break;
  }
  return 0;
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

int de_ptx::_read() {
  if ( !s_bss.has_value() || !s_text.has_value() || !s_rodata.has_value() ) return 0;
  // cicc 13.1 - md5 f3638b32a8740eda5e8cd5e5fe9decfb
  hack_cicc_intr(0xA8BD00, "intr.txt");
  // for 12.8 md5 14dc7bbb0bafae1313489c389e9486eb - NPDOHYX
//  hack_ctor(0x582500, "c15.txt");
//  hack_ctor(0x598620, "c17.txt");
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