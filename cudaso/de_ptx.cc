#include "de_ptx.h"
#include "x64arch.h"

extern int opt_d;

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

void de_ptx::hack_ctor(uint64_t off, const char *fname) {
  diter di(*s_text);
  if ( !di.setup(off) ) return;
  res_map res;
  if ( hack(di, res) ) dump_deres(fname, res);
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

int de_ptx::hack(diter &di, res_map &rm) {
  used_regs<uint64_t> regs;
  while(1) {
    if ( !di.next() ) break;
    di.dasm();
    // luckily there is only limited set of instructions
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
        lat_res what{ 0, di.ud_obj.operand[1].lval.sdword };
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

int de_ptx::_read() {
  if ( !s_bss.has_value() || !s_text.has_value() || !s_rodata.has_value() ) return 0;
  // offsets from V13.1.80
  // md5: f38e5732c94163b96cf797eef252b4cb
  hack_ctor(0x1BAC80, "c4.txt");
  hack_ctor(0x1D0D70, "c5.txt");
  hack_ctor(0x1D3250, "c8.txt");
  return 1;
}