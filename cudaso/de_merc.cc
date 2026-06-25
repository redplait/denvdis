#include "de_merc.h"
#include "x64arch.h"

template <>
void de_merc::dump(const dec_map &res) {
  auto first = res.cbegin();
  for ( auto fi = first; fi != res.cend(); ++fi ) {
    printf("%lX: ", fi->first - first->first);
    if ( fi->second.what == 1 )
      printf("-\n");
    else if ( fi->second.what == 2 )
      printf("%s\n", fi->second.str.c_str());
    else
      printf("%d\n", fi->second.num);
  }
}

template <>
void de_merc::dump(const opt_map &res) {
  auto first = res.cbegin();
  for ( auto fi = first; fi != res.cend(); ++fi ) {
    printf("%lX: ", fi->first - first->first);
    if ( fi->second.what == 1 )
      printf("-\n");
    else if ( fi->second.what == 2 )
      printf("%.*s\n", fi->second.str.size(), fi->second.str.data());
    else
      printf("%d\n", fi->second.num);
  }
}

template <>
int de_merc::check(lat_res<std::string_view> &r, uint64_t off) {
  if ( !in_sec(s_rodata, off) ) return 0;
  r.what = 2;
  read_str(s_rodata.value(), off, r.str);
  return 1;
}

template <>
int de_merc::check(lat_res<std::string> &r, uint64_t off) {
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
    r.str.push_back(c);
    ++s;
  }
  return 1;
}

extern void report(diter &di, const char *pfx);

template <typename T>
int de_merc::_hack(diter &di, std::map<uint64_t, T> &mres) {
  while(1) {
    if ( !di.next() ) break;
    di.dasm();
    // mov [rip], imm
    if ( di.is_mrip(0) && di.ud_obj.mnemonic == UD_Imov ) {
      auto res = di.get_addr(0);
      if ( !in_sec(s_bss, res) ) { report(di, "not in bss"); return 0; }
      if ( di.ud_obj.operand[1].type == UD_OP_IMM ) {
        auto val = di.ud_obj.operand[1].lval.sdword;
        if ( in_sec(s_rodata, val) ) {
          T what{ 0};
          if ( check(what, val) ) {
            mres[res] = what;
            continue;
          }
        }
        if ( in_sec(s_bss, val) ) continue;
        T what{ 0, val };
        mres[res] = what;
        continue;
      }
    }
    if ( di.is_end() ) break;
  }
  return !mres.empty();
}

template <typename T>
int de_merc::hack(uint64_t addr) {
  T res;
  diter di(*s_text);
  if ( !di.setup(addr) ) return 0;
  _hack(di, res);
  if ( res.empty() ) return 0;
  dump(res);
  return 1;
}

int de_merc::_read() {
  int res = 0;
  if ( !s_bss.has_value() || !s_text.has_value() || !s_rodata.has_value() ) return 0;
  // ctor 3 - plain strings
  res += hack<opt_map>(0x402470);
  // ctor 4 - encrypted strings
  printf("-- knobs\n");
  res += hack<dec_map>(0x417860);
  return res;
}