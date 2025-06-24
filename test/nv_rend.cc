#include <dlfcn.h>
#include <fp16.h>
#include "nv_rend.h"

extern int opt_m;

// ripped from sm_version.txt
std::map<int, std::pair<const char *, const char *> > NV_renderer::s_sms = {
 { 0x1E, { "sm30", "sm3" } },
 { 0x20, { "sm32", "sm4" } },
 { 0x23, { "sm35", "sm4" } },
 { 0x25, { "sm37", "sm4" } },
 { 0x32, { "sm50", "sm5" } },
 { 0x34, { "sm52", nullptr } },
 { 0x35, { "sm53", "sm52" } },
 { 0x3c, { "sm60", "sm55" } },
 { 0x3d, { "sm61", "sm57" } },
 { 0x3e, { "sm62", "sm57" } },
 { 0x46, { "sm70", nullptr } },
 { 0x48, { "sm72", nullptr } },
 { 0x4b, { "sm75", nullptr } },
 { 0x50, { "sm80", nullptr } },
 { 0x56, { "sm86", nullptr } },
 { 0x57, { "sm87", "sm86" } },
 { 0x59, { "sm89", nullptr } },
 { 0x5a, { "sm90", nullptr } },
 { 0x64, { "sm100", nullptr } },
 { 0x65, { "sm101", nullptr } },
 { 0x78, { "sm120", nullptr } },
};

const char *NV_renderer::s_ltypes[] = {
 "" /* 0 */,
 "WARP_WIDE_INSTR",
 "COOP_GROUP_INSTR",
 "EXIT_INSTR",
 "S2RCTAID_INSTR",
};

const char *NV_renderer::s_labels[] = {
 "BRANCH_TARGET",
 "LABEL",
 "32LO",
 "32HI",
 "INDIRECT_CALL",
};

const char *NV_renderer::s_fmts[] = {
 "BITSET",
 "UImm",
 "SImm",
 "SSImm",
 "RSImm",
 "f64",
 "f16",
 "f32"
};

// MERCURY relocs
const char *mrelocs[] = {
/*  0 */ "R_MERCURY_NONE",
/*  1 */ "R_MERCURY_G64",
/*  2 */ "R_MERCURY_ABS64",
/*  3 */ "R_MERCURY_ABS32",
/*  4 */ "R_MERCURY_ABS16",
/*  5 */ "R_MERCURY_ABS32_LO",
/*  6 */ "R_MERCURY_ABS32_HI",
/*  7 */ "R_MERCURY_PROG_REL64",
/*  8 */ "R_MERCURY_PROG_REL32",
/*  9 */ "R_MERCURY_PROG_REL32_LO",
/*  a */ "R_MERCURY_PROG_REL32_HI",
/*  b */ "R_MERCURY_TEX_HEADER_INDEX",
/*  c */ "R_MERCURY_SAMP_HEADER_INDEX",
/*  d */ "R_MERCURY_SURF_HEADER_INDEX",
/*  e */ "R_MERCURY_UNUSED_CLEAR64",
/*  f */ "R_MERCURY_FUNC_DESC_64",
/* 10 */ "R_MERCURY_8_0",
/* 11 */ "R_MERCURY_8_8",
/* 12 */ "R_MERCURY_8_16",
/* 13 */ "R_MERCURY_8_24",
/* 14 */ "R_MERCURY_8_32",
/* 15 */ "R_MERCURY_8_40",
/* 16 */ "R_MERCURY_8_48",
/* 17 */ "R_MERCURY_8_56",
/* 18 */ "R_MERCURY_G8_0",
/* 19 */ "R_MERCURY_G8_8",
/* 1a */ "R_MERCURY_G8_16",
/* 1b */ "R_MERCURY_G8_24",
/* 1c */ "R_MERCURY_G8_32",
/* 1d */ "R_MERCURY_G8_40",
/* 1e */ "R_MERCURY_G8_48",
/* 1f */ "R_MERCURY_G8_56",
/* 20 */ "R_MERCURY_FUNC_DESC_8_0",
/* 21 */ "R_MERCURY_FUNC_DESC_8_8",
/* 22 */ "R_MERCURY_FUNC_DESC_8_16",
/* 23 */ "R_MERCURY_FUNC_DESC_8_24",
/* 24 */ "R_MERCURY_FUNC_DESC_8_32",
/* 25 */ "R_MERCURY_FUNC_DESC_8_40",
/* 26 */ "R_MERCURY_FUNC_DESC_8_48",
/* 27 */ "R_MERCURY_FUNC_DESC_8_56",
/* 28 */ "R_MERCURY_ABS_PROG_REL32_LO",
/* 29 */ "R_MERCURY_ABS_PROG_REL32_HI",
/* 2a */ "R_MERCURY_PROG_REL8_0",
/* 2b */ "R_MERCURY_PROG_REL8_8",
/* 2c */ "R_MERCURY_PROG_REL8_16",
/* 2d */ "R_MERCURY_PROG_REL8_24",
/* 2e */ "R_MERCURY_PROG_REL8_32",
/* 2f */ "R_MERCURY_PROG_REL8_40",
/* 30 */ "R_MERCURY_PROG_REL8_48",
/* 31 */ "R_MERCURY_PROG_REL8_56",
/* 32 */ "R_MERCURY_UNIFIED",
/* 33 */ "R_MERCURY_UNIFIED_32",
/* 34 */ "R_MERCURY_UNIFIED_8_0",
/* 35 */ "R_MERCURY_UNIFIED_8_8",
/* 36 */ "R_MERCURY_UNIFIED_8_16",
/* 37 */ "R_MERCURY_UNIFIED_8_24",
/* 38 */ "R_MERCURY_UNIFIED_8_32",
/* 39 */ "R_MERCURY_UNIFIED_8_40",
/* 3a */ "R_MERCURY_UNIFIED_8_48",
/* 3b */ "R_MERCURY_UNIFIED_8_56",
/* 3c */ "R_MERCURY_ABS_PROG_REL32",
/* 3d */ "R_MERCURY_ABS_PROG_REL64",
/* 3e */ "R_MERCURY_UNIFIED32_LO",
/* 3e */ "R_MERCURY_UNIFIED32_HI",
/* 40 */ "R_MERCURY_NONE_LAST",
};

const char* get_merc_reloc_name(unsigned t)
{
  if ( t < 0x10000 ) return nullptr;
  t -= 0x10000;
  if ( t >= sizeof(mrelocs) / sizeof(mrelocs[0]) ) return nullptr;
  return mrelocs[t];
}

// CUDA relocs
const char *cuda_relocs[] = {
/*  0 */ "R_CUDA_NONE",
/*  1 */ "R_CUDA_32",
/*  2 */ "R_CUDA_64",
/*  3 */ "R_CUDA_G32",
/*  4 */ "R_CUDA_G64",
/*  5 */ "R_CUDA_ABS32_26",
/*  6 */ "R_CUDA_TEX_HEADER_INDEX",
/*  7 */ "R_CUDA_SAMP_HEADER_INDEX",
/*  8 */ "R_CUDA_SURF_HW_DESC",
/*  9 */ "R_CUDA_SURF_HW_SW_DESC",
/*  A */ "R_CUDA_ABS32_LO_26",
/*  B */ "R_CUDA_ABS32_HI_26",
/*  C */ "R_CUDA_ABS32_23",
/*  D */ "R_CUDA_ABS32_LO_23",
/*  E */ "R_CUDA_ABS32_HI_23",
/*  F */ "R_CUDA_ABS24_26",
/* 10 */ "R_CUDA_ABS24_23",
/* 11 */ "R_CUDA_ABS16_26",
/* 12 */ "R_CUDA_ABS16_23",
/* 13 */ "R_CUDA_TEX_SLOT",
/* 14 */ "R_CUDA_SAMP_SLOT",
/* 15 */ "R_CUDA_SURF_SLOT",
/* 16 */ "R_CUDA_TEX_BINDLESSOFF13_32",
/* 17 */ "R_CUDA_TEX_BINDLESSOFF13_47",
/* 18 */ "R_CUDA_CONST_FIELD19_28",
/* 19 */ "R_CUDA_CONST_FIELD19_23",
/* 1A */ "R_CUDA_TEX_SLOT9_49",
/* 1B */ "R_CUDA_6_31",
/* 1C */ "R_CUDA_2_47",
/* 1D */ "R_CUDA_TEX_BINDLESSOFF13_41",
/* 1E */ "R_CUDA_TEX_BINDLESSOFF13_45",
/* 1F */ "R_CUDA_FUNC_DESC32_23",
/* 20 */ "R_CUDA_FUNC_DESC32_LO_23",
/* 21 */ "R_CUDA_FUNC_DESC32_HI_23",
/* 22 */ "R_CUDA_FUNC_DESC_32",
/* 23 */ "R_CUDA_FUNC_DESC_64",
/* 24 */ "R_CUDA_CONST_FIELD21_26",
/* 25 */ "R_CUDA_QUERY_DESC21_37",
/* 26 */ "R_CUDA_CONST_FIELD19_26",
/* 27 */ "R_CUDA_CONST_FIELD21_23",
/* 28 */ "R_CUDA_PCREL_IMM24_26",
/* 29 */ "R_CUDA_PCREL_IMM24_23",
/* 2A */ "R_CUDA_ABS32_20",
/* 2B */ "R_CUDA_ABS32_LO_20",
/* 2C */ "R_CUDA_ABS32_HI_20",
/* 2D */ "R_CUDA_ABS24_20",
/* 2E */ "R_CUDA_ABS16_20",
/* 2F */ "R_CUDA_FUNC_DESC32_20",
/* 30 */ "R_CUDA_FUNC_DESC32_LO_20",
/* 31 */ "R_CUDA_FUNC_DESC32_HI_20",
/* 32 */ "R_CUDA_CONST_FIELD19_20",
/* 33 */ "R_CUDA_BINDLESSOFF13_36",
/* 34 */ "R_CUDA_SURF_HEADER_INDEX",
/* 35 */ "R_CUDA_INSTRUCTION64",
/* 36 */ "R_CUDA_CONST_FIELD21_20",
/* 37 */ "R_CUDA_ABS32_32",
/* 38 */ "R_CUDA_ABS32_LO_32",
/* 39 */ "R_CUDA_ABS32_HI_32",
/* 3A */ "R_CUDA_ABS47_34",
/* 3B */ "R_CUDA_ABS16_32",
/* 3C */ "R_CUDA_ABS24_32",
/* 3D */ "R_CUDA_FUNC_DESC32_32",
/* 3E */ "R_CUDA_FUNC_DESC32_LO_32",
/* 3F */ "R_CUDA_FUNC_DESC32_HI_32",
/* 40 */ "R_CUDA_CONST_FIELD19_40",
/* 41 */ "R_CUDA_BINDLESSOFF14_40",
/* 42 */ "R_CUDA_CONST_FIELD21_38",
/* 43 */ "R_CUDA_INSTRUCTION128",
/* 44 */ "R_CUDA_YIELD_OPCODE9_0",
/* 45 */ "R_CUDA_YIELD_CLEAR_PRED4_87",
/* 46 */ "R_CUDA_32_LO",
/* 47 */ "R_CUDA_32_HI",
/* 48 */ "R_CUDA_UNUSED_CLEAR32",
/* 49 */ "R_CUDA_UNUSED_CLEAR64",
/* 4A */ "R_CUDA_ABS24_40",
/* 4B */ "R_CUDA_ABS55_16_34",
/* 4C */ "R_CUDA_8_0",
/* 4D */ "R_CUDA_8_8",
/* 4E */ "R_CUDA_8_16",
/* 4F */ "R_CUDA_8_24",
/* 50 */ "R_CUDA_8_32",
/* 51 */ "R_CUDA_8_40",
/* 52 */ "R_CUDA_8_48",
/* 53 */ "R_CUDA_8_56",
/* 54 */ "R_CUDA_G8_0",
/* 55 */ "R_CUDA_G8_8",
/* 56 */ "R_CUDA_G8_16",
/* 57 */ "R_CUDA_G8_24",
/* 58 */ "R_CUDA_G8_32",
/* 59 */ "R_CUDA_G8_40",
/* 5A */ "R_CUDA_G8_48",
/* 5B */ "R_CUDA_G8_56",
/* 5C */ "R_CUDA_FUNC_DESC_8_0",
/* 5D */ "R_CUDA_FUNC_DESC_8_8",
/* 5E */ "R_CUDA_FUNC_DESC_8_16",
/* 5F */ "R_CUDA_FUNC_DESC_8_24",
/* 60 */ "R_CUDA_FUNC_DESC_8_32",
/* 61 */ "R_CUDA_FUNC_DESC_8_40",
/* 62 */ "R_CUDA_FUNC_DESC_8_48",
/* 63 */ "R_CUDA_FUNC_DESC_8_56",
/* 64 */ "R_CUDA_ABS20_44",
/* 65 */ "R_CUDA_SAMP_HEADER_INDEX_0",
/* 66 */ "R_CUDA_UNIFIED",
/* 67 */ "R_CUDA_UNIFIED_32",
/* 68 */ "R_CUDA_UNIFIED_8_0",
/* 69 */ "R_CUDA_UNIFIED_8_8",
/* 6A */ "R_CUDA_UNIFIED_8_16",
/* 6B */ "R_CUDA_UNIFIED_8_24",
/* 6C */ "R_CUDA_UNIFIED_8_32",
/* 6D */ "R_CUDA_UNIFIED_8_40",
/* 6E */ "R_CUDA_UNIFIED_8_48",
/* 6F */ "R_CUDA_UNIFIED_8_56",
/* 70 */ "R_CUDA_UNIFIED32_LO_32",
/* 71 */ "R_CUDA_UNIFIED32_HI_32",
/* 72 */ "R_CUDA_ABS56_16_34",
/* 73 */ "R_CUDA_CONST_FIELD22_37",
};

const char* get_cuda_reloc_name(unsigned t)
{
  if ( t >= sizeof(cuda_relocs) / sizeof(cuda_relocs[0]) ) return nullptr;
  return cuda_relocs[t];
}

// ripped from https://forums.developer.nvidia.com/t/int-as-float/1557/10
float int_as_float(int a)
{
 union {int a; float b;} u;
 u.a = a;
 return u.b;
}

double longlong_as_double(long long a)
{
 union {long long a; double b;} u;
 u.a = a;
 return u.b;
}

// ripped from math_constants.h
const float NVf_inf = int_as_float(0x7f800000),
 NVf_nan = int_as_float(0x7fffffff);
;

const double NVd_inf = longlong_as_double(0x7ff0000000000000ULL),
 NVd_nan = longlong_as_double(0xfff8000000000000ULL)
;

template <typename C>
void NV_renderer::render_rel(std::string &res, const NV_rel *nr, const C &c) const
{
  switch(nr->first) {
    case 5:    // R_CUDA_ABS32_26
    case 0xc:  // R_CUDA_ABS32_23
    case 0xf:  // R_CUDA_ABS24_26
    case 0x2a: // R_CUDA_ABS32_20
    case 0x2d: // R_CUDA_ABS24_20
    case 0x37: // R_CUDA_ABS32_32
    case 0x3a: // R_CUDA_ABS47_34
    case 0x3c: // R_CUDA_ABS24_32
    case 0x4a: // R_CUDA_ABS24_40
    case 0x64: // R_CUDA_ABS20_44
     res += '`';
     break;
    case 0xa:  // R_CUDA_ABS32_LO_26
    case 0xd:  // R_CUDA_ABS32_LO_23
    case 0x2b: // R_CUDA_ABS32_LO_20
    case 0x38: // R_CUDA_ABS32_LO_32
    case 0x46: // R_CUDA_32_LO
     res += "32lo";
     break;
    case 0xb:  // R_CUDA_ABS32_HI_26
    case 0xe:  // R_CUDA_ABS32_HI_23
    case 0x2c: // R_CUDA_ABS32_LO_20
    case 0x39: // R_CUDA_ABS32_HI_32
    case 0x47: // R_CUDA_32_HI
     res += "32@hi";
     break;
    default:
     res += "UknRel";
     res += std::to_string(nr->first);
  }
  res += '(';
  res += c;
  res += ')';
}

void NV_renderer::dis_stat() const
{
  if ( dis_total )
    fprintf(m_out, "total %ld, not_found %ld, dups %ld\n", dis_total, dis_notfound, dis_dups);
}

int NV_renderer::load(const char *sm_name)
{
     void *dh = dlopen(sm_name, RTLD_NOW);
     if ( !dh ) {
      fprintf(stderr, "cannot load %s, errno %d (%s)\n", sm_name, errno, strerror(errno));
       return 0;
     }
     m_vq = (Dvq_name)dlsym(dh, "get_vq_name");
     if ( !m_vq )
       fprintf(stderr, "cannot find get_vq_nam(%s), errno %d (%s)\n", sm_name, errno, strerror(errno));
     Dproto fn = (Dproto)dlsym(dh, "get_sm");
     if ( !fn ) {
      fprintf(stderr, "cannot find get_sm(%s), errno %d (%s)\n", sm_name, errno, strerror(errno));
      dlclose(dh);
       return 0;
     }
     m_dis = fn();
     if ( m_dis ) m_width = m_dis->width();
     return (m_dis != nullptr);
}

void NV_renderer::dump_sv(const std::string_view &sv) const
{
  std::for_each( sv.cbegin(), sv.cend(), [&](char c){ fputc(c, m_out); });
}

void NV_renderer::dump_out(const std::string_view &sv) const
{
  std::for_each( sv.cbegin(), sv.cend(), [&](char c){ fputc(c, stdout); });
}

void NV_renderer::dump_outln(const std::string_view &sv) const
{
  std::for_each( sv.cbegin(), sv.cend(), [&](char c){ fputc(c, stdout); });
  fputc('\n', stdout);
}

void NV_renderer::dump_out(const std::string_view &sv, FILE *fp) const
{
  std::for_each( sv.cbegin(), sv.cend(), [fp](char c){ fputc(c, fp); });
}

void NV_renderer::dump_outln(const std::string_view &sv, FILE *fp) const
{
  std::for_each( sv.cbegin(), sv.cend(), [&](char c){ fputc(c, fp); });
  fputc('\n', fp);
}

int NV_renderer::cmp(const std::string_view &sv, const char *s) const
{
  size_t i = 0;
  for ( auto c = sv.cbegin(); c != sv.cend(); ++c, ++i ) {
    if ( *c != s[i] ) return 0;
  }
  return 1;
}

bool NV_renderer::contain(const std::string_view &sv, char sym) const
{
  return sv.find(sym) != std::string::npos;
}

void NV_renderer::dump_value(const nv_vattr &a, uint64_t v, NV_Format kind, std::string &res) const
{
  char buf[128];
  uint32_t f32;
  switch(kind)
  {
    case NV_F64Imm:
      snprintf(buf, 127, "%f", *(double *)&v);
     break;
    case NV_F16Imm:
      f32 = fp16_ieee_to_fp32_bits((uint16_t)v);
      snprintf(buf, 127, "%f", *(float *)&f32);
     break;
    case NV_F32Imm:
      snprintf(buf, 127, "%f", *(float *)&v);
     break;
    default:
      if ( !v ) { res += '0'; return; }
      snprintf(buf, 127, "0x%lX", v);
  }
  buf[127] = 0;
  res += buf;
}

void NV_renderer::dump_value(const struct nv_instr *ins, const NV_extracted &kv, const std::string_view &var_name,
  std::string &res, const nv_vattr &a, uint64_t v) const
{
  NV_Format kind = a.kind;
  if ( ins->vf_conv ) {
    auto convi = find(*ins->vf_conv, var_name);
    if ( convi ) {
      auto vi = kv.find(convi->fmt_var);
// printf("ins %s line %d  value fmt_var %d\n", ins->name, ins->line, (int)vi->second);
      if ( vi != kv.end() && ((short)vi->second == convi->v1 || (short)vi->second == convi->v2) )
      {
// printf("ins %s line %d: change kind to %d bcs value fmt_var %d\n", ins->name, ins->line, convi->second.format, (int)vi->second);
        kind = (NV_Format)convi->format;
      }
    }
  }
  dump_value(a, v, kind, res);
}

// old MD has encoders like Mask = Enum
// so check in eas
const nv_eattr *NV_renderer::try_by_ename(const struct nv_instr *ins, const std::string_view &sv) const
{
  if ( contain(sv, '@') ) return nullptr;
  // check in values
  if ( find(ins->vas, sv) ) return nullptr;
  for ( auto &ei: ins->eas ) {
    if ( cmp(sv, ei.ea->ename) ) return ei.ea;
  }
  return nullptr;
}

int NV_renderer::calc_miss(const struct nv_instr *ins, const NV_extracted &kv, int rz) const
{
  int res = 0;
  for ( auto ki: kv ) {
    const nv_eattr *ea = find_ea(ins, ki.first);
    if ( !ea ) continue;
    if ( cmp(ki.first, "NonZeroRegister") && (int)ki.second == rz ) {
      res++; continue;
    }
    if ( cmp(ki.first, "NonZeroUniformRegister") && (int)ki.second == rz ) {
      res++; continue;
    }
    // check in enum
    auto ei = ea->em->find(ki.second);
    if ( ei == ea->em->end() ) res++;
  }
  return res;
}

int NV_renderer::calc_index(const NV_res &res, int rz) const
{
  std::vector<int> missed(res.size());
  for ( size_t i = 0; i < res.size(); ++i ) {
    missed[i] = calc_miss( res[i].first, res[i].second, rz);
  }
  int res_idx = -1;
  bool mult = false;
  for ( size_t i = 0; i < res.size(); ++i )
  {
    if ( !missed[i] ) {
      if ( res_idx != -1 ) { mult = true; continue; }
      res_idx = i;
    }
  }
  if ( !mult ) return res_idx;
  // try the same without alts
  mult = false; res_idx = -1;
  for ( size_t i = 0; i < res.size(); ++i )
  {
    if ( res[i].first->alt ) continue;
    if ( !missed[i] ) {
      if ( res_idx != -1 ) { mult = true; continue; }
      res_idx = i;
    }
  }
  if ( !mult ) return res_idx;
  // no, we still have duplicates - dump missed and return -1
  for ( size_t i = 0; i < res.size(); ++i ) fprintf(m_out, " %d", missed[i]);
  return -1;
}

int NV_renderer::check_abs(const NV_extracted &kv, const char* name) const
{
  std::string mod_name(name);
  mod_name += "@absolute";
  auto kvi = kv.find(mod_name);
  if ( kvi == kv.end() ) { if ( opt_m ) m_missed.insert(mod_name); return 0; }
  if ( !kvi->second ) return 0;
  return 1;
}

int NV_renderer::check_abs(const NV_extracted &kv, const char* name, std::string &r) const
{
  auto res = check_abs(kv, name);
  if ( res ) r += '|';
  return res;
}

int NV_renderer::check_mod(char c, const NV_extracted &kv, const char* name, std::string &r) const
{
  std::string mod_name(name);
  switch(c) {
    case '!': mod_name += "@not"; break;
    case '-': mod_name += "@negate"; break;
    case '~': mod_name += "@invert"; break;
    default: return 0;
  }
  auto kvi = kv.find(mod_name);
  if ( kvi == kv.end() ) { if ( opt_m ) m_missed.insert(mod_name); return 0; }
  if ( !kvi->second ) return 0;
  r += c;
  return 1;
}

// render left [] in C, CX, desc etc
int NV_renderer::render_ve(const ve_base &ve, const struct nv_instr *i, const NV_extracted &kv, std::string &res) const
{
  if ( ve.type == R_value )
  {
    auto kvi = kv.find(ve.arg);
    if ( kvi == kv.end() ) { if ( opt_m ) m_missed.insert(ve.arg); return 1; }
    auto vi = find(i->vas, ve.arg);
    if ( !vi ) return 1;
    dump_value(i, kv, ve.arg, res, *vi, kvi->second);
    return 0;
  }
  // enum
  const nv_eattr *ea = find_ea(i, ve.arg);
  if ( !ea ) return 1;
  auto kvi = kv.find(ve.arg);
  if ( kvi == kv.end() ) { if ( opt_m ) m_missed.insert(ve.arg); return 1; }
  auto eid = ea->em->find(kvi->second);
  if ( eid != ea->em->end() )
    res += eid->second;
  else return 1;
  return 0;
}

// render right []
int NV_renderer::render_ve_list(const std::list<ve_base> &l, const struct nv_instr *i, const NV_extracted &kv, std::string &res) const
{
  auto size = l.size();
  if ( 1 == size )
    return render_ve(*l.begin(), i, kv, res);
  int missed = 0;
  int idx = 0;
  for ( auto ve: l ) {
    if ( ve.type == R_value )
    {
      auto kvi = kv.find(ve.arg);
      if ( kvi == kv.end() ) { if ( opt_m ) m_missed.insert(ve.arg); missed++; idx++; continue; }
      auto vi = find(i->vas, ve.arg);
      if ( !vi ) { missed++; idx++; continue; }
      std::string tmp;
      dump_value(i, kv, ve.arg, tmp, *vi, kvi->second);
      if ( tmp == "0" && idx ) { idx++; continue; } // ignore +0
      if ( ve.pfx ) res += ve.pfx;
      else if ( idx ) res += '+';
      res += tmp;
      idx++;
      continue;
    }
    // this is (optional) enum
    const nv_eattr *ea = find_ea(i, ve.arg);
    if ( !ea ) {
      missed++;
      continue;
    }
    auto kvi = kv.find(ve.arg);
    if ( kvi == kv.end() ) {
      kvi = kv.find(ea->ename);
      if ( kvi == kv.end() ) {
        if ( opt_m ) m_missed.insert(ve.arg);
        missed++;
        continue;
      }
    }
    if ( ea->has_def_value && ea->def_value == (int)kvi->second ) {
      if ( ea->ignore && !ea->print ) continue;
      // ignore zero register even without ea->ignore
      if ( !strcmp(ea->ename, "ZeroRegister") ) continue;
    }
    if ( !ea->ignore ) idx++;
    if ( ea->ignore ) res += '.';
    else {
      if ( ve.pfx ) res += ve.pfx;
      else if ( idx > 1 ) res += " + ";
    }
    auto eid = ea->em->find(kvi->second);
    if ( eid != ea->em->end() )
       res += eid->second;
    else {
       missed++;
       continue;
    }
  }
  return missed;
}

bool NV_renderer::check_branch(const struct nv_instr *i, const NV_extracted::const_iterator &kvi, long &res) const
{
  if ( !i->brt || !i->target_index ) {
    // BSSY has type RSImm
    auto vi = find(i->vas, kvi->first);
    if ( !vi ) return false;
    if ( vi->kind != NV_RSImm ) return false;
  } else {
    if ( kvi->first != i->target_index ) return false;
  }
  // find width
//printf("try to find target_index %s value %lX\n", i->target_index, kvi->second);
  auto wi = find(i->vwidth, kvi->first);
  if ( !wi ) return false;
  // yes, this is some imm for branch, check if it negative
  if ( kvi->second & (1L << (wi->w - 1)) )
    res = kvi->second - (1L << wi->w);
  else
    res = (long)kvi->second;
  return true;
}

template <typename Fs, typename Fl>
int NV_renderer::rend_single(const render_base *r, std::string &res, const char *opcode, Fs &&r1, Fl &&rl) const
{
  switch(r->type) {
      case R_value:
      case R_predicate: {
        const render_named *rn = (const render_named *)r;
        if ( r->pfx ) res += r->pfx;
        res += rn->name;
       }
       break;
      case R_enum:{
        const render_named *rn = (const render_named *)r;
        if ( rn->mod ) {
          res += rn->mod;
          if ( rn->abs ) res += '|';
        }
        res += "E:";
        res += rn->name;
       }
       break;
      case R_opcode:
        if ( opcode )
          res += opcode;
        else
          res += "OPCODE";
       break;
      case R_C:
      case R_CX: {
         const render_C *rn = (const render_C *)r;
         if ( rn->mod ) {
          res += rn->mod;
          if ( rn->abs ) res += '|';
         }
         res += "c:";
         if ( rn->name ) res += rn->name;
         res += "[";
         r1(rn->left, res);
         res += "][";
         rl(rn->right, res);
         res += ']';
       } break;
       case R_TTU: {
         const render_TTU *rt = (const render_TTU *)r;
         if ( rt->pfx ) res += rt->pfx;
         else res += ' ';
         res += "ttu:[";
         r1(rt->left, res);
         res += ']';
       }
       break;
     case R_M1: {
         const render_M1 *rt = (const render_M1 *)r;
         if ( rt->pfx ) res += rt->pfx;
         if ( rt->name ) res += rt->name;
         res += ":[";
         r1(rt->left, res);
         res += ']';
       } break;

      case R_desc: {
         const render_desc *rt = (const render_desc *)r;
         if ( rt->pfx ) res += rt->pfx;
         res += "desc:[";
         r1(rt->left, res);
         res += "],[";
         rl(rt->right, res);
         res += ']';
       } break;

      case R_mem: {
         const render_mem *rt = (const render_mem *)r;
         if ( rt->pfx ) res += rt->pfx;
         res += "[";
         rl(rt->right, res);
         res += ']';
       } break;

      default:
        if ( opcode ) fprintf(stderr, "unknown rend type %d for inst %s\n", r->type, opcode);
        return 0;
    }
 return !res.empty();
}

int NV_renderer::rend_single(const render_base *r, std::string &res, const char *opcode) const
{
  return rend_single(r, res, opcode, 
    std::bind(&NV_renderer::r_ve, this, std::placeholders::_1, std::placeholders::_2),
    std::bind(&NV_renderer::r_velist, this, std::placeholders::_1, std::placeholders::_2)
  );
}

int NV_renderer::rend_singleE(const struct nv_instr *instr, const render_base *r, std::string &res) const
{
  const nv_eattr *ea = nullptr;
  if ( r->type == R_enum || r->type == R_predicate ) {
    const render_named *rn = (const render_named *)r;
    ea = find_ea(instr, rn->name);
    if ( ea && ea->ignore ) res += '.';
  }
  int what = rend_single(r, res, instr ? instr->name: nullptr);
  if ( !what ) return 0;
  if ( ea && ea->has_def_value ) {
    res += ".D(";
    res += std::to_string(ea->def_value);
    res += ")";
  }
  if ( r->type == R_value && instr->vas ) {
    // try to find format in instr->vas
    const render_named *rn = (const render_named *)r;
    auto viter = find(instr->vas, rn->name);
    if ( viter ) {
      res += ':';
      res += s_fmts[viter->kind];
      if ( viter->dval ) {
        res += '(';
        res += std::to_string(viter->dval);
        res += ')';
      }
      if ( viter->has_ast ) res += '*';
    }
  }
  return what;
}

int NV_renderer::rend_rendererE(const struct nv_instr *instr, const NV_rlist *rlist, std::string &res) const
{
  for ( auto r: *rlist ) {
    if ( r->type == R_enum || r->type == R_predicate || r->type == R_value )
      rend_singleE(instr, r, res);
    else
      rend_single(r, res, instr->name, std::bind(&NV_renderer::r_vei, this, instr, std::placeholders::_1, std::placeholders::_2),
       std::bind(&NV_renderer::r_velisti, this, instr, std::placeholders::_1, std::placeholders::_2) );
    res += ' ';
  }
  res.pop_back(); // remove last space
  return !res.empty();
}

int NV_renderer::rend_renderer(const NV_rlist *rlist, const std::string &opcode, std::string &res) const
{
  for ( auto r: *rlist ) {
    rend_single(r, res, opcode.c_str());
    res += ' ';
  }
  res.pop_back(); // remove last space
  return !res.empty();
}

void NV_renderer::r_velist(const std::list<ve_base> &l, std::string &res) const
{
  auto size = l.size();
  if ( 1 == size ) {
    r_ve(*l.begin(), res);
    return;
  }
  int idx = 0;
  for ( auto ve: l ) {
    if ( ve.type == R_value )
    {
      if ( ve.pfx ) res += ve.pfx;
      else if ( idx ) res += '+';
      res += ve.arg;
      idx++;
      continue;
    }
    // enum
    res += "E:";
    res += ve.arg;
    res += " ";
  }
  if ( res.back() == ' ' ) res.pop_back();
}

void NV_renderer::r_velisti(const struct nv_instr *instr, const std::list<ve_base> &l, std::string &res) const
{
  auto size = l.size();
  if ( 1 == size ) {
    r_vei(instr, *l.begin(), res);
    return;
  }
  int idx = 0;
  for ( auto ve: l ) {
    if ( ve.type == R_value )
    {
      if ( ve.pfx ) res += ve.pfx;
      else if ( idx ) res += '+';
      res += ve.arg;
      auto viter = find(instr->vas, ve.arg);
      if ( viter ) {
        res += ':';
        res += s_fmts[viter->kind];
        if ( viter->has_ast ) res += '*';
      }
      idx++;
      continue;
    }
    // enum
    auto ea = find_ea(instr, ve.arg);
    if ( ea && ea->ignore ) res += '.';
    res += "E:";
    res += ve.arg;
    if ( ea && ea->has_def_value ) {
      res += ".D(";
      res += std::to_string(ea->def_value);
      res += ")";
    }
    res += " ";
  }
  if ( res.back() == ' ' ) res.pop_back();
}

void NV_renderer::r_vei(const struct nv_instr *instr, const ve_base &ve, std::string &res) const
{
  if ( ve.type == R_enum ) {
    auto ea = find_ea(instr, ve.arg);
    if ( ea && ea->ignore ) res += '.';
    res += "E:";
    res += ve.arg;
    if ( ea && ea->has_def_value ) {
      res += ".D(";
      res += std::to_string(ea->def_value);
      res += ")";
    }
  } else {
    res += ve.arg;
    auto viter = find(instr->vas, ve.arg);
    if ( viter ) {
      res += ':';
      res += s_fmts[viter->kind];
      if ( viter->has_ast ) res += '*';
    }
  }
}

void NV_renderer::r_ve(const ve_base &ve, std::string &res) const
{
  if ( ve.type == R_enum ) res += "E:";
  res += ve.arg;
}

int NV_renderer::render(const NV_rlist *rl, std::string &res, const struct nv_instr *i,
 const NV_extracted &kv, NV_labels *l, int opt_c) const
{
  int idx = 0;
  int missed = 0;
  int was_bs = 0; // seems that scheduling args always starts with BITSET req_xx
  int prev = -1;  // workaround to fix op, bcs testcc is missed
  for ( auto ri: *rl ) {
    std::string tmp;
    int is_abs = 0, empty = 0;
    switch(ri->type)
    {
      case R_opcode:
       res += i->name;
       break;

      case R_value: {
        char buf[128];
        const render_named *rn = (const render_named *)ri;
        auto kvi = kv.find(rn->name);
        if ( kvi == kv.end() ) {
          if ( opt_m ) m_missed.insert(rn->name);
          missed++;
          empty = 1;
          break;
        }
        auto vi = find(i->vas, rn->name);
        if ( !vi ) { missed++; empty = 1; break; }
        if ( vi->kind == NV_BITSET && !strncmp(rn->name, "req_", 4) ) was_bs = 1;
        if ( was_bs && opt_c ) {
          // unfortunatelly nvdisasm can dump only 4 fields at tail
          // req_bit_set: &req={bit mask}
          // src_rel_sb: &rd=0xnum
          // dst_wr_sb:  &wr=0xnum
          // usched_info: &enum_name
          // two last - batch_t & pm_pred - should be ignored
          // so lets check what we have
          if ( !strcmp(rn->name, "req_bit_set") ) {
            if ( kvi->second != vi->dval ) {
              tmp = " &req={";
              for ( int bi = 0; bi < 6; bi++ ) {
                if ( kvi->second & (1 << bi) ) {
                  tmp += std::to_string(bi);
                  tmp.push_back(',');
                }
              }
              tmp.pop_back();
              tmp.push_back('}');
            }
          } else if ( !strcmp(rn->name, "src_rel_sb") ) {
            if ( kvi->second != vi->dval ) {
             tmp = " &rd=0x";
             snprintf(buf, 127, "%lX", kvi->second);
             tmp += buf;
            }
          } else if ( !strcmp(rn->name, "dst_wr_sb") ) {
            if ( kvi->second != vi->dval ) {
             tmp = " &wr=0x";
             snprintf(buf, 127, "%lX", kvi->second);
             tmp += buf;
            }
          } else if ( !strcmp(rn->name, "usched_info") ) {
            dump_value(i, kv, rn->name, tmp, *vi, kvi->second);
            tmp += ' ';
            res += " ?";
          }
        } else {
          long branch_off = 0;
          if ( check_branch(i, kvi, branch_off) ) {
            // make (LABEL_xxx)
            if ( opt_c )
              snprintf(buf, 127, " `(LABEL_%lX)", branch_off + m_dis->off_next());
            else {
              snprintf(buf, 127, "%ld", branch_off);
              tmp = buf;
              snprintf(buf, 127, " (LABEL_%lX)", branch_off + m_dis->off_next());
            }
            if ( l ) (*l)[branch_off + m_dis->off_next()] = 0;
            tmp += buf;
          } else
            dump_value(i, kv, rn->name, tmp, *vi, kvi->second);
          if ( rn->pfx ) { if ( prev != R_opcode ) res += rn->pfx; }
          res += ' ';
          if ( !rn->pfx && was_bs ) res += '&';
        }
        res += tmp;
       } break;

      case R_enum: {
         const render_named *rn = (const render_named *)ri;
         const nv_eattr *ea = find_ea(i, rn->name);
         if ( !ea ) {
           missed++;
           idx++;
           continue;
         }
         if ( was_bs && opt_c ) continue;
         auto kvi = kv.find(rn->name);
         if ( kvi == kv.end() ) {
           kvi = kv.find(ea->ename);
           if ( kvi == kv.end() ) {
             if ( opt_m ) m_missed.insert(rn->name);
             missed++;
             idx++;
             continue;
           }
         }
         // now we have enum attr in ea and value in kvi
         // we have 2 cases - if this attr has ignore and !print and value == def_value - we should skip it
         if ( ea->has_def_value && ea->def_value == (int)kvi->second && ea->ignore && !ea->print ) {
           idx++; empty = 1; continue;
         }
         if ( ea->ignore ) { res += '.'; empty = 1; }
         else {
           if ( rn->pfx && prev != R_opcode ) {
             if ( '?' == rn->pfx ) res += ' ';
             res += rn->pfx;
           } else res += ' ';
           // check mod
           if ( rn->mod ) check_mod(rn->mod, kv, rn->name, res);
           if ( rn->abs ) is_abs = check_abs(kv, rn->name, res);
         }
         auto eid = ea->em->find(kvi->second);
         if ( eid != ea->em->end() )
           res += eid->second;
         else {
           missed++;
           break;
         }
         if ( is_abs ) res += '|';
         if ( ea->ignore ) {
           idx++; continue;
         }
       } break;

      case R_predicate: { // like enum but can be ignored if has default value
         const render_named *rn = (const render_named *)ri;
         const nv_eattr *ea = find_ea(i, rn->name);
         if ( !ea ) {
           missed++;
           break;
         }
         auto kvi = kv.find(rn->name);
         if ( kvi == kv.end() ) {
           if ( opt_m ) m_missed.insert(rn->name);
           missed++;
           break;
         }
         if ( ea->def_value == (int)kvi->second ) { empty = 1; break; }
         if ( rn->pfx ) res += rn->pfx;
         else if ( idx ) res += ' ';
         if ( rn->mod ) check_mod(rn->mod, kv, rn->name, res);
         auto eid = ea->em->find(kvi->second);
         if ( eid != ea->em->end() )
           res += eid->second;
         else {
           missed++;
           break;
         }
         if ( '@' == rn->pfx ) res += ' ';
       } break;

      case R_C:
      case R_CX: {
         const render_C *rn = (const render_C *)ri;
         if ( rn->pfx && prev != R_opcode ) res += rn->pfx;
         else res += ' ';
         if ( rn->mod ) check_mod(rn->mod, kv, rn->name, res);
         if ( rn->abs ) is_abs = check_abs(kv, rn->name, res);
         res += "c[";
         missed += render_ve(rn->left, i, kv, res);
         res += "][";
         missed += render_ve_list(rn->right, i, kv, res);
         res += ']';
         if ( is_abs ) res += '|';
       } break;

      case R_TTU: {
         const render_TTU *rt = (const render_TTU *)ri;
         if ( rt->pfx && prev != R_opcode ) res += rt->pfx;
         else res += ' ';
         res += "ttu[";
         missed += render_ve(rt->left, i, kv, res);
         res += ']';
       } break;

      case R_M1: {
         const render_M1 *rt = (const render_M1 *)ri;
         if ( rt->pfx && prev != R_opcode ) res += rt->pfx;
         else res += ' ';
         res += rt->name;
         res += "[";
         missed += render_ve(rt->left, i, kv, res);
         res += ']';
       } break;

      case R_desc: {
         const render_desc *rt = (const render_desc *)ri;
         if ( rt->pfx && prev != R_opcode ) res += rt->pfx;
         else res += ' ';
         res += "desc[";
         missed += render_ve(rt->left, i, kv, res);
         res += "][";
         missed += render_ve_list(rt->right, i, kv, res);
         res += ']';
       } break;

      case R_mem: {
         const render_mem *rt = (const render_mem *)ri;
         if ( rt->pfx && prev != R_opcode ) res += rt->pfx;
         else res += ' ';
         res += "[";
         missed += render_ve_list(rt->right, i, kv, res);
         res += ']';
       } break;

      default: fprintf(stderr, "unknown rend type %d at index %d for inst %s\n", ri->type, idx, i->name);
    }
    if ( !empty )
      prev = ri->type;
    idx++;
  }
  return missed;
}

int NV_renderer::dump_predicates(const struct nv_instr *i, const NV_extracted &kv, FILE *fp, const char *pfx) const
{
  if ( !i->predicated ) return 0;
  int ret = 0;
  for ( auto &pred: *i->predicated ) {
    ret++;
    if ( pfx ) fputs(pfx, fp);
    dump_out(pred.first, fp);
    int res = pred.second(kv);
    if ( res >= 0 && m_vq && cmp(pred.first, "VQ") ) {
     auto name = m_vq(res);
     if ( name )
       fprintf(fp, ": %s (%d)", name, res);
     else
       fprintf(fp, ": %d", res);
    } else
      fprintf(fp, ": %d", res);
    fputc('\n', fp);
  }
  return ret;
}

void NV_renderer::dump_predicates(const struct nv_instr *i, const NV_extracted &kv) const
{
  if ( !i->predicated ) return;
  dump_predicates(i, kv, m_out, "P> ");
}

void NV_renderer::dump_ops(const struct nv_instr *i, const NV_extracted &kv) const
{
  for ( auto kv1: kv )
  {
    std::string name(kv1.first.begin(), kv1.first.end());
    // check in values
    if ( i->vas ) {
      auto vi = find(*i->vas, kv1.first);
      if ( vi ) {
        std::string buf;
        dump_value(i, kv, kv1.first, buf, *vi, kv1.second);
        fprintf(m_out, " V %s: %s type %d\n", name.c_str(), buf.c_str(), vi->kind);
        continue;
      }
    }
    // check in enums
    const nv_eattr *ea = find_ea(i, kv1.first);
    if ( ea ) {
      fprintf(m_out, " E %s: %s %lX", name.c_str(), ea->ename, kv1.second);
      auto eid = ea->em->find(kv1.second);
      if ( eid != ea->em->end() )
        fprintf(m_out, " %s\n", eid->second);
      else
        fprintf(m_out," UNKNOWN_ENUM %lX\n", kv1.second);
      continue;
    }
    if ( name.find('@') != std::string::npos ) {
      fprintf(m_out, " @ %s: %lX\n", name.c_str(), kv1.second);
      continue;
    }
    fprintf(m_out, " U %s: %lX\n", name.c_str(), kv1.second);
  }
}

bool NV_renderer::check_sched_cond(const struct nv_instr *i, const NV_extracted &kv, const NV_one_cond &clist)
{
  return check_sched_cond(i, kv, clist, nullptr);
}

bool NV_renderer::check_sched_cond(const struct nv_instr *i, const NV_extracted &kv, const NV_one_cond &clist,
 NV_Tabset *out_res)
{
  if ( !clist.second || !clist.second->size() ) return true;
  int res = 0;
  for ( auto &cond: *clist.second ) {
    if ( cond.second ) {
      scond_count++;
      if ( !cond.second(i, kv) ) continue;
      scond_succ++;
    }
    auto kiter = kv.find(cond.first);
    if ( kiter == kv.end() ) { if ( opt_m ) m_missed.insert({cond.first.begin(), cond.first.end()}); continue; }
    if ( out_res )
     (*out_res)[cond.first] = (int)kiter->second;
    res++;
  }
  return res != 0;
}

int NV_renderer::fill_sched(const struct nv_instr *i, const NV_extracted &kv)
{
  m_sched.clear();
  m_cached_tabsets.clear();
  std::unordered_map< const NV_cond_list *, NV_Tabset *> cached;
  if ( !i->cols ) return 0;
  int res = 0;
  for ( auto &titer: *i->cols ) {
    if ( titer.filter ) {
      sfilters++;
      if ( !titer.filter(i, kv) ) continue;
      sfilters_succ++;
    }
    NV_Tabset *tset = nullptr;
    // check in cache
    auto cres = cached.find( get_it(titer.tab->cols, titer.idx).second );
    if ( cres != cached.end() ) {
      scond_hits++;
      if ( !cres->second ) continue;
      tset = cres->second;
    } else {
      // check tab.cols[titer.idx].second for condition
      NV_Tabset row_res;
      if ( !check_sched_cond(i, kv, get_it(titer.tab->cols, titer.idx), &row_res) ) {
        if ( get_it(titer.tab->cols, titer.idx).second )
          cached[ get_it(titer.tab->cols, titer.idx).second ] = nullptr; // store bad result in cache too
        continue;
      }
      // put row_res to m_cached_tabsets
      m_cached_tabsets.push_back( std::move(row_res) );
      tset = &m_cached_tabsets.back();
      if ( get_it(titer.tab->cols, titer.idx).second )
        cached[ get_it(titer.tab->cols, titer.idx).second ] = tset; // store res in cache
    }
    auto ct = m_sched.find(titer.tab);
    if ( ct == m_sched.end() )
      m_sched[titer.tab] = { { titer.idx, tset } };
    else
      ct->second.push_back( { titer.idx, tset });
    res++;
  }
  return res;
}

void NV_renderer::dump_cond_list(const NV_Tabset *cset) const
{
  if ( cset->empty() ) return;
  int latch = 0;
  fputc('{', m_out);
  for ( auto &i: *cset ) {
    if ( latch ) fputc(' ', m_out);
    latch |= 1;
    dump_sv(i.first);
    fprintf(m_out, ":%d", i.second);
  }
  fputc('}', m_out);
}

int NV_renderer::dump_sched(const struct nv_instr *i, const NV_extracted &kv)
{
  if ( !i->rows ) return 0;
  int res = 0;
  std::unordered_map< const NV_cond_list *, NV_Tabset *> cached;
  for ( auto &titer: *i->rows ) {
    auto ci = m_sched.find(titer.tab);
    if ( ci == m_sched.end() ) continue;
    if ( titer.filter ) {
      sfilters++;
      if ( !titer.filter(i, kv) ) continue;
      sfilters_succ++;
    }
    NV_Tabset *tset = nullptr;
    // check in cache
    auto cres = cached.find( get_it(titer.tab->rows, titer.idx).second );
    if ( cres != cached.end() ) {
      scond_hits++;
      if ( !cres->second ) continue;
      tset = cres->second;
    } else {
      NV_Tabset row_res;
      if ( !check_sched_cond(i, kv, get_it(titer.tab->rows, titer.idx), &row_res) ) {
        if ( get_it(titer.tab->rows, titer.idx).second )
          cached[ get_it(titer.tab->rows, titer.idx).second ] = nullptr; // store bad result in cache too
        continue;
      }
      // put row_res to m_cached_tabsets
      m_cached_tabsets.push_back( std::move(row_res) );
      tset = &m_cached_tabsets.back();
      if ( get_it(titer.tab->rows, titer.idx).second )
        cached[ get_it(titer.tab->rows, titer.idx).second ] = tset; // store res in cache
    }
    // we have titer.tab & titer.idx for row and
    // ci->list of table columns
    for ( auto cidx: ci->second ) {
      auto value = ci->first->get(cidx.first, titer.idx);
      if ( !value ) continue;
      fprintf(m_out, "S> tab %s %s row %d", ci->first->name, ci->first->connection, titer.idx);
      auto row_name = get_it(ci->first->rows, titer.idx).first;
      if ( row_name ) fprintf(m_out, " (%s)", row_name);
      dump_cond_list(tset);
      fprintf(m_out, " col %d", cidx.first);
      auto col_name = get_it(ci->first->cols, cidx.first).first;
      if ( col_name ) fprintf(m_out, " (%s)", col_name);
      dump_cond_list(cidx.second);
      fprintf(m_out, ": %d\n", value.value());
      res++;
    }
  }
  return res;
}

bool NV_renderer::check_dual(const NV_extracted &kv)
{
  if ( m_width != 88 ) return false;
  auto kvi = kv.find("usched_info");
  if ( kvi == kv.end() ) return false;
  return kvi->second == 0x10; // floxy2
}
