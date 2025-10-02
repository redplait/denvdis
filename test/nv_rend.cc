#include <dlfcn.h>
#include <stdarg.h>
#include <fp16.h>
#include "nv_rend.h"

extern int opt_m;
// for sv literals
using namespace std::string_view_literals;

static const char hexes[] = "0123456789ABCDEF";

void HexDump(FILE *f, const unsigned char *From, int Len)
{
 int i;
 int j,k;
 char buffer[256];
 char *ptr;

 for(i=0;i<Len;)
     {
          ptr = buffer;
          sprintf(ptr, "%08X ",i);
          ptr += 9;
          for(j=0;j<16 && i<Len;j++,i++)
          {
             *ptr++ = j && !(j%4)?(!(j%8)?'|':'-'):' ';
             *ptr++ = hexes[From[i] >> 4];
             *ptr++ = hexes[From[i] & 0xF];
          }
          for(k=16-j;k!=0;k--)
          {
            ptr[0] = ptr[1] = ptr[2] = ' ';
            ptr += 3;

          }
          ptr[0] = ptr[1] = ' ';
          ptr += 2;
          for(;j!=0;j--)
          {
               if(From[i-j]>=0x20 && From[i-j]<0x80)
                    *ptr = From[i-j];

               else
                    *ptr = '.';
               ptr++;
          }
          *ptr = 0;
          fprintf(f, "%s\n", buffer);
     }
     fprintf(f, "\n");
}


// ripped from sm_version.txt
std::map<int, std::pair<const char *, const char *> > NV_renderer::s_sms = {
 { 0x14, { "sm2", nullptr } },
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
 "LD_CACHEMODE_INSTR",
 "MBARRIER_INSTR",
 "SW_WAR_MEMBAR_SYS_INSTR",
 "INDIRECT_BRANCH_TARGETS",
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
    case 0x20: // R_CUDA_FUNC_DESC32_LO_23
    case 0x2b: // R_CUDA_ABS32_LO_20
    case 0x30: // R_CUDA_FUNC_DESC32_LO_20
    case 0x38: // R_CUDA_ABS32_LO_32
    case 0x3e: // R_CUDA_FUNC_DESC32_LO_32
    case 0x46: // R_CUDA_32_LO
    case 0x70: // R_CUDA_UNIFIED32_LO_32
     res += "32@lo";
     break;
    case 0xb:  // R_CUDA_ABS32_HI_26
    case 0xe:  // R_CUDA_ABS32_HI_23
    case 0x21: // R_CUDA_FUNC_DESC32_HI_23
    case 0x2c: // R_CUDA_ABS32_HI_20
    case 0x31: // R_CUDA_FUNC_DESC32_HI_20
    case 0x39: // R_CUDA_ABS32_HI_32
    case 0x3f: // R_CUDA_FUNC_DESC32_HI_32
    case 0x47: // R_CUDA_32_HI
    case 0x71: // R_CUDA_UNIFIED32_HI_32
     res += "32@hi";
     break;
    default:
     res += "UnkRel";
     res += std::to_string(nr->first);
  }
  res += '(';
  res += c;
  res += ')';
}

static const char *prop_type_names[] = {
 "INTEGER", // 0
 "SIGNED_INTEGER", // 1
 "UNSIGNED_INTEGER", // 2
 "FLOAT", // 3
 "DOUBLE", // 4
 "GENERIC_ADDRESS", // 5
 "SHARED_ADDRESS", // 6
 "LOCAL_ADDRESS", // 7
 "TRAM_ADDRESS", // 8
 "LOGICAL_ATTR_ADDRESS", // 9
 "PHYSICAL_ATTR_ADDRESS", // 10
 "GENERIC", // 11
 "NON_EXISTENT_OPERAND", // 12
 "CONSTANT_ADDRESS", // 13
 "VILD_INDEX", // 14
 "VOTE_INDEX", // 15
 "STP_INDEX", // 16
 "PIXLD_INDEX", // 17
 "PATCH_OFFSET_ADDRESS", // 18
 "RAW_ISBE_ACCESS", // 19
 "GLOBAL_ADDRESS", // 20
 "TEX", // 21
 "GS_STATE", // 22
 "SURFACE_COORDINATES", // 23
 "FP16SIMD", // 24
 "BINDLESS_CONSTANT_ADDRESS", // 25
 "VERTEX_HANDLE", // 26
 "MEMORY_DESCRIPTOR", // 27
 "FP8SIMD", // 28
};

const char *get_prop_type_name(int i) {
  if ( i < 0 || i >= int(sizeof(prop_type_names) / sizeof(prop_type_names[0])) )
    return nullptr;
  return prop_type_names[i];
}

static const char *prop_op_names[] = {
 "IDEST", // 0
 "IDEST2", // 1
 "ISRC_A", // 2
 "ISRC_B", // 3
 "ISRC_C", // 4
 "ISRC_E", // 5
 "ISRC_H", // 6
};

const char *get_prop_op_name(int i) {
  if ( i < 0 || i >= int(sizeof(prop_op_names) / sizeof(prop_op_names[0])) )
    return nullptr;
  return prop_op_names[i];
}

static const char *lut_ops[] = {
#include "lut/lut.inc"
};

const char *get_lut(int i) {
  if ( i <= 0 ) return nullptr;
  if ( --i >= int(sizeof(lut_ops) / sizeof(lut_ops[0])) )
    return nullptr;
  return lut_ops[i];
}

void NV_renderer::Err(const char *fmt, ...) const
{
 va_list args;
 va_start(args, fmt);
 if ( m_elog )
   m_elog->verr(fmt, &args);
 else
  vfprintf(stderr, fmt, args);
}

void NV_renderer::dis_stat() const
{
  if ( dis_total )
    fprintf(m_out, "total %ld, not_found %ld, dups %ld, missed_enums %ld\n",
     dis_total, dis_notfound, dis_dups, missed_enums);
}

// in sm5x there are lots of registers enums with names like hfma2__v1_Ra
// last letters are always Rd Ra Rb Rc and v can be 0, 1 or 2
// so it's enough to check last 5 letters of ename
bool NV_renderer::crack_h2(const char *ename) const
{
  auto len = strlen(ename);
  if ( len < 6 ) return 0;
  return (ename[len-5] == 'v') &&
   (ename[len-4] == '0' || ename[len-4] == '1' || ename[len-4] == '2') &&
   (ename[len-3] == '_') &&
   (ename[len-2] == 'R') &&
   (ename[len-1] == 'd' || ename[len-1] == 'a' || ename[len-1] == 'b' || ename[len-1] == 'c');
}

int NV_renderer::load(const char *sm_name)
{
     void *dh = dlopen(sm_name, RTLD_NOW);
     if ( !dh ) {
       Err("cannot load %s, errno %d (%s)\n", sm_name, errno, strerror(errno));
       return 0;
     }
     m_vq = (Dvq_name)dlsym(dh, "get_vq_name");
     if ( !m_vq )
       Err("cannot find get_vq_nam(%s), errno %d (%s)\n", sm_name, errno, strerror(errno));
     Dproto fn = (Dproto)dlsym(dh, "get_sm");
     if ( !fn ) {
       Err("cannot find get_sm(%s), errno %d (%s)\n", sm_name, errno, strerror(errno));
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

bool NV_renderer::cmp(const std::string_view &sv, const char *s) {
  size_t i = 0;
  for ( auto c = sv.cbegin(); c != sv.cend(); ++c, ++i ) {
    if ( *c != s[i] ) return false;
  }
  return true;
}

bool NV_renderer::contain(const std::string_view &sv, char sym) const
{
  return sv.find(sym) != std::string::npos;
}

void NV_renderer::dump_tab_fields(const NV_tab_fields *t) const
{
  // make offsets of fields names
  std::vector<int> offsets;
  int prev = 8;
  for ( size_t i = 0; i < t->fields.size(); ++i ) {
      auto fn = get_it(t->fields, i);
      offsets.push_back(prev);
      prev += fn.size() + 1;
      dump_out(fn);
      fputc(' ', m_out);
  }
  fputc('\n', m_out);
  // dump whole tab
  auto tab = t->tab;
  for ( auto &titer: *tab ) {
    fprintf(m_out, " %d\t", titer.first);
    auto ar = titer.second;
    prev = 8;
    for ( int i = 1; i <= ar[0]; ++i ) {
      for ( int p = prev; p < offsets.at(i - 1); ++p ) fputc(' ', m_out);
      prev = offsets.at(i - 1) + fprintf(m_out, "%d", ar[i]);
    }
    fputc('\n', m_out);
  }
}

void NV_renderer::dump_value(const nv_vattr &a, uint64_t v, NV_Format kind, std::string &res) const
{
  constexpr int buf_size = 63;
  char buf[buf_size + 1];
  uint32_t f32;
  switch(kind)
  {
    case NV_F64Imm:
      snprintf(buf, buf_size, "%f", *(double *)&v);
     break;
    case NV_F16Imm:
      f32 = fp16_ieee_to_fp32_bits((uint16_t)v);
      snprintf(buf, buf_size, "%f", *(float *)&f32);
     break;
    case NV_F32Imm:
      snprintf(buf, buf_size, "%f", *(float *)&v);
     break;
    default:
      if ( !v ) { res += '0'; return; }
      snprintf(buf, buf_size, "0x%lX", v);
  }
  buf[buf_size] = 0;
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

// check if some enum is RELONLY
bool NV_renderer::check_rel(const struct nv_instr *ins) const
{
  for ( auto &ei: ins->eas ) {
    if ( cmp("RELONLY", ei.ea->ename) ) return true;
  }
  return false;
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
  else { missed_enums++; return 1; }
  return 0;
}

// render right []
int NV_renderer::render_ve_list(const std::list<ve_base> &l, const struct nv_instr *i, const NV_extracted &kv, std::string &res) const
{
  auto size = l.size();
  if ( 1 == size )
    return render_ve(*l.begin(), i, kv, res);
  int missed = 0, has_prev = 0;
  int idx = 0;
  for ( auto &ve: l ) {
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
      else if ( idx && has_prev ) res += '+';
      res += tmp;
      idx++;
      has_prev = 1;
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
#ifdef DEBUG
 printf("%s: ignore %d has_prev %d def %X curr %lX\n", ea->ename, ea->ignore, has_prev, ea->def_value, kvi->second);
#endif
    if ( ea->has_def_value && ea->def_value == (int)kvi->second ) {
      if ( ea->ignore && !ea->print ) continue;
      // ignore zero register even without ea->ignore
      if ( !strcmp(ea->ename, "ZeroRegister") ) continue;
    }
    if ( !ea->ignore ) idx++;
    if ( ea->ignore ) {
      if ( !has_prev ) continue;
      res += '.';
    } else {
      if ( ve.pfx ) res += ve.pfx;
      else if ( idx > 1 && has_prev ) res += " + ";
    }
    auto eid = ea->em->find(kvi->second);
    if ( eid != ea->em->end() ) {
      res += eid->second;
      has_prev = 1;
    } else {
       missed_enums++;
       continue;
    }
  }
  return missed;
}

void NV_renderer::finalize_rt(reg_pad *rtdb) {
 if ( !rtdb ) return;
 // why we need to sort all those vectors? they already processed by ascending offsets
 // well, bcs we processing operands from left to right
 // so for example: 'imad regZ, regZ' will produce assign first
 //  regZ <- off
 //  regZ off
 // therefore we must sort by mask 0x8000 for the same offsets
 auto srt = [](const reg_history &a, const reg_history &b) -> bool {
   if ( a.off == b.off ) {
     bool res = ((a.kind & 0x8000) < (b.kind & 0x8000));
#ifdef DEBUG
 printf("a %lX %X <-> b %lX %X %d\n", a.off, a.kind, b.off, b.kind, res);
#endif
     return res;
   }
   return a.off < b.off;
 };
 if ( !rtdb->gpr.empty() )
  for ( auto &r: rtdb->gpr ) std::sort(r.second.begin(), r.second.end(), srt);
 if ( !rtdb->ugpr.empty() )
  for ( auto &r: rtdb->ugpr ) std::sort(r.second.begin(), r.second.end(), srt);
 if ( !rtdb->pred.empty() )
  for ( auto &r: rtdb->pred ) std::sort(r.second.begin(), r.second.end(), srt);
 if ( !rtdb->upred.empty() )
  for ( auto &r: rtdb->upred ) std::sort(r.second.begin(), r.second.end(), srt);
}

void NV_renderer::dump_rt(reg_pad *rtdb) const {
  if ( !rtdb ) return;
  if ( !rtdb->gpr.empty() ) {
    fprintf(m_out, ";;; %ld GPR\n", rtdb->gpr.size());
    dump_trset(rtdb->gpr, "R");
  }
  if ( !rtdb->ugpr.empty() ) {
    fprintf(m_out, ";;; %ld UGPR\n", rtdb->ugpr.size());
    dump_trset(rtdb->ugpr, "UR");
  }
  if ( !rtdb->pred.empty() ) {
    fprintf(m_out, ";;; %ld PRED\n", rtdb->pred.size());
    dump_rset(rtdb->pred, "P");
  }
  if ( !rtdb->upred.empty() ) {
    fprintf(m_out, ";;; %ld UPRED\n", rtdb->upred.size());
    dump_rset(rtdb->upred, "UP");
  }
  if ( !rtdb->cbs.empty() ) {
   fprintf(m_out, ";;; %ld CBanks\n", rtdb->cbs.size());
   for ( auto &c: rtdb->cbs )
     fprintf(m_out, " ;   %lX: %X %lX size %d\n", c.off, c.cb_num, c.cb_off, c.kind & 0xf);
  }
}

void NV_renderer::dump_rset(const reg_pad::RSet &rs, const char *pfx) const
{
  constexpr int mask = (1 << 11) - 1;
  for ( auto r: rs ) {
    fprintf(m_out, " ;  %s%d %ld:\n", pfx, r.first, r.second.size());
    for ( auto &tr: r.second ) {
      int pred = 0;
      bool is_pred = tr.has_pred(pred);
      if ( tr.kind & 0x8000 )
      {
        if ( is_pred )
          fprintf(m_out, " ;   %lX <- %X %s%d\n", tr.off, tr.kind & mask, tr.kind & 0x4000 ? "UP" : "P", pred);
        else
          fprintf(m_out, " ;   %lX <- %X\n", tr.off, tr.kind & mask);
      } else {
        if ( is_pred )
          fprintf(m_out, " ;   %lX %X %s%d\n", tr.off, tr.kind & mask, tr.kind & 0x4000 ? "UP" : "P", pred);
        else
          fprintf(m_out, " ;   %lX %X\n", tr.off, tr.kind & mask);
      }
    }
  }
}

void NV_renderer::dump_trset(const reg_pad::TRSet &rs, const char *pfx) const
{
  constexpr int mask = (1 << 11) - 1;
  for ( auto r: rs ) {
    fprintf(m_out, " ;  %s%d %ld:\n", pfx, r.first, r.second.size());
    for ( auto &tr: r.second ) {
      int pred = 0;
      const char *tname = nullptr;
      if ( tr.type != GENERIC ) tname = get_prop_type_name(tr.type);
      bool is_pred = tr.has_pred(pred);
      if ( tr.kind & 0x8000 )
      {
        if ( is_pred )
          fprintf(m_out, " ;   %lX <- %X %s%d", tr.off, tr.kind & mask, tr.kind & 0x4000 ? "UP" : "P", pred);
        else
          fprintf(m_out, " ;   %lX <- %X", tr.off, tr.kind & mask);
      } else {
        if ( is_pred )
          fprintf(m_out, " ;   %lX %X %s%d", tr.off, tr.kind & mask, tr.kind & 0x4000 ? "UP" : "P", pred);
        else
          fprintf(m_out, " ;   %lX %X", tr.off, tr.kind & mask);
      }
      if ( tname ) fprintf(m_out, " %s\n", tname);
      else fputc('\n', m_out);
    }
  }
}

// fill values for tail
int NV_renderer::copy_tail_values(const struct nv_instr *ins, const NV_rlist *rl,
  const NV_extracted &in_values, NV_extracted &out_res) const
{
  int state = 0;
  int res = 0;
  for ( auto r: *rl ) {
    if ( !state ) {
      if ( !is_tail(ins, r) ) continue;
      state = 1;
    }
    if ( r->type != R_enum && r->type != R_value ) continue; // wtd?
    const render_named *rn = (const render_named *)r;
    // check if out_res already contains thus values
    auto ip = out_res.find(rn->name);
    if ( ip != out_res.end() ) continue;
    auto iv = in_values.find(rn->name);
    if ( iv == in_values.end() ) continue;
    out_res[rn->name] = iv->second;
    res++;
  }
  return res;
}

int NV_renderer::make_tab_row(int optv, const struct nv_instr *ins, const NV_tab_fields *tf,
     const NV_extracted &ex, std::vector<unsigned short> &res, int ignore) const
{
  res.clear();
  res.resize(tf->fields.size());
  int bad_cnt = 0;
  for ( int i = 0; int(tf->fields.size()); i++ ) {
    if ( i == ignore ) continue;
    auto &cfname = get_it(tf->fields, i);
    auto kvi = ex.find(cfname);
    if ( kvi == ex.end() ) {
      res[i] = 0;
      if ( optv ) {
        fprintf(m_out, "make_tab_row; cannot find "); dump_outln(cfname);
      }
      bad_cnt++;
    } else
      res[i] = (unsigned short)kvi->second;
  }
  return bad_cnt;
}

int NV_renderer::validate_tabs(const struct nv_instr *ins, NV_extracted &res)
{
  if ( !ins->tab_fields.size() ) return 1; // nothing to check
  // calc max size of tab
  size_t mt = 0;
  for ( auto tf: ins->tab_fields ) mt = std::max(mt, tf->fields.size());
  std::vector<unsigned short> usd;
  usd.reserve(mt);
  int tab_idx = 0;
  for ( auto tf: ins->tab_fields ) {
    usd.clear();
    for ( int i = 0; i < int(tf->fields.size()); i++ ) {
      auto &cfname = get_it(tf->fields, i);
      auto kvi = res.find(cfname);
      if ( kvi == res.end() )
        usd.push_back(0);
      else
        usd.push_back((unsigned short)kvi->second);
    }
    int res_val = 0;
    if ( !ins->check_tab(tf->tab, usd, res_val) ) {
      fprintf(m_out, "check_tab(%d) failed for %d\n", tab_idx, ins->line);
      return 0;
    }
    tab_idx++;
  }
  return 1;
}

bool NV_renderer::check_prmt(const struct nv_instr *ins, const NV_rlist *rend, const NV_extracted &kv, unsigned long &mask) const
{
  int state = 0;
  mask = 0;
  for ( auto &r: *rend ) {
    if ( r->type == R_opcode ) {
      state = 1;
      continue;
    }
    if ( r->type == R_value ) {
      const render_named *rn = (const render_named *)r;
      auto vi = find(ins->vas, rn->name);
      if ( is_tail(vi, rn) ) break;
      if ( state && (vi->kind == NV_SImm || vi->kind == NV_UImm) ) {
        auto kvi = kv.find(rn->name);
        if ( kvi == kv.end() ) return false;
        mask = (int)kvi->second;
        return true;
      }
    }
  }
  return false;
}

bool NV_renderer::check_lut(const struct nv_instr *ins, const NV_rlist *rend, const NV_extracted &kv, int &idx) const
{
  // 1 - opcode, 2 - has enum LUTOnly
  int state = 0;
  idx = 0;
  for ( auto &r: *rend ) {
    if ( r->type == R_opcode ) {
      state = 1;
      continue;
    }
    // check if we have tail - then end loop
    if ( r->type == R_value ) {
      if ( 2 != state ) return false;
      const render_named *rn = (const render_named *)r;
      auto vi = find(ins->vas, rn->name);
      if ( is_tail(vi, rn) ) break;
      if ( 2 == state && (!strcasecmp(rn->name, "imm8") || !strcasecmp(rn->name, "uimm8")) ) {
        auto kvi = kv.find(rn->name);
        if ( kvi == kv.end() ) return false;
        idx = (int)kvi->second;
        return true;
      }
    }
    // check LUTOnly enum
    if ( state == 1 && r->type == R_enum ) {
      const render_named *rn = (const render_named *)r;
      const nv_eattr *ea = find_ea(ins, rn->name);
      if ( !ea ) continue;
      if ( !ea->ignore ) return false;
      if ( !strcmp(ea->ename, "LUTOnly") ) state = 2;
    }
  }
  return false;
}

int NV_renderer::track_regs(reg_pad *rtdb, const NV_rlist *rend, const NV_pair &p, unsigned long off)
{
  int res = 0;
  bool has_props = p.first->props != nullptr;
  const std::string_view *d_sv = nullptr,
   *d2_sv = nullptr,
   *a_sv = nullptr,
   *b_sv = nullptr,
   *c_sv = nullptr,
   *e_sv = nullptr,
   *h_sv = nullptr;
  NVP_type t1 = GENERIC, t2 = GENERIC,
   t_a = GENERIC, t_b = GENERIC, t_c = GENERIC, t_e = GENERIC, t_h = GENERIC;
  int ends2 = 0;
  bool setp = is_setp(p.first, ends2);
  if ( has_props ) {
    for ( auto pr: *p.first->props ) {
      if ( pr->op == IDEST ) {
        t1 = pr->t;
        if ( pr->fields.size() == 1 ) d_sv = &get_it(pr->fields, 0);
        continue;
      }
      if ( pr->op == IDEST2 ) {
        t2 = pr->t;
        if ( pr->fields.size() == 1 ) d2_sv = &get_it(pr->fields, 0);
        continue;
      }
      if ( pr->op == ISRC_A ) {
        t_a = pr->t;
        if ( pr->fields.size() == 1 ) a_sv = &get_it(pr->fields, 0);
        continue;
      }
      if ( pr->op == ISRC_B ) {
        t_b = pr->t;
        if ( pr->fields.size() == 1 ) b_sv = &get_it(pr->fields, 0);
        continue;
      }
      if ( pr->op == ISRC_C ) {
        t_c = pr->t;
        if ( pr->fields.size() == 1 ) c_sv = &get_it(pr->fields, 0);
        continue;
      }
      if ( pr->op == ISRC_E ) {
        t_e = pr->t;
        if ( pr->fields.size() == 1 ) e_sv = &get_it(pr->fields, 0);
        continue;
      }
      if ( pr->op == ISRC_H ) {
        t_h = pr->t;
        if ( pr->fields.size() == 1 ) h_sv = &get_it(pr->fields, 0);
        continue;
      }
    }
  }
  // predicates
  int d_size = 0, d2_size = 0, a_size = 0, b_size = 0, c_size = 0, e_size = 0;
  if ( p.first->predicated ) {
    auto pi = p.first->predicated->find("IDEST_SIZE"sv);
    if ( pi != p.first->predicated->end() )
      d_size = pi->second(p.second);
    pi = p.first->predicated->find("IDEST2_SIZE"sv);
    if ( pi != p.first->predicated->end() )
      d2_size = pi->second(p.second);
    pi = p.first->predicated->find("ISRC_A_SIZE"sv);
    if ( pi != p.first->predicated->end() )
      a_size = pi->second(p.second);
    pi = p.first->predicated->find("ISRC_B_SIZE"sv);
    if ( pi != p.first->predicated->end() )
      b_size = pi->second(p.second);
    pi = p.first->predicated->find("ISRC_C_SIZE"sv);
    if ( pi != p.first->predicated->end() )
      c_size = pi->second(p.second);
    pi = p.first->predicated->find("ISRC_E_SIZE"sv);
    if ( pi != p.first->predicated->end() )
      e_size = pi->second(p.second);
  }
  int idx = -1;
  rtdb->pred_mask = 0;
  if ( is_s2xx(p.first) ) rtdb->pred_mask = (1 << 10);
  for ( auto &r: *rend ) {
    // check if we have tail - then end loop
    if ( r->type == R_value ) {
      const render_named *rn = (const render_named *)r;
      auto vi = find(p.first->vas, rn->name);
      if ( is_tail(vi, rn) ) break;
      idx++;
      continue;
    }
    unsigned short cb_idx = 0;
    unsigned long cb_off = 0;
    if ( check_cbank(p.first, r, p.second, cb_idx, cb_off) ) {
      rtdb->add_cb(off, cb_off, cb_idx, d_size >> 3);
      idx++;
      continue;
    }
    // predicate - before opcode
    if ( idx < 0 && r->type == R_predicate ) {
      // check if this is not PT
      const render_named *rn = (const render_named *)r;
      const nv_eattr *ea = find_ea(p.first, rn->name);
      if ( !ea ) continue;
      auto kvi = p.second.find(rn->name);
      if ( kvi == p.second.end() ) continue;
      if ( kvi->second == 7 ) continue;
      if ( !strcmp(ea->ename, "Predicate") )
       { rtdb->pred_mask = (1 + (unsigned short)kvi->second) << 11;
         rtdb->rpred(kvi->second, off, 0); res++; }
      else if ( !strcmp(ea->ename, "UniformPredicate") )
       { rtdb->pred_mask = 0x4000 | (1 + (unsigned short)kvi->second) << 11;
         rtdb->rupred(kvi->second, off, 0); res++; }
      else
       fprintf(m_out, "unknown predicate %s at %lX\n", ea->ename, off);
      continue;
    }
    // xxSETP
    if ( setp && !idx && (r->type == R_predicate || r->type == R_enum) ) {
      const render_named *rn = (const render_named *)r;
      const nv_eattr *ea = find_ea(p.first, rn->name);
      if ( !ea ) continue;
      if ( ea->ignore ) continue;
      auto kvi = p.second.find(rn->name);
      if ( kvi == p.second.end() ) continue;
      if ( !strcmp(ea->ename, "Predicate") && kvi->second != 7 )
       { rtdb->wpred(kvi->second, off, 0); res++; }
      else if ( !strcmp(ea->ename, "UniformPredicate") && kvi->second != 7 )
       { rtdb->wupred(kvi->second, off, 0); res++; }
      idx++;
      continue;
    }
    // it seems that some SETP variants can assign 2 predicate register in one instruction, like
    //  DSETP.MAX.AND P2, P3, R2, R12, PT
    // here first predicate in MD described as Pu and next as Pv
    // those second Pv will have idx == 1 (Pu - 0)
    // also PSETP in old MDs has fields Pd & nPd
    if ( setp && idx == 1 && (r->type == R_predicate || r->type == R_enum) ) {
      const render_named *rn = (const render_named *)r;
      if ( !strcmp("nPd", rn->name) ||
           (ends2 && (!strcmp("Pv", rn->name) || !strcmp("UPv", rn->name))) ) {
        const nv_eattr *ea = find_ea(p.first, rn->name);
        if ( !ea ) continue;
        if ( ea->ignore ) continue;
        auto kvi = p.second.find(rn->name);
        if ( kvi == p.second.end() ) continue;
        if ( !strcmp(ea->ename, "Predicate") && kvi->second != 7 )
         { rtdb->wpred(kvi->second, off, 0); res++; }
        else if ( !strcmp(ea->ename, "UniformPredicate") && kvi->second != 7 )
         { rtdb->wupred(kvi->second, off, 0); res++; }
        idx++;
        continue;
      }
    }
    if ( r->type == R_opcode ) {
      idx = 0;
      continue;
    }
    auto rgpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi, NVP_type _t = GENERIC) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = i;
        if ( (int)kvi->second + i >= m_dis->rz ) break;
        rtdb->rgpr(kvi->second + i, off, what, _t);
        res++;
      }
      return res;
    };
    auto gpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi, NVP_type _t = GENERIC) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = i;
        if ( (int)kvi->second + i >= m_dis->rz ) break;
        rtdb->wgpr(kvi->second + i, off, what, _t);
        res++;
      }
      return res;
    };
    auto rugpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi, NVP_type _t = GENERIC) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = i;
        if ( (int)kvi->second + i >= m_dis->rz ) break;
        rtdb->rugpr(kvi->second + i, off, what, _t);
        res++;
      }
      return res;
    };
    auto ugpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi, NVP_type _t = GENERIC) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = i;
        if ( (int)kvi->second + i >= m_dis->rz ) break;
        rtdb->wugpr(kvi->second + i, off, what, _t);
        res++;
      }
      return res;
    };
    // dest(2)
    if ( idx >= 0 && (r->type == R_predicate || r->type == R_enum) ) {
      const render_named *rn = (const render_named *)r;
      const nv_eattr *ea = find_ea(p.first, rn->name);
      if ( !ea ) continue;
      if ( ea->ignore ) continue;
      auto kvi = p.second.find(rn->name);
      if ( kvi == p.second.end() ) continue;
      if ( is_pred(ea, kvi) )
       { rtdb->rpred(kvi->second, off, 0); res++; }
      else if ( is_upred(ea, kvi) )
       { rtdb->rupred(kvi->second, off, 0); res++; }
      else if ( is_reg(ea, kvi) )
      {
        if ( is_sv(d_sv, rn->name) ) {
         if ( d_size <= 32 )
          { rtdb->wgpr(kvi->second, off, 0, t1); res++; }
         else res += gpr_multi(d_size, kvi, t1);
        } else if ( is_sv(d2_sv, rn->name) ) {
         if ( d2_size <= 32 )
         { rtdb->wgpr(kvi->second, off, 0, t2); res++; }
         else res += gpr_multi(d2_size, kvi, t2);
        } else if ( !strcmp(rn->name, "Rd") ) {
         if ( d_size <= 32 )
          { rtdb->wgpr(kvi->second, off, 0, t1); res++; }
         else res += gpr_multi(d_size, kvi, t1);
        } else if ( !strcmp(rn->name, "Rd2") ) {
         if ( d2_size <= 32 )
          { rtdb->wgpr(kvi->second, off, 0, t2); res++; }
         else res += gpr_multi(d2_size, kvi, t2);
        } else {
         if ( a_size > 32 && is_sv2(a_sv, rn->name, "Ra") )
          res += rgpr_multi(a_size, kvi, t_a);
         else if ( b_size > 32 && is_sv2(b_sv, rn->name, "Rb") )
          res += rgpr_multi(b_size, kvi, t_b);
         else if ( c_size > 32 && is_sv2(c_sv, rn->name, "Rc") )
          res += rgpr_multi(c_size, kvi, t_c);
         else if ( e_size > 32 && is_sv2(e_sv, rn->name, "Re") )
          res += rgpr_multi(e_size, kvi, t_e);
         else
         {
           if ( is_sv2(a_sv, rn->name, "Ra") )
             rtdb->rgpr(kvi->second, off, 0, t_a);
           else if ( is_sv2(b_sv, rn->name, "Rb") )
             rtdb->rgpr(kvi->second, off, 0, t_b);
           else if ( is_sv2(c_sv, rn->name, "Rc") )
             rtdb->rgpr(kvi->second, off, 0, t_c);
           else if ( is_sv2(e_sv, rn->name, "Re") )
             rtdb->rgpr(kvi->second, off, 0, t_e);
           else if ( is_sv2(h_sv, rn->name, "Rh") ) // NOTE - ISRC_H_SIZE is always 32bit at time of writing this
             rtdb->rgpr(kvi->second, off, 0, t_h);
           else rtdb->rgpr(kvi->second, off, 0);
           res++;
         }
        }
      } else if ( is_ureg(ea, kvi) )
      {
        if ( is_sv(d_sv, rn->name) ) {
         if ( d_size <= 32 )
          { rtdb->wugpr(kvi->second, off, 0, t1); res++; }
          else res += ugpr_multi(d_size, kvi, t1);
        } else if ( is_sv(d2_sv, rn->name) ) {
         if ( d2_size <= 32 )
           { rtdb->wugpr(kvi->second, off, 0, t2); res++; }
         else res += ugpr_multi(d2_size, kvi, t2);
        } else if ( !strcmp(rn->name, "URd") ) {
         if ( d_size <= 32 )
          { rtdb->wugpr(kvi->second, off, 0, t1); res++; }
         else res += ugpr_multi(d_size, kvi, t1);
        } else if ( !strcmp(rn->name, "URd2") ) {
         if ( d2_size <= 32 )
          { rtdb->wugpr(kvi->second, off, 0, t2); res++; }
         else res += ugpr_multi(d2_size, kvi, t2);
        } else {
         if ( a_size > 32 && is_sv2(a_sv, rn->name, "URa") )
          res += rugpr_multi(a_size, kvi, t_a);
         else if ( b_size > 32 && is_sv2(b_sv, rn->name, "URb") )
          res += rugpr_multi(b_size, kvi, t_b);
         else if ( c_size > 32 && is_sv2(c_sv, rn->name, "URc") )
          res += rugpr_multi(c_size, kvi, t_c);
         else if ( e_size > 32 && is_sv2(e_sv, rn->name, "URe") )
          res += rugpr_multi(e_size, kvi, t_e);
         else
         {
           if ( is_sv2(a_sv, rn->name, "URa") )
             rtdb->rugpr(kvi->second, off, 0, t_a);
           else if ( is_sv2(a_sv, rn->name, "URb") )
             rtdb->rugpr(kvi->second, off, 0, t_b);
           else if ( is_sv2(a_sv, rn->name, "URc") )
             rtdb->rugpr(kvi->second, off, 0, t_c);
           else if ( is_sv2(a_sv, rn->name, "URe") )
             rtdb->rugpr(kvi->second, off, 0, t_e);
           else rtdb->rugpr(kvi->second, off, 0);
           res++;
         }
        }
      }
    }
    // ok, we have something compound
    auto ve_type = [&](const ve_base &ve) -> NVP_type {
      auto len = strlen(ve.arg);
      if ( len < 2 ) return GENERIC;
      if ( len > 7 && !strcmp(ve.arg + len - 7, "_offset") ) len -= 7;
      if ( ve.arg[len - 2] != 'R' ) return GENERIC;
      if ( ve.arg[len - 1] == 'a' ) return t_a;
      if ( ve.arg[len - 1] == 'b' ) return t_b;
      if ( ve.arg[len - 1] == 'e' ) return t_e;
      if ( ve.arg[len - 1] == 'h' ) return t_h;
      return GENERIC;
    };
    auto check_ve = [&](const ve_base &ve, reg_history::RH what) {
        if ( ve.type == R_value ) return 0;
        const nv_eattr *ea = find_ea(p.first, ve.arg);
        if ( !ea ) return 0;
        auto kvi = p.second.find(ve.arg);
        if ( kvi == p.second.end() ) return 0;
        // check what we have
        if ( is_pred(ea, kvi) )
        { rtdb->rpred(kvi->second, off, what); return 1; }
        if ( is_upred(ea, kvi) )
        { rtdb->rupred(kvi->second, off, what); return 1; }
        if ( is_reg(ea, kvi) )
        { rtdb->rgpr(kvi->second, off, what, ve_type(ve)); return 1; }
        if ( is_ureg(ea, kvi) )
        { rtdb->rugpr(kvi->second, off, what, ve_type(ve)); return 1; }
        return 0;
    };
    auto check_ve_list = [&](const std::list<ve_base> &l, reg_history::RH what) {
        int res = 0;
        for ( auto &ve: l ) {
          if ( ve.type == R_value ) continue;
          const nv_eattr *ea = find_ea(p.first, ve.arg);
          if ( !ea ) continue;
          if ( ea->ignore ) continue;
          res += check_ve(ve, what);
        }
        return res;
    };
#ifdef DEBUG
 fprintf(m_out, "@%lX: r->type %d\n", off, r->type);
#endif
    if ( r->type == R_C || r->type == R_CX ) {
      const render_C *rn = (const render_C *)r;
      res += check_ve(rn->left, 1 << 3);
      res += check_ve_list(rn->right, 4 | (1 << 3));
    } else if ( r->type == R_desc ) {
      const render_desc *rd = (const render_desc *)r;
      res += check_ve(rd->left, 2 << 3);
      res += check_ve_list(rd->right, 4 | (2 << 3));
    } else if ( r->type == R_mem ) {
      const render_mem *rm = (const render_mem *)r;
      res += check_ve_list(rm->right, 4 | (3 << 3));
    } else if ( r->type == R_TTU ) {
      const render_TTU *rt = (const render_TTU *)r;
      res += check_ve(rt->left, 3 << 3);
    } else if ( r->type == R_M1 ) {
      const render_M1 *rt = (const render_M1 *)r;
      res += check_ve(rt->left, 3 << 3);
    }
    idx++;
    continue;
  }
  return res;
}

const char *NV_renderer::has_predicate(const NV_rlist *rl) const
{
  for ( auto r: *rl ) {
    if ( r->type == R_predicate ) {
      const render_named *rn = (const render_named *)r;
      return rn->name;
    }
    if ( r->type == R_opcode ) break;
  }
  return nullptr;
}

bool NV_renderer::has_not(const render_named *rn, const NV_extracted &kv) const
{
  std::string n_not = rn->name;
  n_not += "@not";
  auto kvi = kv.find(n_not);
  return ( kvi != kv.end() && kvi->second);
}

bool NV_renderer::always_false(const struct nv_instr *i, const NV_rlist *rl, const NV_extracted &kv) const
{
  for ( auto r: *rl ) {
    if ( r->type == R_opcode ) break;
    if ( r->type == R_predicate ) {
      const render_named *rn = (const render_named *)r;
      // check for value 7
      auto kvi = kv.find(rn->name);
      if ( kvi != kv.end() && kvi->second != 7 ) return false;
      // check for @not
      return has_not(rn, kv);
    }
  }
  return false;
}

bool NV_renderer::is_s2xx(const struct nv_instr *i) const
{
  return !strcmp(i->name, "S2R") ||
   !strcmp(i->name, "CS2R") ||
   !strcmp(i->name, "S2UR");
}

bool NV_renderer::is_setp(const struct nv_instr *i, int &ends2) const
{
  if ( 2 == i->setp )
   ends2 = 1;
  else
   ends2 = 0;
  return i->setp;
}

bool NV_renderer::check_ret(const struct nv_instr *i, const NV_extracted::const_iterator &kvi, long &res) const
{
  if ( i->brt != BRT_RETURN ) return false;
  return extract(i, kvi, res);
}

bool NV_renderer::extract(const struct nv_instr *i, const NV_extracted::const_iterator &kvi, long &res) const
{
  auto wi = find(i->vwidth, kvi->first);
  if ( !wi ) return false;
  // yes, this is some imm for branch, check if it negative
  if ( kvi->second & (1L << (wi->w - 1)) )
    res = kvi->second - (1L << wi->w);
  else
    res = (long)kvi->second;
  return true;
}

bool NV_renderer::conv_simm(const struct nv_instr *i, const NV_extracted::const_iterator &kvi, long &res) const
{
  auto wi = find(i->vwidth, kvi->first);
  if ( !wi ) return false;
  // check if value negative
  if ( kvi->second & (1L << (wi->w - 1)) ) {
    res = -1; // all ff
    for ( int i = 0; i < wi->w - 1; i++ ) {
      auto mask = kvi->second & (1L << i); // check bit at index i
      if ( !mask ) res &= ~(1L << i);
    }
  } else
    res = (long)kvi->second;
  return true;
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
  return extract(i, kvi, res);
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
        if ( opcode ) Err("unknown rend type %d for inst %s\n", r->type, opcode);
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
  for ( auto &ve: l ) {
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

std::optional<long> NV_renderer::check_cbank_right(const std::list<ve_base> &rl, const NV_extracted &kv) const
{
  for ( auto &ve: rl ) {
    if ( ve.type == R_value )
    {
      auto kvi = kv.find(ve.arg);
      if ( kvi == kv.end() ) continue;
      return std::optional<long>(kvi->second);
    }
  }
  return std::nullopt;
}

std::optional<long> NV_renderer::check_cbank(const NV_rlist *rl, const NV_extracted &kv, unsigned short *cb_idx) const
{
  for ( auto ri: *rl ) {
    if ( ri->type == R_C || ri->type == R_CX ) {
      const render_C *rn = (const render_C *)ri;
      if ( rn->left.type != R_value )
        return std::nullopt;
      auto kvi = kv.find(rn->left.arg);
      if ( kvi == kv.end() )
        return std::nullopt;
      if ( cb_idx ) *cb_idx = (unsigned short)(kvi->second);
      // if ( kvi->second ) return std::nullopt;
      return check_cbank_right(rn->right, kv);
    }
  }
  return std::nullopt;
}

bool NV_renderer::check_cbank(const struct nv_instr *i, const render_base *rb, const NV_extracted &kv, unsigned short &cb_idx,
     unsigned long &cb_off) const
{
  if ( rb->type != R_C && rb->type != R_CX ) return false;
  const render_C *rn = (const render_C *)rb;
  if ( rn->left.type != R_value )
    return false;
  auto kvi = kv.find(rn->left.arg);
  if ( kvi == kv.end() )
    return false;
  cb_idx = (unsigned short)kvi->second;
  auto res = check_cbank_right(rn->right, kv);
  if ( !res.has_value() ) return false;
  cb_off = res.value();
  return true;
}

int NV_renderer::render(const NV_rlist *rl, std::string &res, const struct nv_instr *i,
 const NV_extracted &kv, NV_labels *l, int opt_c) const
{
  int idx = 0;
  int missed = 0;
  int in_tail = 0; // seems that scheduling args often starts with BITSET req_xx
  int prev = -1;  // workaround to fix op, bcs testcc is missed
  const NV_rel *rel_info = nullptr;
  std::string_view rel_name;
  if ( has_relocs && m_dis->offset() == m_next_roff ) {
    rel_info = next_reloc(rel_name);
  }
  for ( auto ri: *rl ) {
    std::string tmp;
    int is_abs = 0, empty = 0;
    if ( !in_tail && is_tail(i, ri) ) in_tail = 1;
    switch(ri->type)
    {
      case R_opcode:
       res += i->name;
       break;

      case R_value: {
        constexpr int buf_size = 511;
        char buf[buf_size + 1];
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
        if ( in_tail && opt_c ) {
          // unfortunatelly nvdisasm can dump only 4 fields at tail
          // req_bit_set: &req={bit mask}
          // src_rel_sb: &rd=0xnum
          // dst_wr_sb:  &wr=0xnum
          // usched_info: ?enum_name
          // two last - batch_t & pm_pred - should be dumped with '?' prefix only they are non-defailted values
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
             snprintf(buf, buf_size, "%lX", kvi->second);
             buf[buf_size] = 0;
             tmp += buf;
            }
          } else if ( !strcmp(rn->name, "dst_wr_sb") ) {
            if ( kvi->second != vi->dval ) {
             tmp = " &wr=0x";
             snprintf(buf, buf_size, "%lX", kvi->second);
             buf[buf_size] = 0;
             tmp += buf;
            }
          }
        } else {
          long branch_off = 0;
          if ( rel_info && i->target_index && !strcmp(rn->name, i->target_index) ) {
            render_rel(tmp, rel_info, rel_name);
            rel_info = nullptr;
          } else if ( check_ret(i, kvi, branch_off) ) {
            long addr = check_rel(i) ? branch_off + m_dis->off_next() : branch_off;
            auto lname = try_name(addr);
            // make (LABEL_xxx)
            if ( opt_c ) {
              if ( lname )
                snprintf(buf, buf_size, " `(%s)", lname->c_str());
              else
                snprintf(buf, buf_size, " `(LABEL_%lX)", addr);
            } else {
              snprintf(buf, buf_size, "%ld", branch_off);
              buf[buf_size] = 0;
              tmp = buf;
              if ( lname )
                snprintf(buf, buf_size, " (%s)", lname->c_str());
              else
                snprintf(buf, buf_size, " (LABEL_%lX)", addr);
            }
            if ( l && !lname ) (*l)[addr] = 0;
            buf[buf_size] = 0;
            tmp += buf;
          } else if ( check_branch(i, kvi, branch_off) ) {
            long addr = branch_off + m_dis->off_next();
            auto lname = try_name(addr);
            // make (LABEL_xxx)
            if ( opt_c ) {
              if ( lname )
                snprintf(buf, buf_size, " `(%s)", lname->c_str());
              else
                snprintf(buf, buf_size, " `(LABEL_%lX)", addr);
            } else {
              snprintf(buf, buf_size, "%ld", branch_off);
              buf[buf_size] = 0;
              tmp = buf;
              if ( lname )
                snprintf(buf, buf_size, " (%s)", lname->c_str());
              else
                snprintf(buf, buf_size, " (LABEL_%lX)", addr);
            }
            if ( l && !lname ) (*l)[addr] = 0;
            buf[buf_size] = 0;
            tmp += buf;
          } else {
             if ( rel_info && !vi->dval ) {
               render_rel(tmp, rel_info, rel_name);
               rel_info = nullptr;
             } else
               dump_value(i, kv, rn->name, tmp, *vi, kvi->second);
          }
          if ( rn->pfx ) { if ( prev != R_opcode ) res += rn->pfx; }
          res += ' ';
          if ( !rn->pfx && in_tail ) res += '&';
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
         auto kvi = kv.find(rn->name);
         if ( kvi == kv.end() ) {
           kvi = kv.find(ea->ename);
           if ( kvi == kv.end() ) {
             // special case - this is enum attached to opcode with single value
             if ( ea->ignore && prev == R_opcode && 1 == ea->em->size() ) {
               res += '.'; empty = 1;
               res += ea->em->cbegin()->second;
               break;
             }
             if ( opt_m ) m_missed.insert(rn->name);
             missed++;
             idx++;
             continue;
           }
         }
         if ( in_tail && opt_c ) {
           if ( ea->has_def_value && ea->def_value == (int)kvi->second ) continue;
           auto eid = ea->em->find(kvi->second);
           if ( eid != ea->em->end() ) {
             res += " ?";
             res += eid->second;
           } else {
             missed_enums++;
             break;
           }
           continue;
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
           missed_enums++;
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
         if ( ea->def_value == (int)kvi->second && !has_not(rn, kv) ) { empty = 1; break; }
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

      default: Err("unknown rend type %d at index %d for inst %s\n", ri->type, idx, i->name);
    }
    if ( !empty )
      prev = ri->type;
    idx++;
  }
  return missed;
}

bool NV_renderer::_cmp_prop(const std::list<ve_base> &vb, const NV_Prop *pr) const
{
  for ( auto &v: vb ) {
    for ( size_t idx = 0; idx < pr->fields.size(); ++idx ) {
      auto &f = get_it(pr->fields, idx);
      if ( cmp(f, v.arg) ) return true;
    }
  }
  return false;
}

const render_base *NV_renderer::try_compound_prop(const NV_rlist *r, const NV_Prop *pr) const
{
  auto psize = pr->fields.size();
  if ( psize < 2 ) return nullptr;
  for ( auto ri: *r ) {
    switch(ri->type) {
      case R_C:
      case R_CX: {
         const render_C *rn = (const render_C *)ri;
         if ( _cmp_prop(rn->right, pr) ) return rn;
       }
       break;
      case R_desc: {
         const render_desc *rd = (const render_desc *)ri;
         if ( _cmp_prop(rd->right, pr) ) return rd;
       }
       break;
     case R_mem: {
         const render_mem *rm = (const render_mem *)ri;
         if ( _cmp_prop(rm->right, pr) ) return rm;
       }
       break;
     default: ;
    }
  }
  return nullptr;
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

int NV_renderer::dump_op_props(const struct nv_instr *i, FILE *fp, const char *pfx) const
{
  if ( !i->props ) return 0;
  int res = 0;
  for ( size_t pi = 0; pi < i->props->size(); pi++ ) {
    auto curr = get_it(*i->props, pi);
    fprintf(fp, "%s%s type %s: ", pfx, get_prop_op_name(curr->op), get_prop_type_name(curr->t));
    for ( size_t fi = 0; fi < curr->fields.size(); ++fi ) {
      if ( fi ) fputs(", ", fp);
      dump_out(get_it(curr->fields, fi), fp);
    }
    fputc('\n', fp);
  }
  return res;
}

void NV_renderer::dump_predicates(const struct nv_instr *i, const NV_extracted &kv, const char *pfx) const
{
  if ( i->props )
   dump_op_props(i, m_out, pfx);
  if ( !i->predicated ) return;
  dump_predicates(i, kv, m_out, pfx);
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

bool NV_renderer::check_dual(const NV_extracted &kv) const
{
  if ( m_width != 88 ) return false;
  auto kvi = kv.find("usched_info");
  if ( kvi == kv.end() ) return false;
  return kvi->second == 0x10; // floxy2
}
