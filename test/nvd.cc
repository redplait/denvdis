#include <stdlib.h>
#include <stdio.h>
#include <map>
#include <dlfcn.h>
#include <getopt.h>
#include "elfio/elfio.hpp"
#include "include/nv_types.h"

int opt_e = 0,
    opt_h = 0,
    opt_t = 0,
    opt_N = 0,
    opt_O = 0;

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

using namespace ELFIO;

// ripped from sm_version.txt
static std::map<int, std::pair<const char *, const char *> > s_sms = {
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

static std::map<char, const char *> s_ei = {
 { 0x1, "EIATTR_PAD" },
 { 0x2, "EIATTR_IMAGE_SLOT" },
 { 0x3, "EIATTR_JUMPTABLE_RELOCS" },
 { 0x4, "EIATTR_CTAIDZ_USED" },
 { 0x5, "EIATTR_MAX_THREADS" },
 { 0x6, "EIATTR_IMAGE_OFFSET" },
 { 0x7, "EIATTR_IMAGE_SIZE" },
 { 0x8, "EIATTR_TEXTURE_NORMALIZED" },
 { 0x9, "EIATTR_SAMPLER_INIT" },
 { 0xa, "EIATTR_PARAM_CBANK" },
 { 0xb, "EIATTR_SMEM_PARAM_OFFSETS" },
 { 0xc, "EIATTR_CBANK_PARAM_OFFSETS" },
 { 0xd, "EIATTR_SYNC_STACK" },
 { 0xe, "EIATTR_TEXID_SAMPID_MAP" },
 { 0xf, "EIATTR_EXTERNS" },
 { 0x10, "EIATTR_REQNTID" },
 { 0x11, "EIATTR_FRAME_SIZE" },
 { 0x12, "EIATTR_MIN_STACK_SIZE" },
 { 0x13, "EIATTR_SAMPLER_FORCE_UNNORMALIZED" },
 { 0x14, "EIATTR_BINDLESS_IMAGE_OFFSETS" },
 { 0x15, "EIATTR_BINDLESS_TEXTURE_BANK" },
 { 0x16, "EIATTR_BINDLESS_SURFACE_BANK" },
 { 0x17, "EIATTR_KPARAM_INFO" },
 { 0x18, "EIATTR_SMEM_PARAM_SIZE" },
 { 0x19, "EIATTR_CBANK_PARAM_SIZE" },
 { 0x1a, "EIATTR_QUERY_NUMATTRIB" },
 { 0x1b, "EIATTR_MAXREG_COUNT" },
 { 0x1c, "EIATTR_EXIT_INSTR_OFFSETS" },
 { 0x1d, "EIATTR_S2RCTAID_INSTR_OFFSETS" },
 { 0x1e, "EIATTR_CRS_STACK_SIZE" },
 { 0x1f, "EIATTR_NEED_CNP_WRAPPER" },
 { 0x20, "EIATTR_NEED_CNP_PATCH" },
 { 0x21, "EIATTR_EXPLICIT_CACHING" },
 { 0x22, "EIATTR_ISTYPEP_USED" },
 { 0x23, "EIATTR_MAX_STACK_SIZE" },
 { 0x24, "EIATTR_SUQ_USED" },
 { 0x25, "EIATTR_LD_CACHEMOD_INSTR_OFFSETS" },
 { 0x26, "EIATTR_LOAD_CACHE_REQUEST" },
 { 0x27, "EIATTR_ATOM_SYS_INSTR_OFFSETS" },
 { 0x28, "EIATTR_COOP_GROUP_INSTR_OFFSETS" },
 { 0x29, "EIATTR_COOP_GROUP_MASK_REGIDS" },
 { 0x2a, "EIATTR_SW1850030_WAR" },
 { 0x2b, "EIATTR_WMMA_USED" },
 { 0x2c, "EIATTR_HAS_PRE_V10_OBJECT" },
 { 0x2d, "EIATTR_ATOMF16_EMUL_INSTR_OFFSETS" },
 { 0x2e, "EIATTR_ATOM16_EMUL_INSTR_REG_MAP" },
 { 0x2f, "EIATTR_REGCOUNT" },
 { 0x30, "EIATTR_SW2393858_WAR" },
 { 0x31, "EIATTR_INT_WARP_WIDE_INSTR_OFFSETS" },
 { 0x32, "EIATTR_SHARED_SCRATCH" },
 { 0x33, "EIATTR_STATISTICS" },
 { 0x34, "EIATTR_INDIRECT_BRANCH_TARGETS" },
 { 0x35, "EIATTR_SW2861232_WAR" },
 { 0x36, "EIATTR_SW_WAR" },
 { 0x37, "EIATTR_CUDA_API_VERSION" },
 { 0x38, "EIATTR_NUM_MBARRIERS" },
 { 0x39, "EIATTR_MBARRIER_INSTR_OFFSETS" },
 { 0x3a, "EIATTR_COROUTINE_RESUME_ID_OFFSETS" },
 { 0x3b, "EIATTR_SAM_REGION_STACK_SIZE" },
 { 0x3c, "EIATTR_PER_REG_TARGET_PERF_STATS" },
 { 0x3d, "EIATTR_CTA_PER_CLUSTER" },
 { 0x3e, "EIATTR_EXPLICIT_CLUSTER" },
 { 0x3f, "EIATTR_MAX_CLUSTER_RANK" },
 { 0x40, "EIATTR_INSTR_REG_MAP" },
 { 0x41, "EIATTR_RESERVED_SMEM_USED" },
 { 0x42, "EIATTR_RESERVED_SMEM_0_SIZE" },
 { 0x43, "EIATTR_UCODE_SECTION_DATA" },
 { 0x44, "EIATTR_UNUSED_LOAD_BYTE_OFFSET" },
 { 0x45, "EIATTR_KPARAM_INFO_V2" },
 { 0x46, "EIATTR_SYSCALL_OFFSETS" },
 { 0x47, "EIATTR_SW_WAR_MEMBAR_SYS_INSTR_OFFSETS" },
 { 0x48, "EIATTR_GRAPHICS_GLOBAL_CBANK" },
 { 0x49, "EIATTR_SHADER_TYPE" },
 { 0x4a, "EIATTR_VRC_CTA_INIT_COUNT" },
 { 0x4b, "EIATTR_TOOLS_PATCH_FUNC" },
 { 0x4c, "EIATTR_NUM_BARRIERS" },
 { 0x4d, "EIATTR_TEXMODE_INDEPENDENT" },
 { 0x4e, "EIATTR_PERF_STATISTICS" },
 { 0x4f, "EIATTR_AT_ENTRY_FRAGMENTS" },
 { 0x50, "EIATTR_SPARSE_MMA_MASK" },
 { 0x51, "EIATTR_TCGEN05_1CTA_USED" },
 { 0x52, "EIATTR_TCGEN05_2CTA_USED" },
 { 0x53, "EIATTR_GEN_ERRBAR_AT_EXIT" },
 { 0x54, "EIATTR_REG_RECONFIG" },
 { 0x55, "EIATTR_ANNOTATIONS" },
 { 0x56, "EIATTR_SANITIZE" },
 { 0x57, "EIATTR_STACK_CANARY_TRAP_OFFSETS" },
 { 0x58, "EIATTR_STUB_FUNCTION_KIND" },
 { 0x59, "EIATTR_LOCAL_CTA_ASYNC_STORE_OFFSETS" },
 { 0x5a, "EIATTR_MERCURY_FINALIZER_OPTIONS" },
};

// from sht.txt
static std::map<unsigned int, const char *> s_sht = {
 { 0x70000000, "SHT_CUDA_INFO" },
 { 0x70000001, "SHT_CUDA_CALLGRAPH" },
 { 0x70000002, "SHT_CUDA_PROTOTYPE" },
 { 0x70000003, "SHT_CUDA_RESOLVED_RELA" },
 { 0x70000004, "SHT_CUDA_METADATA" },
 { 0x70000006, "SHT_CUDA_CONSTANT" },
 { 0x70000007, "SHT_CUDA_GLOBAL" },
 { 0x70000008, "SHT_CUDA_GLOBAL_INIT" },
 { 0x70000009, "SHT_CUDA_LOCAL" },
 { 0x7000000A, "SHT_CUDA_SHARED" },
 { 0x7000000B, "SHT_CUDA_RELOCINFO" },
 { 0x7000000E, "SHT_CUDA_UFT" },
 { 0x70000010, "SHT_CUDA_UFT_INDEX" },
 { 0x70000011, "SHT_CUDA_UFT_ENTRY" },
 { 0x70000012, "SHT_CUDA_UDT" },
 { 0x70000014, "SHT_CUDA_UDT_ENTRY" },
 { 0x70000015, "SHT_CUDA_RESERVED_SHARED" },
 { 0x70000064, "SHT_CUDA_CONSTANT_B0" },
 { 0x70000065, "SHT_CUDA_CONSTANT_B1" },
 { 0x70000066, "SHT_CUDA_CONSTANT_B2" },
 { 0x70000067, "SHT_CUDA_CONSTANT_B3" },
 { 0x70000068, "SHT_CUDA_CONSTANT_B4" },
 { 0x70000069, "SHT_CUDA_CONSTANT_B5" },
 { 0x7000006A, "SHT_CUDA_CONSTANT_B6" },
 { 0x7000006B, "SHT_CUDA_CONSTANT_B7" },
 { 0x7000006C, "SHT_CUDA_CONSTANT_B8" },
 { 0x7000006D, "SHT_CUDA_CONSTANT_B9" },
 { 0x7000006E, "SHT_CUDA_CONSTANT_B10" },
 { 0x7000006F, "SHT_CUDA_CONSTANT_B11" },
 { 0x70000070, "SHT_CUDA_CONSTANT_B12" },
 { 0x70000071, "SHT_CUDA_CONSTANT_B13" },
 { 0x70000072, "SHT_CUDA_CONSTANT_B14" },
 { 0x70000073, "SHT_CUDA_CONSTANT_B15" },
 { 0x70000074, "SHT_CUDA_CONSTANT_B16" },
 { 0x70000075, "SHT_CUDA_CONSTANT_B17" },
};

typedef INV_disasm *(*Dproto)(void);

class nv_dis
{
  public:
   nv_dis() {
      n_sec = 0;
      m_out = stdout;
    }
   ~nv_dis() {
     if ( m_dis != nullptr ) delete m_dis;
     if ( m_out && m_out != stdout) fclose(m_out);
   }
   int open(const char *fname) {
     if ( !reader.load(fname) ) {
       fprintf(stderr, "cannot load\n");
       return 0;
     }
     if ( reader.get_machine() != 190 ) {
      fprintf(stderr, "not CUBIN\n");
       return 0;
     }
     // try load smXX
     int sm = (reader.get_flags() >> 0x10) & 0xff;
     if ( !sm ) sm = (reader.get_flags() >> 8) & 0xff;
     auto smi = s_sms.find(sm);
     if ( smi == s_sms.end() ) {
      fprintf(stderr, "unknown SM %X\n", sm);
       return 0;
     }
     std::string sm_name = "./";
     sm_name += smi->second.second ? smi->second.second : smi->second.first;
     sm_name += ".so";
     void *dh = dlopen(sm_name.c_str(), RTLD_NOW);
     if ( !dh ) {
      fprintf(stderr, "cannot load %s, errno %d (%s)\n", sm_name.c_str(), errno, strerror(errno));
       return 0;
     }
     Dproto fn = (Dproto)dlsym(dh, "get_sm");
     if ( !fn ) {
      fprintf(stderr, "cannot find get_sm, errno %d (%s)\n", sm_name.c_str(), errno, strerror(errno));
      dlclose(dh);
       return 0;
     }
     m_dis = fn();
     if ( m_dis ) m_width = m_dis->width();
     return (m_dis != nullptr);
   }
   void process();
   int single_section(int idx);
   void open_log(const char *of) {
     if ( m_out && m_out != stdout ) {
       fclose(m_out);
     }
     m_out = fopen(of, "a");
     if ( !m_out ) {
       fprintf(stderr, "cannot open output file %s, errno %d (%s)\n", of, errno, strerror(errno));
       m_out = stdout;
     }
   }
   void dis_stat() const
   {
     if ( dis_total )
       fprintf(m_out, "total %ld, not_found %ld, dups %ld\n", dis_total, dis_notfound, dis_dups);
   }
  protected:
   typedef std::pair<const struct nv_instr *, NV_extracted> NV_pair;
   typedef std::vector< std::pair<const struct nv_instr *, NV_extracted> > NV_res;
   void try_dis();
   void hdump_section(section *);
   void parse_attrs(Elf_Half idx, section *);
   void dump_ins(const NV_pair &p);
   const nv_eattr *try_by_ename(const struct nv_instr *, const std::string_view &sv) const;
   void dump_ops(const struct nv_instr *, const NV_extracted &);
   int cmp(const std::string_view &, const char *) const;
   bool contain(const std::string_view &, char) const;
   int calc_miss(const struct nv_instr *, const NV_extracted &, int) const;
   int calc_index(const NV_res &, int) const;
   FILE *m_out;
   Elf_Half n_sec;
   elfio reader;
   INV_disasm *m_dis = nullptr;
   int m_width;
   // disasm stat
   long dis_total = 0;
   long dis_notfound = 0;
   long dis_dups = 0;
};

int nv_dis::cmp(const std::string_view &sv, const char *s) const
{
  size_t i = 0;
  for ( auto c = sv.cbegin(); c != sv.cend(); ++c, ++i ) {
    if ( *c != s[i] ) return 0;
  }
  return 1;
}

bool nv_dis::contain(const std::string_view &sv, char sym) const
{
  return sv.find(sym) != std::string::npos;
}

// old MD has encoders like Mask = Enum
// so check in eas
const nv_eattr *nv_dis::try_by_ename(const struct nv_instr *ins, const std::string_view &sv) const
{
  if ( contain(sv, '@') ) return nullptr;
  // check in values
  auto vi = ins->vas.find(sv);
  if ( vi != ins->vas.end() ) return nullptr;
  for ( auto ei: ins->eas ) {
    if ( cmp(sv, ei.second->ename) ) return ei.second;
  }
  return nullptr;
}

int nv_dis::calc_miss(const struct nv_instr *ins, const NV_extracted &kv, int rz) const
{
  int res = 0;
  for ( auto ki: kv ) {
    const nv_eattr *ea = nullptr;
    auto kiter = ins->eas.find(ki.first);
    if ( kiter != ins->eas.end() ) { ea = kiter->second; }
    else { ea = try_by_ename(ins, ki.first); }
    if ( !ea ) continue;
    if ( cmp(ki.first, "NonZeroRegister") && ki.second == rz ) {
      res++; continue;
    }
    if ( cmp(ki.first, "NonZeroUniformRegister") && ki.second == rz ) {
      res++; continue;
    }
    // check in enum
    auto ei = ea->em->find(ki.second);
    if ( ei == ea->em->end() ) res++;
  }
  return res;
}

int nv_dis::calc_index(const NV_res &res, int rz) const
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
  if ( mult ) {
    for ( size_t i = 0; i < res.size(); ++i ) fprintf(m_out, " %d", missed[i]);
    return -1;
  }
  return res_idx;
}

void nv_dis::dump_ops(const struct nv_instr *i, const NV_extracted &kv)
{
  for ( auto kv1: kv )
  {
    std::string name(kv1.first.begin(), kv1.first.end());
    // check in values
    auto vi = i->vas.find(kv1.first);
    if ( vi != i->vas.end() ) {
      fprintf(m_out, " V %s: %X type %d\n", name.c_str(), kv1.second, vi->second.kind);
      continue;
    }
    // check in enums
    auto ei = i->eas.find(kv1.first);
    if ( ei != i->eas.end() ) {
      fprintf(m_out, " E %s: %s %X", name.c_str(), ei->second->ename, kv1.second);
      auto eid = ei->second->em->find(kv1.second);
      if ( eid != ei->second->em->end() )
        fprintf(m_out, " %s\n", eid->second);
      else
        fprintf(m_out," UNKNOWN_ENUM %X\n", kv1.second);
      continue;
    }
    if ( name.find('@') != std::string::npos ) {
      fprintf(m_out, " @ %s: %X\n", name.c_str(), kv1.second);
      continue;
    }
    fprintf(m_out, " U %s: %X\n", name.c_str(), kv1.second);
  }
}

static const char *s_brts[4] = {
 "BRT_CALL",
 "BRT_RETURN",
 "BRT_BRANCH",
 "BRT_BRANCHOUT"
};

void nv_dis::dump_ins(const NV_pair &p)
{
  fprintf(m_out, "%s line %d n %d", p.first->name, p.first->line, p.first->n);
  if ( p.first->brt ) fprintf(m_out, " %s", s_brts[p.first->brt-1]);
  if ( p.first->alt ) fprintf(m_out, " ALT");
  fprintf(m_out, "\n");
  if ( opt_O ) dump_ops( p.first, p.second );
}

void nv_dis::try_dis()
{
  while(1) {
    NV_res res;
    int get_res = m_dis->get(res);
    if ( -1 == get_res ) { fprintf(m_out, "stop at %X\n", m_dis->offset()); break; }
    dis_total++;
    if ( !get_res ) {
      dis_notfound++;
      fprintf(m_out, "Not found at %X", m_dis->offset());
      if ( opt_N ) {
        std::string bstr;
        if ( m_dis->gen_mask(bstr) )
          fprintf(m_out, " %s", bstr.c_str());
      }
      fprintf(m_out, "\n");
      continue;
    }
    int res_idx = 0;
    if ( res.size() > 1 ) res_idx = calc_index(res, m_dis->rz);
    fprintf(m_out, "/* res %d %X ", res.size(), m_dis->offset());
    if ( m_width == 64 ) {
      unsigned char op = 0, ctrl = 0;
      m_dis->get_ctrl(op, ctrl);
      fprintf(m_out, "op %2.2X ctrl %2.2X ", op, ctrl);
    } else if ( m_width == 88 ) {
      unsigned char op = 0, ctrl = 0;
      m_dis->get_ctrl(op, ctrl);
      fprintf(m_out, "ctrl %2.2X ", ctrl);
    }
    if ( res_idx == -1 ) fprintf(m_out, " DUPS ");
    fprintf(m_out, "*/\n");
    if ( res_idx == -1 ) {
      // dump ins
      dis_dups++;
      for ( auto &p: res ) dump_ins(p);
    } else {
      // dump single
      dump_ins(res[res_idx]);
    }
  }
}

void nv_dis::hdump_section(section *sec)
{
  if ( !opt_h ) return;
  if ( sec->get_type() == SHT_NOBITS ) return;
  if ( !sec->get_size() ) return;
  HexDump(m_out, (const unsigned char *)sec->get_data(), (int)sec->get_size());
}

void nv_dis::parse_attrs(Elf_Half idx, section *sec)
{
  if ( !opt_e ) return;
  if ( sec->get_type() == SHT_NOBITS ) return;
  auto size = sec->get_size();
  if ( !size ) return;
  const char *data = sec->get_data();
  const char *start, *end = data + size;
  start = data;
  while( data < end )
  {
    if ( end - data < 2 ) {
      fprintf(stderr, "bad attrs data. section %d\n", idx);
      return;
    }
    char format = data[0];
    char attr = data[1];
    unsigned short a_len;
    auto a_i = s_ei.find(attr);
    if ( a_i != s_ei.end() )
      fprintf(m_out, "%X: %s", data - start, a_i->second);
    else
      fprintf(m_out, "%X: UNKNOWN ATTR %X", data - start, attr);
    switch (format)
    {
      case 2:
        fprintf(m_out, " %2.2X\n", data[2]);
        data += 3;
        // check align
        if ( (data - start) & 0x1 ) data++;
       break;
      case 3:
        fprintf(m_out, " %4.4X\n", *(unsigned short *)(data + 2));
        data += 4;
       break;
      case 4:
        a_len = *(unsigned short *)(data + 2);
        fprintf(m_out, " len %4.4X\n", a_len);
        if ( data + 4 + a_len <= end && opt_h )
          HexDump(m_out, (const unsigned char *)(data + 4), a_len);
        data += 4 + a_len;
        break;
      default: fprintf(stderr, "unknown format %d, section %d off %X (%s)\n",
        format, idx, data - start, sec->get_name().c_str());
         return;
    }
  }
}

int nv_dis::single_section(int idx)
{
  n_sec = reader.sections.size();
  if ( idx < 0 || idx >= n_sec ) return -1;
  section *sec = reader.sections[idx];
  if ( sec->get_type() == SHT_NOBITS ) return 0;
  if ( !sec->get_size() ) return 0;
  m_dis->init( (const unsigned char *)sec->get_data(), sec->get_size() );
  try_dis();
  return 1;
}

void nv_dis::process()
{
  n_sec = reader.sections.size();
  if ( !n_sec ) {
    fprintf(stderr, "no sections\n");
  }
  auto et = reader.get_type();
  fprintf(m_out, "type %X, %d sections\n", et, n_sec);
  // enum sections
  for ( Elf_Half i = 0; i < n_sec; ++i )
  {
    section *sec = reader.sections[i];
    auto st = sec->get_type();
    auto sf = sec->get_flags();
    auto st_i = s_sht.find(st);
    auto sname = sec->get_name();
    if ( st_i != s_sht.end() ) {
      fprintf(m_out, "[%d] %s type %X [%s] flags %X\n", i, sname.c_str(), st, st_i->second, sf);
      if ( st == 0x70000000 ) parse_attrs(i, sec);
      else if ( st > 0x70000000 ) hdump_section(sec);
    } else {
      if ( st == SHT_NOTE )
      {
        auto tf = sf & 0xFF00000;
        if ( tf == 0x1000000 ) {
          fprintf(m_out, "[%d] %s type %X [NOTE] flags %X SHF_NOTE_NV_CUVER\n", i, sec->get_name().c_str(), st, sf);
          hdump_section(sec);
        } else if ( tf == 0x2000000 ) {
          fprintf(m_out, "[%d] %s type %X [NOTE] flags %X SHF_NOTE_NV_TKINFO\n", i, sec->get_name().c_str(), st, sf);
          hdump_section(sec);
        } else
          fprintf(m_out, "[%d] %s type %X [NOTE] flags %X SHF_NOTE_NV_UNKNOWN\n", i, sec->get_name().c_str(), st, sf);
      } else {
       fprintf(m_out, "[%d] %s type %X flags %X\n", i, sec->get_name().c_str(), st, sf);
       if ( st == 0x70000083 ) parse_attrs(i, sec);
       else if ( st > 0x70000000 ) hdump_section(sec);
      }
    }
    if ( st == SHT_NOBITS ) continue;
    if ( !sec->get_size() ) continue;
    // dump addr, size & info
    fprintf(m_out, "  addr %X size %X info %d link %d\n", sec->get_address(), sec->get_size(), sec->get_info(), sec->get_link());
    if ( !strncmp(sname.c_str(), ".text.", 6) )
    {
      m_dis->init( (const unsigned char *)sec->get_data(), sec->get_size() );
      try_dis();
    }
  }
}

void usage(const char *prog)
{
  printf("%s usage: [options] cubin(s)]n");
  printf("Options:\n");
  printf("-e - dump attributes\n");
  printf("-h - hex dump\n");
  printf("-N - dump not found masks\n");
  printf("-o - output file\n");
  printf("-O - dump operands\n");
  printf("-s index - disasm only single section withh index\n");
  printf("-t - dump symbols\n");
  exit(6);
}

int main(int argc, char **argv)
{
  int c;
  int s = -1;
  const char *o_fname = nullptr;
  while(1) {
    c = getopt(argc, argv, "ehtNOs:o:");
    if ( c == -1 ) break;
    switch(c) {
      case 'e': opt_e = 1; break;
      case 'h': opt_h = 1; break;
      case 't': opt_t = 1; break;
      case 'O': opt_O = 1; break;
      case 'N': opt_N = 1; break;
      case 'o': o_fname = optarg; break;
      case 's': s = atoi(optarg); break;
      default: usage(argv[0]);
    }
  }
  if ( argc == optind ) {
    usage(argv[0]);
    return 6;
  }
  for ( int i = optind; i < argc; i++ )
  {
    printf("%s:\n", argv[i]);
    nv_dis dis;
    if ( o_fname ) dis.open_log(o_fname);
    if ( dis.open(argv[i]) )
    {
      if ( s != -1 )
        dis.single_section(s);
      else
        dis.process();
    }
    dis.dis_stat();
  }
}