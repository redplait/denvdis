#include "elfio/elfio.hpp"
#include "nv_rend.h"
#include <unistd.h>

int opt_e = 0,
    opt_h = 0,
    opt_m = 0,
    opt_t = 0,
    opt_p = 0,
    opt_r = 0,
    opt_S = 0,
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

static const char* get_merc_reloc_name(unsigned t)
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

static const char* get_cuda_reloc_name(unsigned t)
{
  if ( t >= sizeof(cuda_relocs) / sizeof(cuda_relocs[0]) ) return nullptr;
  return cuda_relocs[t];
}

// extracted from EIATTR_INDIRECT_BRANCH_TARGETS
struct bt_per_section
{
  std::unordered_map<uint32_t, uint32_t> branches; // key - offset, value - target
  NV_labels labels; // offset of label
};

class nv_dis: public NV_renderer
{
  public:
   nv_dis(): NV_renderer() {
      n_sec = 0;
    }
   ~nv_dis() {
     for ( auto bi: m_branches ) delete bi.second;
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
     printf("load %s\n", sm_name.c_str());
     return NV_renderer::load(sm_name);
   }
   void process();
   int single_section(int idx);
   void dump_total() const
   {
    dis_stat();
    if ( opt_S )
      fprintf(m_out, "filters %ld success %ld, conditions %ld (%ld) cached %ld\n",
        sfilters, sfilters_succ, scond_count, scond_succ, scond_hits);
   }
  protected:
   void fill_eaddrs(NV_labels *, int ltype, const char *, int alen);
   void try_dis(Elf_Word idx);
   void dump_ins(const NV_pair &p, uint32_t, NV_labels *);
   void hdump_section(section *);
   void dump_mrelocs(section *);
   void dump_crelocs(section *);
   void parse_attrs(Elf_Half idx, section *);
   bt_per_section *get_branch(Elf_Word i) {
     auto bi = m_branches.find(i);
     if ( bi != m_branches.end() ) return bi->second;
     auto res = new bt_per_section();
     m_branches[i] = res;
     return res;
   }

   Elf_Half n_sec;
   elfio reader;
   // indirect branches
   std::unordered_map<Elf_Word, bt_per_section *> m_branches;
};

static const char *s_brts[4] = {
 "BRT_CALL",
 "BRT_RETURN",
 "BRT_BRANCH",
 "BRT_BRANCHOUT"
};

void nv_dis::dump_ins(const NV_pair &p, uint32_t label, NV_labels *l)
{
  m_missed.clear();
  fprintf(m_out, "; %s line %d n %d", p.first->name, p.first->line, p.first->n);
  if ( p.first->brt ) fprintf(m_out, " %s", s_brts[p.first->brt-1]);
  if ( p.first->alt ) fprintf(m_out, " ALT");
  auto rend = m_dis->get_rend(p.first->n);
  if ( rend ) {
    fprintf(m_out, " %ld render items", rend->size());
    std::string r;
    int miss = render(rend, r, p.first, p.second, l);
    if ( miss ) {
      fprintf(m_out, " %d missed", miss);
      if ( opt_m ) {
        fputc(':', m_out);
        for ( auto &ms: m_missed ) fprintf(m_out, " %s", ms.c_str());
      }
    }
    fputc('\n', m_out);
    // body of instruction
    fputc('>', m_out);
    if ( dual_first ) fputs(" {", m_out);
    else if ( dual_last ) fputs("  ", m_out);
    if ( label )
     fprintf(m_out, " %s (* BRANCH_TARGET LABEL_%X *)", r.c_str(), label);
    else
      fprintf(m_out, " %s", r.c_str());
    if ( dual_last ) fputs(" }", m_out);
    fputc('\n', m_out);
  } else
    fprintf(m_out, " NO_Render\n");
  if ( opt_O ) dump_ops( p.first, p.second );
  if ( opt_p ) dump_predicates( p.first, p.second );
}

void nv_dis::try_dis(Elf_Word idx)
{
  auto branches = get_branch(idx);
  dual_first = dual_last = false;
  while(1) {
    NV_res res;
    int get_res = m_dis->get(res);
    if ( -1 == get_res ) { fprintf(m_out, "stop at %lX\n", m_dis->offset()); break; }
    dis_total++;
    if ( !get_res ) {
      dis_notfound++;
      fprintf(m_out, "Not found at %lX", m_dis->offset());
      if ( opt_N ) {
        std::string bstr;
        if ( m_dis->gen_mask(bstr) )
          fprintf(m_out, " %s", bstr.c_str());
      }
      fprintf(m_out, "\n");
      dual_first = dual_last = false;
      m_sched.clear();
      continue;
    }
    int res_idx = 0;
    if ( res.size() > 1 ) res_idx = calc_index(res, m_dis->rz);
    // check branch label
    auto off = m_dis->offset();
    uint32_t curr_label = 0;
    if ( branches ) {
      auto li = branches->labels.find(off);
      if ( li != branches->labels.end() ) {
        if ( li->second )
          fprintf(m_out, "LABEL_%lX: ; %s\n", off, s_ltypes[li->second]);
        else
          fprintf(m_out, "LABEL_%lX:\n", off);
      }
      auto bi = branches->branches.find(off);
      if ( bi != branches->branches.end() )
        curr_label = bi->second;
    }
    fprintf(m_out, "/* res %ld %lX ", res.size(), off);
    if ( m_width == 64 ) {
      unsigned char op = 0, ctrl = 0;
      m_dis->get_ctrl(op, ctrl);
      fprintf(m_out, "op %2.2X ctrl %2.2X ", op, ctrl);
    } else if ( m_width == 88 ) {
      auto cword = m_dis->get_cword();
      fprintf(m_out, "ctrl %lX ", cword);
    }
    if ( res_idx == -1 ) fprintf(m_out, " DUPS ");
    fprintf(m_out, "*/\n");
    if ( res_idx == -1 ) {
      // dump multiple ins
      dis_dups++;
      dual_first = dual_last = false;
      for ( auto &p: res ) dump_ins(p, 0, nullptr);
    } else {
      if ( opt_S && !m_sched.empty() )
        dump_sched(res[res_idx].first, res[res_idx].second);
      // dump single
      if ( m_width == 88 && !dual_first && !dual_last )
        dual_first = check_dual(res[res_idx].second);
      dump_ins(res[res_idx], curr_label, branches ? &branches->labels: nullptr);
      if ( opt_S && res[res_idx].first->scbd_type != BB_ENDING_INST && !res[res_idx].first->brt ) // store sched rows of current instruction
        fill_sched(res[res_idx].first, res[res_idx].second);
      else
        m_sched.clear();
    }
    if ( dual_first ) {
      dual_first = false;
      dual_last = true;
    } else if ( dual_last )
      dual_last = false;
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
    int ltype = 0;
    if ( a_i != s_ei.end() )
      fprintf(m_out, "%lX: %s", data - start, a_i->second);
    else
      fprintf(m_out, "%lX: UNKNOWN ATTR %X", data - start, attr);
    switch (format)
    {
      case 1: data += 2;
        // check align
        if ( (data - start) & 0x3 ) data += 4 - ((data - start) & 0x3);
        fputc('\n', m_out);
        break;
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
        if ( attr == 0x28 ) // EIATTR_COOP_GROUP_INSTR_OFFSETS
          ltype = NVLType::Coop_grp;
        else if ( attr == 0x1c ) // EIATTR_COOP_GROUP_INSTR_OFFSETS
          ltype = NVLType::Exit;
        else if ( attr == 0x1d ) // EIATTR_COOP_GROUP_INSTR_OFFSETS
          ltype = NVLType::S2Rctaid;
        else if ( attr == 0x31 ) // EIATTR_INT_WARP_WIDE_INSTR_OFFSETS
          ltype = NVLType::Warp_wide;
        // read offsets
        if ( ltype ) {
          auto ib = get_branch(sec->get_info());
          fill_eaddrs(&ib->labels, ltype, data, a_len);
        } else if ( attr == 0x34 ) {
          // collect indirect branches
          auto ib = get_branch(sec->get_info());
          for ( const char *bcurr = data + 4; data + 4 + a_len - bcurr >= 0x10; bcurr += 0x10 ) {
            // offset 0 - address of instruction
            // offset c - address of label
            uint32_t addr = *(uint32_t *)(bcurr),
              lab = *(uint32_t *)(bcurr + 0xc);
 // fprintf(m_out, "addr %lX label %X\n", addr, lab);
            ib->labels[lab] = 0;
            ib->branches[addr] = lab;
          }
        }
        data += 4 + a_len;
        break;
      default: fprintf(stderr, "unknown format %d, section %d off %lX (%s)\n",
        format, idx, data - start, sec->get_name().c_str());
         return;
    }
  }
}

void nv_dis::fill_eaddrs(NV_labels *l, int ltype, const char *data, int alen)
{
  for ( const char *bcurr = data + 4; data + 4 + alen - bcurr >= 0x4; bcurr += 0x4 )
  {
    uint32_t addr = *(uint32_t *)(bcurr);
    (*l)[addr] = ltype;
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
  try_dis(idx);
  return 1;
}

void nv_dis::dump_mrelocs(section *sec)
{
  const_relocation_section_accessor rsa(reader, sec);
  auto n = rsa.get_entries_num();
  fprintf(m_out, "%ld relocs:\n", n);
  if ( !n ) return;
  auto old_type = sec->get_type();
  sec->set_type(SHT_RELA);
  for ( Elf_Xword i = 0; i < n; i++ ) {
    Elf64_Addr addr;
    Elf_Word sym;
    unsigned type;
    Elf_Sxword add;
    if ( rsa.get_entry(i, addr, sym, type, add) ) {
      auto tname = get_merc_reloc_name(type);
      fprintf(m_out, " [%ld] %lX sym %d add %lX", n, addr, sym, add);
      if ( tname )
       fprintf(m_out, " %s\n", tname);
      else
       fprintf(m_out, " type %d\n", type);
    }
  }
  sec->set_type(old_type);
}

void nv_dis::dump_crelocs(section *sec)
{
  if ( !sec->get_size() ) return;
  const_relocation_section_accessor rsa(reader, sec);
  auto n = rsa.get_entries_num();
  fprintf(m_out, "%ld relocs:\n", n);
  if ( !n ) return;
  for ( Elf_Xword i = 0; i < n; i++ ) {
    Elf64_Addr addr;
    Elf_Word sym;
    unsigned type;
    Elf_Sxword add;
    if ( rsa.get_entry(i, addr, sym, type, add) ) {
      auto tname = get_cuda_reloc_name(type);
      fprintf(m_out, " [%ld] %lX sym %d add %lX", n, addr, sym, add);
      if ( tname )
       fprintf(m_out, " %s\n", tname);
      else
       fprintf(m_out, " type %d\n", type);
    }
  }
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
      fprintf(m_out, "[%d] %s type %X [%s] flags %lX\n", i, sname.c_str(), st, st_i->second, sf);
      if ( st == 0x70000000 ) parse_attrs(i, sec);
      else if ( st > 0x70000000 ) hdump_section(sec);
    } else {
      if ( st == SHT_REL || st == SHT_RELA ) {
        fprintf(m_out, "[%d] %s type %X flags %lX\n", i, sec->get_name().c_str(), st, sf);
        if ( opt_r ) dump_crelocs(sec);
      } else if ( st == SHT_NOTE )
      {
        auto tf = sf & 0xFF00000;
        if ( tf == 0x1000000 ) {
          fprintf(m_out, "[%d] %s type %X [NOTE] flags %lX SHF_NOTE_NV_CUVER\n", i, sec->get_name().c_str(), st, sf);
          hdump_section(sec);
        } else if ( tf == 0x2000000 ) {
          fprintf(m_out, "[%d] %s type %X [NOTE] flags %lX SHF_NOTE_NV_TKINFO\n", i, sec->get_name().c_str(), st, sf);
          hdump_section(sec);
        } else
          fprintf(m_out, "[%d] %s type %X [NOTE] flags %lX SHF_NOTE_NV_UNKNOWN\n", i, sec->get_name().c_str(), st, sf);
      } else {
       fprintf(m_out, "[%d] %s type %X flags %lX\n", i, sec->get_name().c_str(), st, sf);
       if ( st == 0x70000083 ) parse_attrs(i, sec);
       else if ( st == 0x70000082 && opt_r ) dump_mrelocs(sec);
       else if ( st > 0x70000000 ) hdump_section(sec);
      }
    }
    if ( st == SHT_NOBITS ) continue;
    if ( !sec->get_size() ) continue;
    // dump addr, size & info
    fprintf(m_out, "  addr %lX size %lX info %d link %d\n", sec->get_address(), sec->get_size(), sec->get_info(), sec->get_link());
    if ( !strncmp(sname.c_str(), ".text.", 6) )
    {
      m_dis->init( (const unsigned char *)sec->get_data(), sec->get_size() );
      try_dis(i);
    }
  }
}

void usage(const char *prog)
{
  printf("%s usage: [options] cubin(s)]n", prog);
  printf("Options:\n");
  printf("-e - dump attributes\n");
  printf("-h - hex dump\n");
  printf("-m - dump missed fields\n");
  printf("-N - dump not found masks\n");
  printf("-o - output file\n");
  printf("-O - dump operands\n");
  printf("-p - dump predicates\n");
  printf("-r - dump relocs\n");
  printf("-s index - disasm only single section withh index\n");
  printf("-S - dump sched info\n");
  printf("-t - dump symbols\n");
  exit(6);
}

int main(int argc, char **argv)
{
  int c;
  int s = -1;
  const char *o_fname = nullptr;
  while(1) {
    c = getopt(argc, argv, "ehmrtNOpSs:o:");
    if ( c == -1 ) break;
    switch(c) {
      case 'e': opt_e = 1; break;
      case 'h': opt_h = 1; break;
      case 'm': opt_m = 1; break;
      case 't': opt_t = 1; break;
      case 'O': opt_O = 1; break;
      case 'p': opt_p = 1; break;
      case 'r': opt_r = 1; break;
      case 'N': opt_N = 1; break;
      case 'S': opt_S = 1; break;
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
    dis.dump_total();
  }
}