#include "elfio/elfio.hpp"
#include "nv_rend.h"
#include <unistd.h>

int opt_c = 0,
    opt_e = 0,
    opt_h = 0,
    opt_m = 0,
    opt_t = 0,
    opt_T = 0,
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
// for sv literals
using namespace std::string_view_literals;

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

struct asymbol
{
  std::string name;
  Elf64_Addr addr;
  Elf_Xword idx = 0;
  Elf_Xword size = 0;
  Elf_Half section;
  unsigned char bind = 0,
                type = 0,
                other = 0;
};

struct reg_history {
  unsigned long off;
  // 0x8000 - write, else read
  // 0x4000 - Uniform predicate, else just predicate
  // next 3 bits are predicate reg index + 1 (bcs T == 7 and 0 is perfectly valid predicate)
  typedef unsigned short RH;
  RH kind;
  inline bool is_upred() const {
    return kind & 0x4000;
  }
  inline bool has_pred(int &p) const {
    p = (kind >> 11) & 0x7;
    if ( p ) {
      p--;
      return true;
    }
    return false;
  }
};

// register tracks
// there can be 4 groups of register
// - general purpose registers
// - predicate registers
// and since sm75 also
// - uniform gpr
// - uniform predicates
// keys are register index
struct reg_pad {
  typedef std::unordered_map<int, std::vector<reg_history> > RSet;
  RSet gpr, pred, ugpr, upred;
  reg_history::RH pred_mask = 0;
  // boring stuff
  void _add(RSet &rs, int idx, unsigned long off, reg_history::RH k) {
    k |= pred_mask;
    auto ri = rs.find(idx);
    if ( ri != rs.end() ) {
      if ( !ri->second.empty() ) { // check if prev item is the same
        auto &last = ri->second.back();
        if ( last.off == off && last.kind == k ) return;
      }
      ri->second.push_back( { off, k } );
    } else {
     std::vector<reg_history> tmp;
     tmp.push_back( { off, k } );
     rs[idx] = std::move(tmp);
    }
  }
  void rgpr(int r, unsigned long off, reg_history::RH k) {
    _add(gpr, r, off, k);
  }
  void wgpr(int r, unsigned long off, reg_history::RH k) {
    _add(gpr, r, off, k | 0x8000);
  }
  void rugpr(int r, unsigned long off, reg_history::RH k) {
    _add(ugpr, r, off, k);
  }
  void wugpr(int r, unsigned long off, reg_history::RH k) {
    _add(ugpr, r, off, k | 0x8000);
  }
  void rpred(int r, unsigned long off, reg_history::RH k) {
    _add(pred, r, off, k);
  }
  void wpred(int r, unsigned long off, reg_history::RH k) {
    _add(pred, r, off, k | 0x8000);
  }
  void rupred(int r, unsigned long off, reg_history::RH k) {
    _add(upred, r, off, k);
  }
  void wupred(int r, unsigned long off, reg_history::RH k) {
    _add(upred, r, off, k | 0x8000);
  }
  void clear() {
     pred_mask = 0;
     gpr.clear();
     pred.clear();
     ugpr.clear();
     upred.clear();
  }
};

// extracted from EIATTR_INDIRECT_BRANCH_TARGETS
struct bt_per_section
{
  std::unordered_map<uint32_t, uint32_t> branches; // key - offset, value - target
  NV_labels labels; // offset of label
};

// const bank params
struct cb_param {
  int ordinal;
  uint32_t size;
  unsigned short offset;
};

// const banks per section
struct cbank_per_section {
  std::vector<cb_param> params;
  Elf_Word section; // from EIATTR_PARAM_CBANK
  unsigned short size = 0;
  unsigned short offset = 0;
  inline bool in_cb(unsigned short off) const {
    return off >= offset && off < (offset + size);
  }
  const cb_param *find_param(unsigned short off) const {
    auto pi = std::lower_bound(params.cbegin(), params.cend(), off, [](const cb_param &cb, unsigned short off) {
      return cb.offset < off;
    });
    if ( pi == params.cend() ) return nullptr;
    return &(*pi);
  }
};

class nv_dis: public NV_renderer
{
  public:
   nv_dis(): NV_renderer() {
      n_sec = 0;
    }
   ~nv_dis() {
     for ( auto bi: m_branches ) delete bi.second;
     for ( auto cb: m_cbanks ) delete cb.second;
     if ( m_rtdb ) delete m_rtdb;
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
     // check SM_DIR env
     std::string sm_name;
     char *sm_dir = getenv("SM_DIR");
     if ( sm_dir ) {
      sm_name = sm_dir;
      if ( !sm_name.ends_with("/") ) sm_name += '/';
     } else {
      sm_name = "./";
     }
     sm_name += smi->second.second ? smi->second.second : smi->second.first;
     sm_name += ".so";
     if ( opt_c ) printf(".target sm_%d\n", sm);
     else printf("load %s\n", sm_name.c_str());
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
   // relocs. key - offset
   typedef std::map<unsigned long, NV_rel> SRels;
   // key - section index
   std::unordered_map<int, SRels> m_srels;
   mutable SRels::const_iterator riter;
   SRels::const_iterator riter_end;
   virtual const std::string *try_name(unsigned long off) const override {
     auto si = m_curr_syms.find(off);
     if ( si == m_curr_syms.end() ) return nullptr;
     return &si->second->name;
   }
   virtual const NV_rel *next_reloc(std::string_view &sv) const override {
     if ( !has_relocs ) return nullptr;
     const NV_rel *res = &riter->second;
     auto &sym = m_syms[res->second];
     sv = { sym.name.cbegin(), sym.name.cend() };
     if ( ++riter == riter_end )
       has_relocs = false;
     else {
       m_next_roff = riter->first;
#ifdef DEBUG
  fprintf(m_out, "next_roff %lX\n", m_next_roff);
#endif
     }
     return res;
   }
   int fill_rels();
   void fill_eaddrs(NV_labels *, int ltype, const char *, int alen);
   void try_dis(Elf_Word idx);
   void dump_ins(const NV_pair &p, uint32_t, NV_labels *);
   // boring ELF related stuff
   void hdump_section(section *);
   void dump_mrelocs(section *);
   void dump_crelocs(section *);
   void parse_attrs(Elf_Half idx, section *);
   void _parse_attrs(Elf_Half idx, section *);
   int read_symbols();
   // branches
   bt_per_section *get_branch(Elf_Word i) {
     auto bi = m_branches.find(i);
     if ( bi != m_branches.end() ) return bi->second;
     auto res = new bt_per_section();
     m_branches[i] = res;
     return res;
   }

   Elf_Half n_sec;
   elfio reader;
   // symbols
   std::vector<asymbol> m_syms;
   std::map<unsigned long, asymbol *> m_curr_syms;
   std::map<unsigned long, asymbol *>::const_iterator m_curr_siter = m_curr_syms.cend();
   int grab_syms_for_section(int);
   int next_csym(unsigned long);
   void dump_csym(const asymbol *) const;
   // indirect branches
   std::unordered_map<Elf_Word, bt_per_section *> m_branches;
   // const banks
   const cbank_per_section *get_cbank(Elf_Word idx) {
     auto cb = m_cbanks.find(idx);
     if ( cb == m_cbanks.end() ) return nullptr;
     return cb->second;
   }
   void add_cbank(Elf_Word, Elf_Word, unsigned short off, unsigned short size);
   void add_cparam(Elf_Word, int ordinal, uint32_t, unsigned short);
   void finalize_cparams(Elf_Word idx) {
     auto cb = m_cbanks.find(idx);
     if ( cb == m_cbanks.end() ) return;
     std::sort( cb->second->params.begin(), cb->second->params.end(), [](const cb_param &a, const cb_param &b) {
       return a.offset < b.offset;
     });
   }
   std::unordered_map<Elf_Word, cbank_per_section *> m_cbanks;
   // regs track db
   int track_regs(const NV_rlist *, const NV_pair &p, unsigned long off);
   void dump_rt() const;
   void finalize_rt();
   void dump_rset(const reg_pad::RSet &, const char *pfx) const;
   inline bool is_pred(const nv_eattr *ea, NV_extracted::const_iterator &kvi) const {
     return !strcmp(ea->ename, "Predicate") && 7 != kvi->second;
   }
   inline bool is_upred(const nv_eattr *ea, NV_extracted::const_iterator &kvi) const {
     return !strcmp(ea->ename, "UniformPredicate") && 7 != kvi->second;
   }
   inline bool is_reg(const nv_eattr *ea, NV_extracted::const_iterator &kvi) const {
     return (!strcmp(ea->ename, "Register") || !strcmp(ea->ename, "NonZeroRegister")) && m_dis->rz != (int)kvi->second;
   }
   inline bool is_ureg(const nv_eattr *ea, NV_extracted::const_iterator &kvi) const {
     return (!strcmp(ea->ename, "UniformRegister") || !strcmp(ea->ename, "NonZeroUniformRegister")) && m_dis->rz != (int)kvi->second;
   }
   reg_pad *m_rtdb = nullptr;
};

void nv_dis::finalize_rt() {
 if ( !m_rtdb ) return;
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
 if ( !m_rtdb->gpr.empty() )
  for ( auto &r: m_rtdb->gpr ) std::sort(r.second.begin(), r.second.end(), srt);
 if ( !m_rtdb->ugpr.empty() )
  for ( auto &r: m_rtdb->ugpr ) std::sort(r.second.begin(), r.second.end(), srt);
 if ( !m_rtdb->pred.empty() )
  for ( auto &r: m_rtdb->pred ) std::sort(r.second.begin(), r.second.end(), srt);
 if ( !m_rtdb->upred.empty() )
  for ( auto &r: m_rtdb->upred ) std::sort(r.second.begin(), r.second.end(), srt);
}

void nv_dis::dump_rt() const {
  if ( !m_rtdb ) return;
  if ( !m_rtdb->gpr.empty() ) {
    fprintf(m_out, ";;; %ld GPR\n", m_rtdb->gpr.size());
    dump_rset(m_rtdb->gpr, "R");
  }
  if ( !m_rtdb->ugpr.empty() ) {
    fprintf(m_out, ";;; %ld UGPR\n", m_rtdb->ugpr.size());
    dump_rset(m_rtdb->ugpr, "UR");
  }
  if ( !m_rtdb->pred.empty() ) {
    fprintf(m_out, ";;; %ld PRED\n", m_rtdb->pred.size());
    dump_rset(m_rtdb->pred, "P");
  }
  if ( !m_rtdb->upred.empty() ) {
    fprintf(m_out, ";;; %ld UPRED\n", m_rtdb->upred.size());
    dump_rset(m_rtdb->upred, "UP");
  }
}

void nv_dis::dump_rset(const reg_pad::RSet &rs, const char *pfx) const
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
          fprintf(m_out, " ;   %lX <- %X P%d\n", tr.off, tr.kind & mask, pred);
        else
          fprintf(m_out, " ;   %lX <- %X\n", tr.off, tr.kind & mask);
      } else {
        if ( is_pred )
          fprintf(m_out, " ;   %lX %X P%d\n", tr.off, tr.kind & mask, pred);
        else
          fprintf(m_out, " ;   %lX %X\n", tr.off, tr.kind & mask);
      }
    }
  }
}

int nv_dis::track_regs(const NV_rlist *rend, const NV_pair &p, unsigned long off)
{
  int res = 0;
  bool has_props = p.first->props != nullptr;
  const std::string_view *d_sv = nullptr,
   *d2_sv = nullptr;
  bool setp = is_setp(p.first);
  if ( has_props ) {
    for ( auto pr: *p.first->props ) {
      if ( pr->op == IDEST && pr->fields.size() == 1 ) d_sv = &get_it(pr->fields, 0);
      if ( pr->op == IDEST2 && pr->fields.size() == 1 ) d2_sv = &get_it(pr->fields, 0);
    }
  }
  // predicates
  int d_size = 0, d2_size = 0, a_size = 0, b_size = 0, c_size = 0;
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
  }
  int idx = -1;
  m_rtdb->pred_mask = 0;
  for ( auto &r: *rend ) {
    // check if we have taul - then end loop
    if ( r->type == R_value ) {
      const render_named *rn = (const render_named *)r;
      auto vi = find(p.first->vas, rn->name);
      if ( vi && vi->kind == NV_BITSET && !strncmp(rn->name, "req_", 4) ) break;
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
       { m_rtdb->pred_mask = (1 + (unsigned short)kvi->second) << 11;
         m_rtdb->rpred(kvi->second, off, 0); res++; }
      else if ( !strcmp(ea->ename, "UniformPredicate") )
       { m_rtdb->pred_mask = 0x4000 | (1 + (unsigned short)kvi->second) << 11;
         m_rtdb->rupred(kvi->second, off, 0); res++; }
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
      if ( kvi->second == 7 ) continue;
      if ( !strcmp(ea->ename, "Predicate") )
       { m_rtdb->wpred(kvi->second, off, 0); res++; }
      else if ( !strcmp(ea->ename, "UniformPredicate") )
       { m_rtdb->wupred(kvi->second, off, 0); res++; }
      idx++;
      continue;
    }
    if ( r->type == R_opcode ) {
      idx = 0;
      continue;
    }
    auto rgpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = i;
        if ( kvi->second + i >= m_dis->rz ) break;
        m_rtdb->rgpr(kvi->second + i, off, what);
        res++;
      }
      return res;
    };
    auto gpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = i;
        if ( kvi->second + i >= m_dis->rz ) break;
        m_rtdb->wgpr(kvi->second + i, off, what);
        res++;
      }
      return res;
    };
    auto rugpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = i;
        if ( kvi->second + i >= m_dis->rz ) break;
        m_rtdb->rugpr(kvi->second + i, off, what);
        res++;
      }
      return res;
    };
    auto ugpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = i;
        if ( kvi->second + i >= m_dis->rz ) break;
        m_rtdb->wugpr(kvi->second + i, off, what);
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
       { m_rtdb->rpred(kvi->second, off, 0); res++; }
      else if ( is_upred(ea, kvi) )
       { m_rtdb->rupred(kvi->second, off, 0); res++; }
      else if ( is_reg(ea, kvi) )
      {
        if ( d_sv && cmp(*d_sv, rn->name) ) {
         if ( d_size <= 32 )
          { m_rtdb->wgpr(kvi->second, off, 0); res++; }
         else res += gpr_multi(d_size, kvi);
        } else if ( d2_sv && cmp(*d2_sv, rn->name) ) {
         if ( d2_size <= 32 )
         { m_rtdb->wgpr(kvi->second, off, 0); res++; }
         else res += gpr_multi(d2_size, kvi);
        } else if ( !strcmp(rn->name, "Rd") ) {
         if ( d_size <= 32 )
          { m_rtdb->wgpr(kvi->second, off, 0); res++; }
         else res += gpr_multi(d_size, kvi);
        } else if ( !strcmp(rn->name, "Rd2") ) {
         if ( d2_size <= 32 )
          { m_rtdb->wgpr(kvi->second, off, 0); res++; }
         else res += gpr_multi(d2_size, kvi);
        } else {
         if ( !strcmp(rn->name, "Ra") && a_size > 32 )
          res += rgpr_multi(a_size, kvi);
         else if ( !strcmp(rn->name, "Rb") && b_size > 32 )
          res += rgpr_multi(b_size, kvi);
         else if ( !strcmp(rn->name, "Rc") && c_size > 32 )
          res += rgpr_multi(c_size, kvi);
         else
         { m_rtdb->rgpr(kvi->second, off, 0); res++; }
        }
      } else if ( is_ureg(ea, kvi) )
      {
        if ( d_sv && cmp(*d_sv, rn->name) ) {
         if ( d_size <= 32 )
          { m_rtdb->wugpr(kvi->second, off, 0); res++; }
          else res += ugpr_multi(d_size, kvi);
        } else if ( d2_sv && cmp(*d2_sv, rn->name) ) {
         if ( d2_size <= 32 )
           { m_rtdb->wugpr(kvi->second, off, 0); res++; }
         else res += ugpr_multi(d2_size, kvi);
        } else if ( !strcmp(rn->name, "URd") ) {
         if ( d_size <= 32 )
          { m_rtdb->wugpr(kvi->second, off, 0); res++; }
         else res += ugpr_multi(d_size, kvi);
        } else if ( !strcmp(rn->name, "URd2") ) {
         if ( d2_size <= 32 )
          { m_rtdb->wugpr(kvi->second, off, 0); res++; }
         else res += ugpr_multi(d2_size, kvi);
        } else {
         if ( !strcmp(rn->name, "URa") && a_size > 32 )
          res += rugpr_multi(a_size, kvi);
         else if ( !strcmp(rn->name, "URb") && b_size > 32 )
          res += rugpr_multi(b_size, kvi);
         else if ( !strcmp(rn->name, "URc") && c_size > 32 )
          res += rugpr_multi(c_size, kvi);
         else
         { m_rtdb->rugpr(kvi->second, off, 0); res++; }
        }
      }
    }
    // ok, we have something compound
    auto check_ve = [&](const ve_base &ve, reg_history::RH what) {
        if ( ve.type == R_value ) return 0;
        const nv_eattr *ea = find_ea(p.first, ve.arg);
        if ( !ea ) return 0;
        auto kvi = p.second.find(ve.arg);
        if ( kvi == p.second.end() ) return 0;
        // check what we have
        if ( is_pred(ea, kvi) )
        { m_rtdb->rpred(kvi->second, off, what); return 1; }
        if ( is_upred(ea, kvi) )
        { m_rtdb->rupred(kvi->second, off, what); return 1; }
        if ( is_reg(ea, kvi) )
        { m_rtdb->rgpr(kvi->second, off, what); return 1; }
        if ( is_ureg(ea, kvi) )
        { m_rtdb->rugpr(kvi->second, off, what); return 1; }
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

void nv_dis::add_cparam(Elf_Word idx, int ordinal, uint32_t size, unsigned short off)
{
  cbank_per_section *cps = nullptr;
  auto cb = m_cbanks.find(idx);
  if ( cb == m_cbanks.end() ) {
    cps = new cbank_per_section;
    m_cbanks[idx] = cps;
  } else
    cps = cb->second;
  cps->params.push_back( { ordinal, size, off } );
}

void nv_dis::add_cbank(Elf_Word idx, Elf_Word s, unsigned short off, unsigned short size)
{
  auto cb = m_cbanks.find(idx);
  if ( cb != m_cbanks.end() ) {
    cb->second->section = s;
    cb->second->offset = off;
    cb->second->size = size;
  } else {
    cbank_per_section *cps = new cbank_per_section;
    cps->section = s;
    cps->offset = off;
    cps->size = size;
    m_cbanks[idx] = cps;
  }
}

static const char *s_brts[4] = {
 "BRT_CALL",
 "BRT_RETURN",
 "BRT_BRANCH",
 "BRT_BRANCHOUT"
};

int nv_dis::read_symbols()
{
  section *sym_sec = nullptr;
  for ( Elf_Half i = 0; i < n_sec; ++i )
  {
    section* sec = reader.sections[i];
    if ( sec->get_type() == SHT_SYMTAB ) { sym_sec = sec; break; }
  }
  if ( !sym_sec ) return 0;
  // read symtab
  symbol_section_accessor symbols( reader, sym_sec );
  Elf_Xword sym_no = symbols.get_symbols_num();
  if ( !sym_no )
  {
    fprintf(m_out, "no symbols\n");
    return 0;
  }
  if ( opt_t ) {
    fprintf(m_out, "%ld symbols\n", sym_no);
  }
  for ( Elf_Xword i = 0; i < sym_no; ++i )
  {
    asymbol sym;
    sym.idx = i;
    symbols.get_symbol( i, sym.name, sym.addr, sym.size, sym.bind, sym.type, sym.section, sym.other );
    if ( opt_t ) {
      if ( sym.type != STT_SECTION )
        fprintf(m_out, " [%ld] %lX sec %d type %d %s\n", i, sym.addr, sym.section, sym.type, sym.name.c_str());
    }
    if ( opt_r )
      m_syms.push_back(sym);
  }
  int res = !m_syms.empty();
  if ( !res ) return res;
  if ( opt_r ) return fill_rels();
  next_csym(0);
  return res;
}

int nv_dis::next_csym(unsigned long off)
{
  if ( m_curr_siter == m_curr_syms.cend() ) return 0;
  if ( m_curr_siter->first != off ) return 0;
  dump_csym(m_curr_siter->second);
  ++m_curr_siter;
  return 1;
}

void nv_dis::dump_csym(const asymbol *as) const
{
  if ( as->bind == STB_GLOBAL )
    fprintf(m_out, "\t.global %s\n", as->name.c_str());
  if ( as->type == STT_OBJECT )
    fprintf(m_out, "\t.type %s,@object\n", as->name.c_str());
  else if ( as->type == STT_FUNC )
    fprintf(m_out, "\t.type %s,@function\n", as->name.c_str());
  if ( as->size )
    fprintf(m_out, "\t.size %lX\n", as->size);
  if ( as->other ) {
    fprintf(m_out, "\t.other %s, @\"", as->name.c_str());
    char upE = as->other & 0xE0;
    char lo2 = as->other & 3;
    int idx = 0;
    // as far I understod order is not matters
#define _DA(c) { if ( idx ) fputc(' ', m_out); fprintf(m_out, "%s", c); idx++; }
    if ( as->other & 0x10 ) _DA("STO_CUDA_ENTRY")
    if ( as->other & 0x4 )  _DA("STO_CUDA_MANAGED")
    if ( as->other & 0x8 )  _DA("STO_CUDA_OBSCURE")
    if ( upE == 0x20 ) _DA("STO_CUDA_GLOBAL")
    if ( upE == 0x40 ) _DA("STO_CUDA_SHARED")
    if ( upE == 0x60 ) _DA("STO_CUDA_LOCAL")
    if ( upE == 0x80 ) _DA("STO_CUDA_CONSTANT")
    if ( upE == 0xa0 ) _DA("STO_CUDA_RESERVED_SHARED")
    if ( lo2 ) {
      if ( lo2 == 1 ) _DA("STV_INTERNAL")
      else if ( lo2 == 2 ) _DA("STV_HIDDEN")
      else if ( lo2 == 3 ) _DA("STV_PROTECTED")
    } else
     _DA("STV_DEFAULT")
    fprintf(m_out, "\"\n");
#undef _DA
  }
}

int nv_dis::grab_syms_for_section(int s_idx)
{
  if ( m_syms.empty() ) return 0;
  m_curr_syms.clear();
  m_curr_siter = m_curr_syms.cend();
  // O(N) but that's fine
  std::for_each(m_syms.begin(), m_syms.end(), [&](asymbol &as) {
    if ( as.section != s_idx ) return;
    if ( as.type == STT_SECTION || as.type == STT_FILE ) return;
    m_curr_syms[as.addr] = &as;
  });
  if ( m_curr_syms.empty() ) return 0;
  m_curr_siter = m_curr_syms.cbegin();
  return 1;
}

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
    int miss = render(rend, r, p.first, p.second, l, opt_c);
    if ( miss ) {
      fprintf(m_out, " %d missed", miss);
      if ( opt_m ) {
        fputc(':', m_out);
        for ( auto &ms: m_missed ) fprintf(m_out, " %s", ms.c_str());
      }
    }
    fputc('\n', m_out);
    // body of instruction
    if ( opt_c ) {
      fprintf(m_out, " /*%lX*/ ", m_dis->offset());
    } else fputc('>', m_out);
    if ( dual_first ) fputs(" {", m_out);
    else if ( dual_last ) fputs("  ", m_out);
    if ( label )
     fprintf(m_out, " %s (* BRANCH_TARGET LABEL_%X *)", r.c_str(), label);
    else
      fprintf(m_out, " %s", r.c_str());
    if ( dual_last ) fputs(" }", m_out);
    if ( opt_c ) fputc(';', m_out);
    fputc('\n', m_out);
  } else
    fprintf(m_out, " NO_Render\n");
  if ( opt_O ) dump_ops( p.first, p.second );
  if ( opt_p ) dump_predicates( p.first, p.second, opt_c ? "; " : "P> " );
}

void nv_dis::try_dis(Elf_Word idx)
{
  auto branches = get_branch(idx);
  auto cbank = get_cbank(idx);
  auto rels = m_srels.find(idx);
  if ( rels != m_srels.end() ) {
    has_relocs = true;
    riter = rels->second.cbegin();
    riter_end = rels->second.cend();
    m_next_roff = riter->first;
#ifdef DEBUG
 fprintf(m_out, "idx %d size %ld first %lX\n", idx, rels->second.size(), m_next_roff);
#endif
  }
  if ( opt_c ) grab_syms_for_section(idx);
  if ( opt_T ) {
    if ( !m_rtdb ) m_rtdb = new reg_pad;
    else m_rtdb->clear();
  }
  dual_first = dual_last = false;
  while(1) {
    NV_res res;
    int get_res = m_dis->get(res);
    auto off = m_dis->offset();
    if ( -1 == get_res ) { fprintf(m_out, "; stop at %lX\n", off); break; }
    dis_total++;
    if ( !get_res ) {
      dis_notfound++;
      fprintf(m_out, "Not found at %lX", off);
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
    next_csym(off);
    int res_idx = 0;
    if ( res.size() > 1 ) res_idx = calc_index(res, m_dis->rz);
    // check branch label
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
      // check const bank
      auto rend = m_dis->get_rend(res[res_idx].first->n);
      if ( cbank ) {
        auto cb = check_cbank(rend, res[res_idx].second);
        if ( cb.has_value() ) {
          auto off = cb.value();
          if ( cbank->in_cb(off) ) {
            fprintf(m_out, " ; cb in section %d, offset %lX - %X = %lX\n",
              cbank->section, off, cbank->offset, off - cbank->offset);
          } else {
            auto cp = cbank->find_param(off);
            if ( cp )
              fprintf(m_out, " ; cb param %d off %lX size %X\n", cp->ordinal, off, cp->size);
            else
              fprintf(m_out, " ; unknown cb off %lX\n", off);
          }
        }
      }
      if ( m_rtdb )
        track_regs(rend, res[res_idx], off);
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
  if ( m_rtdb ) {
    finalize_rt();
    dump_rt();
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
  _parse_attrs(idx, sec);
  finalize_cparams(sec->get_info());
}

void nv_dis::_parse_attrs(Elf_Half idx, section *sec)
{
  if ( !opt_e ) return;
  if ( sec->get_type() == SHT_NOBITS ) return;
  auto size = sec->get_size();
  if ( !size ) return;
  const char *data = sec->get_data();
  auto sidx = sec->get_info();
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
    const char *kp = nullptr;
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
        kp = data + 4;
        if ( attr == 0xa ) { // EIATTR_PARAM_CBANK
          if ( a_len != 8 ) fprintf(m_out, "invalid PARAM_CBANK size %X\n", a_len);
          else {
            uint32_t sec_id = *(uint32_t *)kp;
            kp += 4;
            fprintf(m_out, " section index: %d\n", sec_id);
            unsigned short off = *(unsigned short *)kp;
            fprintf(m_out, " offset: %X\n", off);
            kp += 2;
            unsigned short size = *(unsigned short *)kp;
            fprintf(m_out, " size: %X\n", size);
            add_cbank(sidx, sec_id, off, size);
          }
        } else if ( attr == 0x17 ) // EIATTR_KPARAM_INFO
        {
          // from https://github.com/VivekPanyam/cudaparsers/blob/main/src/cubin.rs
          if ( a_len != 0xc ) fprintf(m_out, "invalid KPARAM_INFO size %X\n", a_len);
          else {
            fprintf(m_out, " Index: %X\n", *(uint32_t *)kp);
            kp += 4;
            unsigned short ord = *(unsigned short *)kp;
            fprintf(m_out, " ordinal: %d\n", ord);
            kp += 2;
            unsigned short off = *(unsigned short *)kp;
            fprintf(m_out, " offset: %X\n", off);
            kp += 2;
            uint32_t tmp = *(uint32_t *)kp;
            if ( tmp & 0xff ) fprintf(m_out, " align %d\n", tmp & 0xff);
            unsigned space = (tmp >> 0x8) & 0xf;
            if ( space ) fprintf(m_out, " space %X\n", space);
            int is_cbank = ((tmp >> 0x10) & 2) == 0;
            uint32_t csize = (((tmp >> 0x10) & 0xffff) >> 2);
            fprintf(m_out, " size %X %s\n", csize, is_cbank ? "cbank" : "");
            if ( is_cbank ) add_cparam(sidx, ord, off, csize);
          }
        } else if ( attr == 0x28 ) // EIATTR_COOP_GROUP_INSTR_OFFSETS
          ltype = NVLType::Coop_grp;
        else if ( attr == 0x1c ) // EIATTR_EXIT_INSTR_OFFSETS
          ltype = NVLType::Exit;
        else if ( attr == 0x1d ) // EIATTR_S2RCTAID_INSTR_OFFSETS
          ltype = NVLType::S2Rctaid;
        else if ( attr == 0x25 ) // EIATTR_LD_CACHEMOD_INSTR_OFFSETS
          ltype = NVLType::Ld_cachemode;
        else if ( attr == 0x31 ) // EIATTR_INT_WARP_WIDE_INSTR_OFFSETS
          ltype = NVLType::Warp_wide;
        else if ( attr == 0x39 ) // EIATTR_MBARRIER_INSTR_OFFSETS
          ltype = NVLType::MBarier;
        else if ( attr == 0x47 ) // EIATTR_SW_WAR_MEMBAR_SYS_INSTR_OFFSETS
          ltype = NVLType::War_membar;
        // read offsets
        if ( ltype ) {
          auto ib = get_branch(sidx);
          fill_eaddrs(&ib->labels, ltype, data, a_len);
        } else if ( attr == 0x34 ) {
          // collect indirect branches
          auto ib = get_branch(sidx);
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
    // there can be several labels for some addr, so add only if not exists yet
    auto ri = l->find(addr);
    if ( ri == l->end() )
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
      // resolve symbol
      asymbol *asym = nullptr;
      if ( sym < m_syms.size() ) asym = &m_syms[sym];
      fprintf(m_out, " [%ld] %lX sym %d %s", n, addr, sym, asym ? asym->name.c_str() : "");
      if ( add ) fprintf(m_out, " add %lX", add);
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
      // resolve symbol
      asymbol *asym = nullptr;
      if ( sym < m_syms.size() ) asym = &m_syms[sym];
      fprintf(m_out, " [%ld] %lX sym %d %s", n, addr, sym, asym ? asym->name.c_str() : "");
      if ( add ) fprintf(m_out, " add %lX", add);
      if ( tname )
       fprintf(m_out, " %s\n", tname);
      else
       fprintf(m_out, " type %d\n", type);
    }
  }
}

int nv_dis::fill_rels()
{
  for ( Elf_Half i = 0; i < n_sec; ++i ) {
    section *sec = reader.sections[i];
    auto st = sec->get_type();
    if ( st == SHT_REL || st == SHT_RELA ) {
      auto slink = sec->get_info();
      section *ls = reader.sections[slink];
#ifdef DEBUG
 fprintf(m_out, "link %d %s\n", slink, ls->get_name().c_str());
#endif
      auto st2 = ls->get_type();
      if ( st2 == SHT_NOBITS || !ls->get_size() ) continue;
      if ( strncmp(ls->get_name().c_str(), ".text.", 6) ) continue;
      // yup, this is our client
      const_relocation_section_accessor rsa(reader, sec);
      auto n = rsa.get_entries_num();
      SRels srels;
      for ( Elf_Xword ri = 0; ri < n; ri++ ) {
        Elf64_Addr addr;
        Elf_Word sym;
        unsigned type;
        Elf_Sxword add;
        if ( rsa.get_entry(ri, addr, sym, type, add) ) {
          srels[addr] = { type, sym };
        }
      }
#ifdef DEBUG
      fprintf(m_out, "store %ld relocs for section %d %s\n", srels.size(), slink, ls->get_name().c_str());
#endif
      if ( !srels.empty() )
      {
        auto prev = m_srels.find(slink);
        if ( prev == m_srels.end() )
          m_srels[slink] = std::move(srels);
        else {
          SRels &old = prev->second;
          for ( auto &p: srels ) old[p.first] = std::move(p.second);
        }
      }
    }
  }
  return !m_srels.empty();
}

void nv_dis::process()
{
  n_sec = reader.sections.size();
  if ( !n_sec ) {
    fprintf(stderr, "no sections\n");
  }
  auto et = reader.get_type();
  fprintf(m_out, "type %X, %d sections\n", et, n_sec);
  if ( opt_t || opt_r )
    read_symbols();
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
      if ( opt_c )
       fprintf(m_out, "\t.section %s\n", sname.c_str());
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
  printf("-T - track registers\n");
  exit(6);
}

int main(int argc, char **argv)
{
  int c;
  int s = -1;
  const char *o_fname = nullptr;
  while(1) {
    c = getopt(argc, argv, "cehmrtTNOpSs:o:");
    if ( c == -1 ) break;
    switch(c) {
      case 'c': opt_c = 1; break;
      case 'e': opt_e = 1; break;
      case 'h': opt_h = 1; break;
      case 'm': opt_m = 1; break;
      case 't': opt_t = 1; break;
      case 'T': opt_T = 1; break;
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