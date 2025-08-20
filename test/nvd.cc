#include "celf.h"
#include <unistd.h>

int opt_c = 0,
    opt_e = 0,
    opt_h = 0,
    opt_m = 0,
    opt_t = 0,
    opt_T = 0,
    opt_p = 0,
    opt_P = 0,
    opt_r = 0,
    opt_S = 0,
    opt_N = 0,
    opt_O = 0;

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

class nv_dis: public CElf<NV_renderer>
{
  public:
   virtual ~nv_dis() {
     for ( auto bi: m_branches ) delete bi.second;
     for ( auto cb: m_cbanks ) delete cb.second;
     if ( m_rtdb ) delete m_rtdb;
   }
   void process();
   int single_section(int idx);
   void dump_total() const
   {
    dis_stat();
    if ( !m_nopi.empty() ) {
      fprintf(m_out, "%ld instructions without properties\n", m_nopi.size());
      for ( auto pi: m_nopi )
       fprintf(m_out, " %d %s (%s) - %ld\n", pi.first->line, pi.first->name, pi.first->cname, pi.second);
    }
    if ( opt_S )
      fprintf(m_out, "filters %ld success %ld, conditions %ld (%ld) cached %ld\n",
        sfilters, sfilters_succ, scond_count, scond_succ, scond_hits);
   }
  protected:
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

   std::map<unsigned long, asymbol *> m_curr_syms;
   std::map<unsigned long, asymbol *>::const_iterator m_curr_siter = m_curr_syms.cend();
   int grab_syms_for_section(int);
   int next_csym(unsigned long);
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
   reg_pad *m_rtdb = nullptr;
   // no properties instructions
   std::unordered_map<const nv_instr *, unsigned long> m_nopi;
   void add_nopi(const nv_instr *i) {
     if ( !strcmp(i->name, "NOP") ||
          !strcmp(i->name, "EXIT") ||
          !strcmp(i->name, "DEPBAR") ||
          !strcmp(i->name, "MEMBAR") ||
          !strcmp(i->name, "LDGDEPBAR")
        ) return;
     auto iter = m_nopi.find(i);
     if ( iter == m_nopi.end() )
       m_nopi[i] = 1;
     else
       iter->second++;
   }
};

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
  int res = _read_symbols(opt_t, [&](asymbol &sym) { if ( opt_r ) m_syms.push_back(std::move(sym)); });
  if ( !res || m_syms.empty() ) return res;
  if ( opt_r ) fill_rels();
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
    if ( opt_c ) {
      int lut = 0;
      fputc(';', m_out);
      if ( check_lut(p.first, rend, p.second, lut) ) {
        auto lut_op = get_lut(lut);
        if ( lut_op ) fprintf(m_out, " LUT %X: %s", lut, lut_op);
        else fprintf(m_out, " unknown LUT %X", lut);
      }
    }
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
      bool skip_false = false;
      if ( m_rtdb ) {
        skip_false = always_false(res[res_idx].first, rend, res[res_idx].second);
        if ( !skip_false )
          track_regs(m_rtdb, rend, res[res_idx], off);
      }
      if ( !skip_false && opt_P && !res[res_idx].first->props )
        add_nopi(res[res_idx].first);
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
    finalize_rt(m_rtdb);
    dump_rt(m_rtdb);
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

int nv_dis::single_section(int idx)
{
  n_sec = m_reader->sections.size();
  if ( idx < 0 || idx >= n_sec ) return -1;
  section *sec = m_reader->sections[idx];
  if ( sec->get_type() == SHT_NOBITS ) return 0;
  if ( !sec->get_size() ) return 0;
  m_dis->init( (const unsigned char *)sec->get_data(), sec->get_size(), 0 );
  try_dis(idx);
  return 1;
}

void nv_dis::dump_mrelocs(section *sec)
{
  const_relocation_section_accessor rsa(*m_reader, sec);
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
  const_relocation_section_accessor rsa(*m_reader, sec);
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

void nv_dis::process()
{
  n_sec = m_reader->sections.size();
  if ( !n_sec ) {
    fprintf(stderr, "no sections\n");
  }
  auto et = m_reader->get_type();
  fprintf(m_out, "type %X, %d sections\n", et, n_sec);
  if ( opt_t || opt_r )
    read_symbols();
  // enum sections
  for ( Elf_Half i = 0; i < n_sec; ++i )
  {
    section *sec = m_reader->sections[i];
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
      m_dis->init( (const unsigned char *)sec->get_data(), sec->get_size(), 0 );
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
  printf("-c - dump instruction in form similar to original nvdisasm\n");
  printf("-e - dump attributes\n");
  printf("-h - hex dump\n");
  printf("-m - dump missed fields\n");
  printf("-N - dump not found masks\n");
  printf("-o - output file\n");
  printf("-O - dump operands\n");
  printf("-p - dump predicates\n");
  printf("-P - dump instructions without properties\n");
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
    c = getopt(argc, argv, "cehmrtTNOpPSs:o:");
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
      case 'P': opt_P = 1; break;
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
    elfio er;
    if ( o_fname ) dis.open_log(o_fname);
    if ( dis.open(&er, argv[i], opt_c) )
    {
      if ( s != -1 )
        dis.single_section(s);
      else
        dis.process();
    }
    dis.dump_total();
  }
}