#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <map>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include "include/nv_types.h"

template <typename T>
struct dumb_freea
{
  dumb_freea(T *ptr)
    : m_ptr(ptr)
   { }
  void operator=(T *arg)
   {
     if ( (m_ptr != NULL) && (m_ptr != arg) )
       delete []m_ptr;
     m_ptr = arg;
   }
  protected:
   T *m_ptr;
};

// labels type
enum NVLType {
  Label = 0,
  Warp_wide = 1, // from EIATTR_INT_WARP_WIDE_INSTR_OFFSETS
  Coop_grp, // EIATTR_COOP_GROUP_INSTR_OFFSETS
  Exit,     // EIATTR_EXIT_INSTR_OFFSETS
  S2Rctaid, // EIATTR_S2RCTAID_INSTR_OFFSETS
  Ld_cachemode, // EIATTR_LD_CACHEMOD_INSTR_OFFSETS
  MBarier,  // EIATTR_MBARRIER_INSTR_OFFSETS
  War_membar,  // EIATTR_SW_WAR_MEMBAR_SYS_INSTR_OFFSETS
  Ind_BT,   // reffered from EIATTR_INDIRECT_BRANCH_TARGETS
};

typedef std::unordered_map<uint32_t, int> NV_labels;

enum NV_LType {
  BRANCH_TARGET = 1,
  LABEL,
  L32,
  H32,
  INDIRECT_CALL,
};

inline std::string& rstrip(std::string &s)
{
  while(!s.empty()) {
    auto c = s.back();
    if ( !isspace(c) ) break;
    s.pop_back();
  }
  return s;
}

struct reg_reuse {
  unsigned short mask = 0, // actual values of reuse_src_X
    mask2 = 0; // if reuse_src_X presents in this instruction
  unsigned char keep = 0, // actual values of keep_X
    keep2 = 0; // if keep_x presents in this instruction
  inline void clear() {
    mask = mask2 = 0;
  }
  int apply(const struct nv_instr *, const NV_extracted &kv);
  // 1 << (idx - ISRC_A)
  inline int ra() const { return mask & 1; }
  inline int rb() const { return mask & 2; }
  inline int rc() const { return mask & 4; }
  inline int re() const { return mask & 8; }
  inline int rh() const { return mask & 16; }
  // there are only keep_a & keep_b - both were introduced in sm100
  inline int ka() const { return keep & 1; }
  inline int kb() const { return keep & 2; }
};

struct reg_history {
  unsigned long off;
  // 0x8000 - write, else read
  // 0x4000 - Uniform predicate, else just predicate
  // next 3 bits are predicate reg index + 1 (bcs T == 7 and 0 is perfectly valid predicate)
  // next 1 bit - if was load from Special Reg (1 << 10)
  // next 1 bit - reuse flag (1 << 9)
  // next 1 bit - part of compound (1 << 8)
  // next 1 bit - list of compound (1 << 7)
  // next 3 bit is index for wide operation - it can be up to 256 bit (like SRC_I) / 32 = 8
  // finally low 4 bit is NVP_ops
  typedef unsigned short RH;
  static constexpr RH reuse = 1 << 9;
  static constexpr RH comp  = 1 << 8;
  static constexpr RH in_list = 1 << 7;
  RH kind;
  inline bool is_upred() const {
    return kind & 0x4000;
  }
  inline bool is_reuse() const {
    return kind & reuse;
  }
  inline bool has_pred(int &p) const {
    p = (kind >> 11) & 0x7;
    if ( p ) {
      p--;
      return true;
    }
    return false;
  }
  inline bool has_ops(int &op) const {
    op = kind & 0x7;
    if ( op ) {
      op--;
      return true;
    }
    return false;
  }
  static inline RH windex(int w) {
    return (w & 7) << 4;
  }
  inline int windex() const {
    return (kind >> 4) & 7;
  }
};

struct typed_reg_history: public reg_history {
  NVP_type type = GENERIC;
};

struct cbank_history {
  unsigned long off, cb_off;
  // kind - low 4 bits is size in bytes
  unsigned short cb_num, kind;
};

// snapshot of registers acessed/patched for current single instruction
struct track_snap {
  // key: GPR has prefix 0, UGPR 0x8000
  // value: 0x80 - write
  //        0x40 - reuse
  //        0x20 - read even if we already have write
  //        0x0x - ISRC_XX
  std::unordered_map<unsigned short, unsigned char> gpr;
  // 2 set of predictes: 1 - read, 2 - write
  static constexpr int pr_size = 7;
  char pr[pr_size] = { 0, 0, 0, 0, 0, 0, 0 },
      upr[pr_size] = { 0, 0, 0, 0, 0, 0, 0 };
  void reset() {
    gpr.clear();
    memset(pr, 0, pr_size); memset(upr, 0, pr_size);
  }
  bool empty_pr() const {
    return std::all_of(pr, pr + pr_size, [](char c) -> bool { return !c; });
  }
  bool empty_upr() const {
    return std::all_of(upr, upr + pr_size, [](char c) -> bool { return !c; });
  }
  bool empty() const {
    if ( !gpr.empty() ) return false;
    return empty_pr() && empty_upr();
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
  typedef std::unordered_map<int, std::vector<typed_reg_history> > TRSet;
  TRSet gpr, ugpr;
  RSet pred, upred;
  std::vector<cbank_history> cbs;
  track_snap *snap = nullptr;
  reg_reuse m_reuse;
  reg_history::RH pred_mask = 0;
  // if you want some inheritance - make destructor virtual
  ~reg_pad() {
    if ( snap ) delete snap;
  }
  // boring stuff
  reg_history::RH check_reuse(int op) const {
    if ( op < ISRC_A) return 0;
    if ( m_reuse.mask & (1 << (op - ISRC_A)) ) return reg_history::reuse;
    return 0;
  }
  void add_cb(unsigned long off, unsigned long cb_off, unsigned short cb_num, unsigned short k) {
    cbs.push_back( { off, cb_off, cb_num, k });
  }
  void _add(RSet &rs, int idx, unsigned long off, reg_history::RH k) {
    if ( snap ) {
      if ( &rs == &pred ) {
        if ( k & 0x8000 )
         snap->pr[idx] |= 2;
        else
         snap->pr[idx] |= 1;
      } else {
        if ( k & 0x8000 )
         snap->upr[idx] |= 2;
        else
         snap->upr[idx] |= 1;
      }
    }
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
  void _add(TRSet &rs, int idx, unsigned long off, reg_history::RH k, NVP_type t = GENERIC) {
    k |= pred_mask;
    auto ri = rs.find(idx);
    if ( ri != rs.end() ) {
      if ( !ri->second.empty() ) { // check if prev item is the same
        auto &last = ri->second.back();
        if ( last.off == off && last.kind == k ) return;
      }
      ri->second.push_back( { off, k, t } );
    } else {
     std::vector<typed_reg_history> tmp;
     tmp.push_back( { off, k, t } );
     rs[idx] = std::move(tmp);
    }
  }
  void rgpr(int r, unsigned long off, reg_history::RH k, int op, NVP_type t = GENERIC) {
     auto reuse = check_reuse(op);
     if ( snap ) {
       std::unordered_map<unsigned short, unsigned char>::iterator si = snap->gpr.find(r);
       if ( si != snap->gpr.end() ) {
         si->second |= 0x20;
         if ( reuse ) si->second |= 0x40;
       } else
       snap->gpr[r] = op | (reuse ? 0x40 : 0) | (k & 0x8000 ? 0x80: 0);
     }
    _add(gpr, r, off, k | reuse, t);
  }
  void wgpr(int r, unsigned long off, reg_history::RH k, NVP_type t = GENERIC) {
     if ( snap ) snap->gpr[r] = 0x80;
    _add(gpr, r, off, k | 0x8000, t);
  }
  void rugpr(int r, unsigned long off, reg_history::RH k, int op, NVP_type t = GENERIC) {
     auto reuse = check_reuse(op);
     if ( snap ) {
       std::unordered_map<unsigned short, unsigned char>::iterator si = snap->gpr.find(r | 0x8000);
       if ( si != snap->gpr.end() ) {
         si->second |= 0x20;
         if ( reuse ) si->second |= 0x40;
       } else
       snap->gpr[r | 0x8000] = op | (reuse ? 0x40 : 0) | (k & 0x8000 ? 0x80: 0);
     }
    _add(ugpr, r, off, k | reuse, t);
  }
  void wugpr(int r, unsigned long off, reg_history::RH k, NVP_type t = GENERIC) {
     if ( snap ) snap->gpr[r | 0x8000] = 0x80;
    _add(ugpr, r, off, k | 0x8000, t);
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
  bool empty() const {
    return gpr.empty() && pred.empty() && ugpr.empty() && upred.empty() && cbs.empty();
  }
  void clear() {
     pred_mask = 0;
     gpr.clear();
     pred.clear();
     ugpr.clear();
     upred.clear();
     cbs.clear();
  }
};

void HexDump(FILE *f, const unsigned char *From, int Len);
const char* get_merc_reloc_name(unsigned t);
const char* get_cuda_reloc_name(unsigned t);
float int_as_float(int);
double longlong_as_double(long long);

extern const float NVf_inf, NVf_nan;
extern const double NVd_inf, NVd_nan;

const char *get_prop_type_name(int i);
const char *get_prop_op_name(int i);
const char *get_lut(int i);

// error log interface
struct NV_ELog {
  virtual void verr(const char *format, va_list *ap) = 0;
};

class NV_renderer {
 public:
   NV_renderer() {
     m_out = stdout;
   }
  virtual ~NV_renderer() {
    if ( m_dis != nullptr ) delete m_dis;
    if ( m_out && m_out != stdout) fclose(m_out);
  }
  static void finalize_rt(reg_pad *);
  static bool is_compound(NV_rend r) { return r >= R_C; }
  // string_view methods
  static bool cmp(const std::string_view &, const char *);
  static bool is_sv(const std::string_view *sv, const char *name)
  {
     if ( !sv ) return false;
     size_t i = 0;
     for ( auto c = sv->cbegin(); c != sv->cend(); ++c, ++i ) {
       char nc = name[i];
       if ( !nc ) return false;
       if ( *c != nc ) return false;
     }
    return !name[i] && (i == sv->size());
  }
  static bool is_sv2(const std::string_view *sv, const char *name, const char *pfx)
  {
     return NV_renderer::is_sv(sv, name) || !strcmp(name, pfx);
  }
  int load(const char *);
  int load(std::string &s) {
    return load(s.c_str());
  }
  void dis_stat() const;
  void open_log(const char *of) {
     if ( m_out && m_out != stdout ) {
       fclose(m_out);
     }
     m_out = fopen(of, "a");
     if ( !m_out ) {
       Err("cannot open output file %s, errno %d (%s)\n", of, errno, strerror(errno));
       m_out = stdout;
     }
   }
   void render_cword(uint64_t, char *, size_t) const;
   typedef INV_disasm *(*Dproto)(void);
   typedef const char *(*Dvq_name)(int);
   typedef const NV_tabrefs * nv_instr::*Tab_field;
   typedef std::pair<const struct nv_instr *, NV_extracted> NV_pair;
   typedef std::vector<NV_pair> NV_res;
   typedef std::unordered_map<std::string_view, int> NV_Tabset;
   // relocs
   typedef std::pair<int, unsigned long> NV_rel;
   // error log interface
   NV_ELog *m_elog = nullptr;
#ifdef __GNUC__
          __attribute__ (( format( printf, 2, 3 ) ))
#endif
    void Err(const char *, ...) const;
  protected:
   template <typename T, typename I>
   const T& get_it(const std::initializer_list<T>& list, I index) const {
     return *(list.begin() + index);
   }
   template <typename T>
   const T *find(const std::initializer_list<T>& list, const std::string_view &what) const {
     if ( !list.size() ) return nullptr;
     int low = 0, high = list.size() - 1;
     while (low <= high) {
       size_t mid = low + (high - low) / 2;
       auto mid_e = list.begin() + mid;
       auto cmp_res = (mid_e->name <=> what);
       if ( cmp_res == std::strong_ordering::equal )
            return mid_e;
        if ( cmp_res == std::strong_ordering::less )
            low = mid + 1;
        else
            high = mid - 1;
     }
     return nullptr;
   }
   template <typename T>
   const T *find(const std::initializer_list<T>* list, const std::string_view &what) const {
     if ( !list ) return nullptr;
     return find(*list, what);
   }
   // monadic version for std::pairs
   template <typename T, typename S>
   const T *find_il(const std::vector<std::pair<const std::string_view, T> > *list, const S &what) const {
     if ( !list || !list->size() ) return nullptr;
     int low = 0, high = list->size() - 1;
     while (low <= high) {
       size_t mid = low + (high - low) / 2;
       auto mid_e = list->begin() + mid;
       auto cmp_res = (mid_e->first <=> what);
       if ( cmp_res == std::strong_ordering::equal )
            return &mid_e->second;
        if ( cmp_res == std::strong_ordering::less )
            low = mid + 1;
        else
            high = mid - 1;
     }
     return nullptr;
   }
   // fields types
   template <typename S>
   const NV_field *find_field(const struct nv_instr *ins, const S& fn) const {
     auto fi = std::lower_bound(ins->fields.begin(), ins->fields.end(), fn, [](const NV_field &f, const S& fn) {
        return f.name < fn;
      });
     if ( fi == ins->fields.end() ) return nullptr;
     return &(*fi);
   }
   const char *has_predicate(const NV_rlist *) const;
   template <typename S>
   const NV_cbank *is_cb_field(const struct nv_instr *ins, const S& fn) const
   {
     if ( !ins->cb_field ) return nullptr;
     if ( ins->cb_field->f1 == fn ) return ins->cb_field;
     if ( ins->cb_field->f2 == fn ) return ins->cb_field;
     return nullptr;
   }
   template <typename S>
   const NV_cbank *is_cb_field(const struct nv_instr *ins, const S& fn, int idx) const
   {
     if ( !ins->cb_field ) return nullptr;
     if ( ins->cb_field->f1 == fn ) { idx = 0; return ins->cb_field; }
     if ( ins->cb_field->f2 == fn ) { idx = 1; return ins->cb_field; }
     return nullptr;
   }
   template <typename S>
   const NV_tab_fields *is_tab_field(const struct nv_instr *ins, const S& fn) const
   {
     if ( !ins->tab_fields.size() ) return nullptr;
     for ( auto tf: ins->tab_fields ) {
       for ( auto &s: tf->fields )
         if ( s == fn ) return tf;
     }
     return nullptr;
   }
   template <typename S>
   const NV_tab_fields *is_tab_field(const struct nv_instr *ins, const S& fn, int &idx) const
   {
     if ( !ins->tab_fields.size() ) return nullptr;
     for ( auto tf: ins->tab_fields ) {
       idx = 0;
       for ( auto &s: tf->fields ) {
         if ( s == fn ) return tf;
         ++idx;
       }
     }
     return nullptr;
   }
   int copy_tail_values(const struct nv_instr *, const NV_rlist *, const NV_extracted &, NV_extracted &out_res) const;
   int make_tab_row(int optv, const struct nv_instr *ins, const NV_tab_fields *,
     const NV_extracted &, std::vector<unsigned short> &res, int ignore) const;

   int collect_labels(const NV_rlist *, const struct nv_instr *, const NV_extracted &, NV_labels *, long *out_addr) const;
   bool check_dual(const NV_extracted &) const;
   template <typename C>
   void render_rel(std::string &res, const NV_rel *, const C &) const;
   int render(const NV_rlist *, std::string &res, const struct nv_instr *, const NV_extracted &, NV_labels *, int opt_c = 0) const;
   const nv_eattr *try_by_ename(const struct nv_instr *, const std::string_view &sv) const;
   int fill_sched(const struct nv_instr *, const NV_extracted &);
   int dump_sched(const struct nv_instr *, const NV_extracted &);
   void dump_cond_list(const NV_Tabset *) const;
   bool check_sched_cond(const struct nv_instr *i, const NV_extracted &kv, const NV_one_cond &clist) const;
   bool check_sched_cond(const struct nv_instr *i, const NV_extracted &kv, const NV_one_cond &clist, NV_Tabset *) const;
   void dump_ops(const struct nv_instr *, const NV_extracted &) const;
   inline bool is_tail(const nv_vattr *vi, const render_named *rn) const {
     return vi && vi->kind == NV_BITSET && !strncmp(rn->name, "req_", 4);
   }
   inline bool is_tail(const struct nv_instr *i, const render_base *r) const {
     // some instructions missed req_bit_set and so tail starts at USCHED_INFO
     if ( r->type == R_enum ) {
       const render_named *rn = (const render_named *)r;
       const nv_eattr *ea = find_ea(i, rn->name);
       if ( !ea )
         return false;
       return !strcmp(ea->ename, "USCHED_INFO");
     }
     if ( r->type != R_value ) return false;
     const render_named *rn = (const render_named *)r;
     return is_tail(find(i->vas, rn->name), rn);
   }
   void dump_tab_fields(const NV_tab_fields *) const;
   void dump_sv(const std::string_view &) const;
   void dump_out(const std::string_view &) const;
   void dump_outln(const std::string_view &) const;
   void dump_out(const std::string_view &, FILE *) const;
   void dump_outln(const std::string_view &, FILE *) const;
   bool contain(const std::string_view &, char) const;
   // calculating best instruction from candidates
   int calc_miss(const struct nv_instr *, const NV_extracted &, int) const;
   int calc_index(const NV_res &, int) const;
   // predicates
   void dump_predicates(const struct nv_instr *, const NV_extracted &, const char *pfx) const;
   int dump_predicates(const struct nv_instr *, const NV_extracted &, FILE *fp, const char *pfx) const;
   int dump_op_props(const struct nv_instr *, FILE *fp, const char *pfx) const;
   // rend common suffixes logic
   template <typename F, typename T>
   int cs_rend(const NV_rlist *rlist, F f, T arg) const
   {
     int res = 0, state = 0;
     for ( auto r: *rlist ) {
       if ( !state ) {
         if ( r->type == R_opcode ) state = 1;
         continue;
       }
       if ( r->type != R_enum ) break;
       res++;
       f((const render_named *)r, arg);
     }
     return res;
   }
   // rend filters
   template <typename F>
   bool fbn_r_ve(const ve_base &ve, F &f) const {
    if ( ve.arg ) return f(ve.arg);
    return false;
   }
   template <typename F>
   bool fbn_r_velist(const std::list<ve_base> &l, F &f) const
   {
     return std::any_of(l.begin(), l.end(), [&](const ve_base &ve) -> bool { return fbn_r_ve(ve, f); });
   }
   template <typename F>
   bool fbn_rend(const NV_rlist *rlist, F &f) const
   {
     for ( auto r: *rlist ) {
      switch(r->type) {
       case R_value:
       case R_predicate:
       case R_enum: {
        const render_named *rn = (const render_named *)r;
        if ( rn->name && f(rn->name) ) return 1;
       } break;
       case R_C:
       case R_CX: {
        const render_C *rn = (const render_C *)r;
        if ( rn->name && f(rn->name) ) return 1;
        if ( fbn_r_ve(rn->left, f) ) return 1;
        if ( fbn_r_velist(rn->right, f) ) return 1;
       } break;
       case R_TTU: {
        const render_TTU *rt = (const render_TTU *)r;
        if ( fbn_r_ve(rt->left, f) ) return 1;
       } break;
       case R_M1: {
        const render_M1 *rt = (const render_M1 *)r;
        if ( rt->name && f(rt->name) ) return 1;
        if ( fbn_r_ve(rt->left, f) ) return 1;
      } break;
       case R_desc: {
        const render_desc *rt = (const render_desc *)r;
        if ( fbn_r_ve(rt->left, f) ) return 1;
        if ( fbn_r_velist(rt->right, f) ) return 1;
      } break;
       case R_mem: {
        const render_mem *rm = (const render_mem *)r;
        if ( fbn_r_velist(rm->right, f) ) return 1;
      } break;
      default: ; // no name check for R_opcode and to avoid stupid 'not handled in switch' warning
     }
    }
    return 0;
   }
   template <typename S>
   const nv_eattr *find_ea(const struct nv_instr *i, S s) const
   {
     auto ei = find(i->eas, s);
     if ( ei ) return ei->ea;
     return try_by_ename(i, s);
   }
   bool has_not(const render_named *rn, const NV_extracted &kv) const;
   // check for !@PT or !@UPT
   bool always_false(const struct nv_instr *, const NV_rlist *, const NV_extracted &kv) const;
   // check for some @PXX != PT
   bool has_predicate(const NV_rlist *, const NV_extracted &kv) const;
   // PRMT mask
   bool check_prmt(const struct nv_instr *, const NV_rlist *r, const NV_extracted &kv, unsigned long &mask) const;
   // LUT imm
   bool check_lut(const struct nv_instr *, const NV_rlist *r, const NV_extracted &kv, int &idx) const;
   // check for xxSETP
   bool is_setp(const struct nv_instr *, int &ends2) const;
   bool is_s2xx(const struct nv_instr *) const; // (C)S2(U)R
   // const bank methods
   template <typename T>
     requires std::is_member_function_pointer_v<T>
   bool check_cbank_t(T, const render_base *, const NV_extracted &kv, unsigned short &cb_idx,
     unsigned long &cb_off) const;
   bool check_cbank(const render_base *rb, const NV_extracted &kv, unsigned short &cb_idx,
     unsigned long &cb_off) const;
   bool check_cbank_pure(const render_base *rb, const NV_extracted &kv, unsigned short &cb_idx,
     unsigned long &cb_off) const;
   template <typename T>
     requires std::is_member_function_pointer_v<T>
   std::optional<long> get_cbank_t(T, const NV_rlist *r, const NV_extracted &kv, unsigned short *cb_idx) const;
   std::optional<long> check_cbank_right(const std::list<ve_base> &l, const NV_extracted &kv) const;
   std::optional<long> check_cbank_right_pure(const std::list<ve_base> &l, const NV_extracted &kv) const;
   // c[cb_idx][reg + imm] - returns imm
   std::optional<long> check_cbank(const NV_rlist *r, const NV_extracted &kv, unsigned short *cb_idx = nullptr) const;
   // c[cb_idx][imm]
   std::optional<long> check_cbank_pure(const NV_rlist *r, const NV_extracted &kv, unsigned short *cb_idx = nullptr) const;
   // try to find item in renderer for some NV_Prop where count of fields > 1
   const render_base *try_compound_prop(const NV_rlist *r, const NV_Prop *) const;
   bool _cmp_prop(const std::list<ve_base> &vb, const NV_Prop *pr) const;
   // renderer
   int rend_singleE(const struct nv_instr *, const render_base *r, std::string &res) const;
   template <typename Fs, typename Fl>
   int rend_single(const render_base *r, std::string &res, const char *opcode, Fs &&, Fl &&) const;
   int rend_single(const render_base *r, std::string &res, const char *opcode = nullptr) const;
   int rend_renderer(const NV_rlist *rlist, const std::string &opcode, std::string &res) const;
   int rend_rendererE(const struct nv_instr *, const NV_rlist *rlist, std::string &res) const;
   void r_velist(const std::list<ve_base> &l, std::string &res) const;
   void r_ve(const ve_base &ve, std::string &res) const;
   void r_vei(const struct nv_instr *, const ve_base &ve, std::string &res) const;
   void r_velisti(const struct nv_instr *, const std::list<ve_base> &l, std::string &res) const;
   int render_ve(const ve_base &, const struct nv_instr *, const NV_extracted &kv, std::string &) const;
   int render_ve_list(const std::list<ve_base> &, const struct nv_instr *, const NV_extracted &kv, std::string &) const;
   int check_mod(char c, const NV_extracted &, const char* name, std::string &r) const;
   int check_abs(const NV_extracted &, const char* name) const;
   int check_abs(const NV_extracted &, const char* name, std::string &r) const;
   void dump_value(const nv_vattr &, uint64_t v, NV_Format, std::string &res) const;
   void dump_value(const struct nv_instr *, const NV_extracted &kv, const std::string_view &,
     std::string &res, const nv_vattr &, uint64_t v) const;
   bool extract(const struct nv_instr *i, const NV_extracted::const_iterator &kvi, long &res) const;
   bool conv_simm(const struct nv_instr *i, const NV_extracted::const_iterator &kvi, long &res) const;
   bool check_branch(const struct nv_instr *i, const NV_extracted::const_iterator &kvi, long &res) const;
   bool check_ret(const struct nv_instr *i, const NV_extracted::const_iterator &kvi, long &res) const;
   // reg_pads
   const NV_Prop *match_compound_prop(const nv_instr *i, const ve_base &) const;
   const NV_Prop *match_compound_prop(const nv_instr *i, const std::list<ve_base> &) const;
   template <typename T> requires std::is_base_of_v<render_base, T>
   const NV_Prop *find_compound_prop(const nv_instr *i, const T*) const;
   int track_regs(reg_pad *, const NV_rlist *, const NV_pair &p, unsigned long off);
   void dump_rt(reg_pad *) const;
   void dump_rset(const reg_pad::RSet &, const char *pfx) const;
   void dump_trset(const reg_pad::TRSet &, const char *pfx) const;
   inline bool is_pred(const nv_eattr *ea, NV_extracted::const_iterator &kvi) const {
     return !strcmp(ea->ename, "Predicate") && 7 != kvi->second;
   }
   inline bool is_upred(const nv_eattr *ea, NV_extracted::const_iterator &kvi) const {
     return !strcmp(ea->ename, "UniformPredicate") && 7 != kvi->second;
   }
   bool crack_h2(const char *) const;
   inline bool is_reg(const nv_eattr *ea, NV_extracted::const_iterator &kvi) const {
     return (!strcmp(ea->ename, "Register") || !strcmp(ea->ename, "NonZeroRegister") ||
             !strcmp(ea->ename, "RegisterFAU") || !strcmp(ea->ename, "NonZeroRegisterFAU") || crack_h2(ea->ename)
            )
      && m_dis->rz != (int)kvi->second;
   }
   inline bool is_ureg(const nv_eattr *ea, NV_extracted::const_iterator &kvi) const {
     return (!strcmp(ea->ename, "UniformRegister") || !strcmp(ea->ename, "NonZeroUniformRegister")) && m_dis->rz != (int)kvi->second;
   }
   inline bool is_bd(const nv_eattr *ea) const {
     return !strcmp(ea->ename, "BD");
   }
   // validation
   int validate_tabs(const struct nv_instr *, NV_extracted &);

   FILE *m_out;
   INV_disasm *m_dis = nullptr;
   Dvq_name m_vq = nullptr;
   int m_width;
   int m_block_mask = 0;
   // convert offset to start of block
   unsigned long to_block_start(unsigned long loff) const {
     if ( !m_block_mask ) return loff;
     if ( (loff & m_block_mask) == 8 )
       return loff & ~m_block_mask;
     return loff;
   }
   // missed fields
   mutable std::unordered_set<std::string> m_missed;
   // relocs
   mutable unsigned long m_next_roff;
   mutable bool has_relocs = false;
   virtual const NV_rel *next_reloc(std::string_view &) const {
     return nullptr;
   }
   virtual const std::string *try_name(unsigned long off) const {
     return nullptr;
   }
   bool check_rel(const struct nv_instr *i) const;
   // dual issues
   bool dual_first = false;
   bool dual_last = false;
   // scheduling tracking, value - list of column indexes
   std::unordered_map<const NV_tab *, std::list< std::pair<short, NV_Tabset *> > > m_sched;
   std::list<NV_Tabset> m_cached_tabsets;
   // disasm stat
   mutable long
    dis_total = 0,
    dis_notfound = 0,
    dis_dups = 0,
    sfilters = 0,
    sfilters_succ = 0,
    missed_enums = 0,
    scond_count = 0,
    scond_succ = 0,
    scond_hits = 0;
   // static fields
   static const char *s_fmts[];
   static const char *s_labels[];
   static const char *s_ltypes[];
   static std::map<int, std::pair<const char *, const char *> > s_sms;
};