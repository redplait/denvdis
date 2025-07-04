#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <map>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include "include/nv_types.h"

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

const char* get_merc_reloc_name(unsigned t);
const char* get_cuda_reloc_name(unsigned t);
float int_as_float(int);
double longlong_as_double(long long);

extern const float NVf_inf, NVf_nan;
extern const double NVd_inf, NVd_nan;

const char *get_prop_type_name(int i);
const char *get_prop_op_name(int i);

class NV_renderer {
 public:
   NV_renderer() {
     m_out = stdout;
   }
  ~NV_renderer() {
    if ( m_dis != nullptr ) delete m_dis;
    if ( m_out && m_out != stdout) fclose(m_out);
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
       fprintf(stderr, "cannot open output file %s, errno %d (%s)\n", of, errno, strerror(errno));
       m_out = stdout;
     }
   }
   typedef INV_disasm *(*Dproto)(void);
   typedef const char *(*Dvq_name)(int);
   typedef const NV_tabrefs * nv_instr::*Tab_field;
   typedef std::pair<const struct nv_instr *, NV_extracted> NV_pair;
   typedef std::vector<NV_pair> NV_res;
   typedef std::unordered_map<std::string_view, int> NV_Tabset;
   // relocs
   typedef std::pair<int, unsigned long> NV_rel;
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

   bool check_dual(const NV_extracted &);
   template <typename C>
   void render_rel(std::string &res, const NV_rel *, const C &) const;
   int render(const NV_rlist *, std::string &res, const struct nv_instr *, const NV_extracted &, NV_labels *, int opt_c = 0) const;
   const nv_eattr *try_by_ename(const struct nv_instr *, const std::string_view &sv) const;
   int fill_sched(const struct nv_instr *, const NV_extracted &);
   int dump_sched(const struct nv_instr *, const NV_extracted &);
   void dump_cond_list(const NV_Tabset *) const;
   bool check_sched_cond(const struct nv_instr *i, const NV_extracted &kv, const NV_one_cond &clist);
   bool check_sched_cond(const struct nv_instr *i, const NV_extracted &kv, const NV_one_cond &clist, NV_Tabset *);
   void dump_ops(const struct nv_instr *, const NV_extracted &) const;
   // string_view methods
   int cmp(const std::string_view &, const char *) const;
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
   // check const bank[0][imm]
   std::optional<long> check_cbank(const NV_rlist *r, const NV_extracted &kv) const;
   std::optional<long> check_cbank_right(const std::list<ve_base> &l, const NV_extracted &kv) const;
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
   bool check_branch(const struct nv_instr *i, const NV_extracted::const_iterator &kvi, long &res) const;
   bool check_ret(const struct nv_instr *i, const NV_extracted::const_iterator &kvi, long &res) const;

   FILE *m_out;
   INV_disasm *m_dis = nullptr;
   Dvq_name m_vq = nullptr;
   int m_width;
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