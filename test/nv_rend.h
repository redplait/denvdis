#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <map>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include "include/nv_types.h"

typedef std::unordered_set<uint32_t> NV_labels;

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
   int render(const NV_rlist *, std::string &res, const struct nv_instr *, const NV_extracted &, NV_labels *) const;
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
   void dump_out(const std::string_view &, FILE *) const;
   bool contain(const std::string_view &, char) const;
   // calculating best instruction from candidates
   int calc_miss(const struct nv_instr *, const NV_extracted &, int) const;
   int calc_index(const NV_res &, int) const;
   // predicates
   void dump_predicates(const struct nv_instr *, const NV_extracted &) const;
   int dump_predicates(const struct nv_instr *, const NV_extracted &, FILE *fp, const char *pfx) const;
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
   // renderer
   int rend_renderer(const NV_rlist *rlist, const std::string &opcode, std::string &res) const;
   void r_velist(const std::list<ve_base> &l, std::string &res) const;
   void r_ve(const ve_base &ve, std::string &res) const;
   int render_ve(const ve_base &, const struct nv_instr *, const NV_extracted &kv, std::string &) const;
   int render_ve_list(const std::list<ve_base> &, const struct nv_instr *, const NV_extracted &kv, std::string &) const;
   int check_mod(char c, const NV_extracted &, const char* name, std::string &r) const;
   void dump_value(const struct nv_instr *, const NV_extracted &kv, const std::string_view &,
     std::string &res, const nv_vattr &, uint64_t v) const;
   bool check_branch(const struct nv_instr *i, const NV_extracted::const_iterator &kvi, long &res) const;

   FILE *m_out;
   INV_disasm *m_dis = nullptr;
   Dvq_name m_vq = nullptr;
   int m_width;
   static const char *s_fmts[];
   // missed fields
   mutable std::unordered_set<std::string> m_missed;
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
    scond_count = 0,
    scond_succ = 0,
    scond_hits = 0;
};