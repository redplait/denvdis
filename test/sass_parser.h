#pragma once

#include "nv_rend.h"
#include <fp16.h>
#include <regex>

// black magic to get lambda arity from https://stackoverflow.com/questions/40411241/c-lambda-does-not-have-operator
template <typename T>
struct get_arity : get_arity<decltype(&std::remove_reference_t<T>::operator())> {};
template <typename R, typename... Args>
struct get_arity<R(Args...)> : std::integral_constant<unsigned, sizeof...(Args)> {};
template <typename R, typename... Args>
struct get_arity<R(Args...) const> : std::integral_constant<unsigned, sizeof...(Args)> {};
template <typename R, typename C, typename... Args>
struct get_arity<R(C::*)(Args...)> : std::integral_constant<unsigned, sizeof...(Args)> {};
template <typename R, typename C, typename... Args>
struct get_arity<R(C::*)(Args...) const> : std::integral_constant<unsigned, sizeof...(Args)> {};

// for sv literals
using namespace std::string_view_literals;

extern int opt_d, opt_v,
  skip_final_cut, // opt_e
  skip_op_parsing; // opt_o

class ParseSASS: public NV_renderer
{
  public:
   ParseSASS(): NV_renderer()
   { }
   virtual int init(const std::string &s) = 0;
   int add(const std::string &s, int idx = 0);
   inline int fsize() const {
     return (int)m_forms.size();
   }
  protected:
   int init_guts();
   int add_internal(const std::string &s, int idx);
   struct LTuple {
     const render_base *first;
     const nv_eattr *second;
     const NV_Renum *en;
   };
   struct form_list {
     form_list(const render_base *_rb) {
       rb = _rb;
     }
     inline bool empty() const {
       return lr.empty();
     }
     const render_base *rb = nullptr;
     std::list<LTuple> lr;
   };
   struct one_form
   {
     const nv_instr *instr;
     const NV_rlist *rend;
     std::unordered_map<std::string, uint64_t> l_kv; // local key-value for this form
     std::list<form_list *> ops;
     std::list<form_list *>::iterator current;
     one_form(const nv_instr *_i, const NV_rlist *_r) {
       instr = _i;
       rend = _r;
     }
     ~one_form() {
       for ( auto o: ops ) delete o;
     }
     // stored label
     inline bool has_label() const {
       return ltype != 0;
     }
     int ltype = 0;
     std::list<form_list *>::iterator lop = ops.end();
     std::string lname; // name of label
     // boring stuff
     one_form& operator=(one_form&& other) = default;
     one_form(one_form&& other) = default;
   };
   void dump(const one_form &) const;
   void dump(const form_list *fl, const nv_instr *instr) const;
   void dump_forms() const {
     for ( auto &f: m_forms ) dump(f);
   }
   typedef std::vector<one_form> NV_Forms;
   int has_target(const NV_Forms *f) const {
     return std::any_of(f->cbegin(), f->cend(), [](const one_form &of) { return nullptr != of.instr->target_index; });
   }
   int next(NV_Forms &) const;
   int fill_forms(NV_Forms &, const std::vector<const nv_instr *> &);
   // predicate
   bool has_ast;
   std::string m_pred;
   inline bool has_pred() const {
     return !m_pred.empty();
   }
   void reset_pred() {
     has_ast = false;
     m_pred.clear();
   }
   // heart of opcodes processing
   // check kind and return count of matches
   template <typename F>
   int check_kind(NV_Forms &forms, F &&pred) {
     int res = 0;
     constexpr unsigned arity = get_arity<F>{};
     for ( auto &f: forms )
     {
       for ( auto ci = f.current; ci != f.ops.end(); ++ci )
       {
         int pres;
         if constexpr ( arity == 2 )
           pres = pred((*ci)->rb, f);
         else
           pres = pred((*ci)->rb);
         if ( pres ) { res++; break; }
         if ( (*ci)->rb->type == R_predicate || (*ci)->rb->type == R_enum ) {
           // check if those predicate has default
           const render_named *rn = (const render_named *)(*ci)->rb;
           auto ea = find_ea(f.instr, rn->name);
           if ( !ea ) break;
           if ( !ea->has_def_value ) break;
           continue;
         }
         break;
       }
     }
     return res;
   }
   template <typename F>
   int check_op(NV_Forms &forms, F &&pred) {
     int res = 0;
     for ( auto &f: forms )
     {
       for ( auto ci = f.current; ci != f.ops.end(); ++ci )
       {
         if ( pred((*ci), f.instr) ) { res++; break; }
         if ( (*ci)->rb->type == R_predicate || (*ci)->rb->type == R_enum ) {
           // check if those predicate has default
           const render_named *rn = (const render_named *)(*ci)->rb;
           auto ea = find_ea(f.instr, rn->name);
           if ( !ea ) break;
           if ( !ea->has_def_value ) break;
           continue;
         }
         break;
       }
     }
     return res;
   }
   template <typename F>
   int apply_kind(NV_Forms &f, F &&pred) {
     constexpr unsigned arity = get_arity<F>{};
     std::erase_if(f, [&](one_form &f) {
       for ( auto ci = f.current; ci != f.ops.end(); ci++ )
       {
         int pres;
         if constexpr ( arity == 2 )
           pres = pred((*ci)->rb, f);
         else
           pres = pred((*ci)->rb);
         if ( pres ) { f.current = ci; return 0; }
         if ( (*ci)->rb->type == R_predicate ) {
           // check if those predicate has default
           const render_named *rn = (const render_named *)(*ci)->rb;
           auto ea = find_ea(f.instr, rn->name);
           if ( !ea ) break;
           if ( !ea->has_def_value ) break;
           continue;
         }
         return 1;
       }
       return 1;
     });
     return !f.empty();
   }
   // closure receives form_list* & one_form& to store local kv
   template <typename F>
   int apply_op(NV_Forms &f, F &&pred) {
     std::erase_if(f, [&](one_form &f) {
       for ( auto ci = f.current; ci != f.ops.end(); ci++ )
       {
         if ( pred((*ci), f) ) { f.current = ci; return 0; }
         if ( (*ci)->rb->type == R_predicate || (*ci)->rb->type == R_enum ) {
           // check if those predicate has default
           const render_named *rn = (const render_named *)(*ci)->rb;
           auto ea = find_ea(f.instr, rn->name);
           if ( !ea ) break;
           if ( !ea->has_def_value ) break;
           continue;
         }
         return 1;
       }
       return 1;
      });
      return !f.empty();
     }
     // like apply_op but 3rd arg is form_list iterator
     template <typename F>
     int apply_op2(NV_Forms &f, F &&pred) {
     std::erase_if(f, [&](one_form &f) {
       for ( auto ci = f.current; ci != f.ops.end(); ci++ )
       {
         if ( pred((*ci), f, ci) ) { f.current = ci; return 0; }
         if ( (*ci)->rb->type == R_predicate || (*ci)->rb->type == R_enum ) {
           // check if those predicate has default
           const render_named *rn = (const render_named *)(*ci)->rb;
           auto ea = find_ea(f.instr, rn->name);
           if ( !ea ) break;
           if ( !ea->has_def_value ) break;
           continue;
         }
         return 1;
       }
       return 1;
     });
     return !f.empty();
   }

   int set_num_value(const nv_vattr *, const char *name, one_form &f);
   typedef std::vector< std::pair<const std::list<ve_base> *, one_form *> > OFRights;
   template <typename C, typename F>
    OFRights collect_rights(F &&);
   template <typename C, typename F>
    void apply_mem_attrs(F &&);
   template <typename C, typename F>
    int parse_mem_right(int idx, const std::string_view &, F &&);
   template <typename C, typename F>
    int parse_c_left(int idx, const std::string &s, F &&);
   template <typename C>
    int parse_hex_tail(int idx, const C &s, int radix);
   template <typename C>
    int parse_float_tail(int idx, const C &s);
   int try_plus(const std::string_view &s, int start, int end, std::list<std::string_view> &elist);
   int parse_dot(const std::string_view &s, int start, int end, std::list<std::string_view> &elist);
   int parse_bitset(int idx, const std::string_view &s);
   int parse_req(const char *s);
   int parse_digit(const char *s, int &v);
   int parse_pred(int idx, const std::string &s);
   template <typename C>
    std::string extract_label(int idx, const C &s);
   int mark_label(int, std::string &s);
   int reduce_label(int, int, std::string &s);
   std::string process_tail(int idx, const std::string &s, NV_Forms &);
   int tail_attrs(int idx, const std::string_view &s, NV_Forms &);
   int process_tail_attr(int idx, const std::string_view &s, NV_Forms &);
   int process_attr(int idx, const std::string &s, NV_Forms &);
   template <typename T>
    int try_dotted(int, T &, std::string_view &dotted, int &dotted_last);
   int classify_op(int op_idx, const std::string_view &s);
   int reduce(int);
   int reduce_value();
   int reduce_enum(const std::string_view &);
   int reduce_pred(const std::string_view &, int exclamation = 0);
   int apply_enum(const std::string_view &);
   int enum_tail(int idx, const std::string_view &);
   NV_Forms m_forms;
   // current kv - used for predicate & tail like usched_info etc
   NV_extracted m_kv;
   // current numeric value
   enum NumV {
     nan = 1,
     inf = 2,
     num = 3,
     fp  = 4,
   };
   union {
     unsigned long m_v;
     float m_f;
     double m_d;
   };
   int m_numv = 0;
   char m_minus = 0;
   char m_tilda = 0; // ~ for @invert
   char m_abs = 0;   // | for @absolute
   void reset_v() {
     m_numv = 0;
     m_minus = m_tilda = m_abs = 0;
   }
   static std::regex s_digits;
   static std::regex s_commas;
   static constexpr auto c_usched_name = "usched_info";
   const NV_sorted *m_sorted = nullptr;
   const NV_Renums *m_renums = nullptr;
   const NV_Renum *usched = nullptr;
   const NV_Renum *pseudo = nullptr;
   const NV_dotted *m_dotted = nullptr;
};

