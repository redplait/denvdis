#include "nv_rend.h"

// for sv literals
using namespace std::string_view_literals;

template <typename T> requires std::is_base_of_v<render_base, T>
const NV_Prop *NV_renderer::find_compound_prop(const nv_instr *i, const T* ct) const
{
  if ( !i->props ) return nullptr;
  const NV_Prop *res = nullptr;
  constexpr bool has_left = requires(const T *t) {
    t->left;
  };
  if constexpr ( has_left ) {
    res = match_compound_prop(i, ct->left);
    if ( res ) return res;
  }
  constexpr bool has_right = requires(const T *t) {
    t->right;
  };
  if constexpr ( has_right ) {
    res = match_compound_prop(i, ct->right);
    if ( res ) return res;
  }
  return nullptr;
}

static const std::string_view s_tkey_gpr("GPR"), s_tkey_ugpr("UGPR"),
 s_tkey_pred("PRED"), s_tkey_upred("UPRED"),
 s_cc_prop("DOES_READ_CC"); // pred name for cc reading

// for write is_col = 0
static int fill_tab_chains(const NV_renderer::NV_pair &p, const std::string_view &key, RegTabChains *tlist, int is_col) {
  if ( !tlist ) return 0;
  auto what = is_col ? p.first->cols : p.first->rows;
  if ( !what ) return 0;
  int res = 0;
  for ( auto &wi: *what ) {
    // filter out if presents
    if ( wi.filter && !wi.filter(p.first, p.second) ) continue;
    // filter by connection
    if ( key != wi.tab->connection ) continue;
    tlist->push_back( { wi.tab, wi.idx } );
    res++;
  }
  return res;
}

// for CC we need check prefix
static int fill_tab_chain_CC(const NV_renderer::NV_pair &p, RegTabChains *tlist, int is_col) {
  if ( !tlist ) return 0;
  auto what = is_col ? p.first->cols : p.first->rows;
  if ( !what ) return 0;
  int res = 0;
  for ( auto &wi: *what ) {
    // filter out if presents
    if ( wi.filter && !wi.filter(p.first, p.second) ) continue;
    // filter by connection
    if ( wi.tab->connection[0] != 'C' || wi.tab->connection[1] != 'C' ) continue;
    tlist->push_back( { wi.tab, wi.idx } );
    res++;
  }
  return res;
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
  std::unordered_map<std::string_view, int> labels;
  // predicates
  int d_size = 0, d2_size = 0, a_size = 0, b_size = 0, b2_size = 0, c_size = 0, e_size = 0, h_size = 0, i_size = 0;
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
    pi = p.first->predicated->find("ISRC_B2_SIZE"sv);
    if ( pi != p.first->predicated->end() )
      b2_size = pi->second(p.second);
    pi = p.first->predicated->find("ISRC_C_SIZE"sv);
    if ( pi != p.first->predicated->end() )
      c_size = pi->second(p.second);
    pi = p.first->predicated->find("ISRC_E_SIZE"sv);
    if ( pi != p.first->predicated->end() )
      e_size = pi->second(p.second);
    pi = p.first->predicated->find("ISRC_H_SIZE"sv);
    if ( pi != p.first->predicated->end() )
      h_size = pi->second(p.second);
    pi = p.first->predicated->find("ISRC_I_SIZE"sv);
    if ( pi != p.first->predicated->end() )
      i_size = pi->second(p.second);
    // collect labels size
    for ( const auto &pred: *p.first->predicated ) {
      auto &kn = pred.first;
      if ( !kn.starts_with("ILABEL_") ) continue;
      size_t len = kn.size() - 7;
      // check if it ends with _SIZE
      if ( !kn.ends_with("_SIZE") ) continue;
      len -= 5;
      int psize = pred.second(p.second);
      if ( psize ) labels.emplace( std::string_view{ kn.data() + 7, len }, psize );
    }
  }

  int idx = -1;
  rtdb->pred_mask = 0;
  if ( is_s2xx(p.first) ) rtdb->pred_mask = (1 << 10);
  rtdb->m_reuse.apply(p.first, p.second);
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
    if ( check_cbank_pure(r, p.second, cb_idx, cb_off) ) {
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
         fill_tab_chains(p, s_tkey_pred, rtdb->rpred(kvi->second, off, 0), 1); res++; }
      else if ( !strcmp(ea->ename, "UniformPredicate") )
       { rtdb->pred_mask = 0x4000 | (1 + (unsigned short)kvi->second) << 11;
         fill_tab_chains(p, s_tkey_upred, rtdb->rupred(kvi->second, off, 0), 1); res++; }
      else
       fprintf(m_out, "unknown predicate %s at %lX\n", ea->ename, off);
      continue;
    }
    // xxSETP
#ifdef DEBUG
    if ( setp ) printf("%lX setp %d idx %d\n", off, setp, idx);
#endif
    if ( setp && !idx && (r->type == R_predicate || r->type == R_enum) ) {
      const render_named *rn = (const render_named *)r;
      const nv_eattr *ea = find_ea(p.first, rn->name);
      if ( !ea ) continue;
      if ( ea->ignore ) continue;
      auto kvi = p.second.find(rn->name);
      if ( kvi == p.second.end() ) continue;
      idx++;
      if ( !strcmp(ea->ename, "Predicate") && kvi->second != 7 )
       {
         fill_tab_chains(p, s_tkey_pred, rtdb->wpred(kvi->second, off, 0), 0);
         res++;
         continue;
       }
      else if ( !strcmp(ea->ename, "UniformPredicate") && kvi->second != 7 )
       {
         fill_tab_chains(p, s_tkey_upred, rtdb->wupred(kvi->second, off, 0), 0);
         res++;
         continue;
       }
      idx--;
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
        idx++;
        if ( !strcmp(ea->ename, "Predicate") && kvi->second != 7 )
        {
          fill_tab_chains(p, s_tkey_pred, rtdb->wpred(kvi->second, off, 0), 0);
          res++; continue;
        }
        else if ( !strcmp(ea->ename, "UniformPredicate") && kvi->second != 7 )
        {
          fill_tab_chains(p, s_tkey_upred, rtdb->wupred(kvi->second, off, 0), 0);
          res++; continue;
        }
        idx--;
      }
    }
    if ( r->type == R_opcode ) {
      idx = 0;
      continue;
    }
    auto rgpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi, int op_idx, NVP_type _t = GENERIC) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = reg_history::windex(i);
        if ( (int)kvi->second + i >= m_dis->rz ) break;
        fill_tab_chains(p, s_tkey_gpr, rtdb->rgpr(kvi->second + i, off, what, op_idx, _t), 1);
        res++;
      }
      return res;
    };
    auto gpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi, NVP_type _t = GENERIC) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = reg_history::windex(i);
        if ( (int)kvi->second + i >= m_dis->rz ) break;
        fill_tab_chains(p, s_tkey_gpr, rtdb->wgpr(kvi->second + i, off, what, _t), 0);
        res++;
      }
      return res;
    };
    auto rugpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi, int op_idx, NVP_type _t = GENERIC) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = reg_history::windex(i);
        if ( (int)kvi->second + i >= m_dis->rz ) break;
        fill_tab_chains(p, s_tkey_ugpr, rtdb->rugpr(kvi->second + i, off, what, op_idx, _t), 1);
        res++;
      }
      return res;
    };
    auto ugpr_multi = [&](unsigned short dsize, NV_extracted::const_iterator kvi, NVP_type _t = GENERIC) {
      int res = 0;
      for ( unsigned short i = 0; i < dsize / 32; i++ ) {
        reg_history::RH what = reg_history::windex(i);
        if ( (int)kvi->second + i >= m_dis->rz ) break;
        fill_tab_chains(p, s_tkey_ugpr, rtdb->wugpr(kvi->second + i, off, what, _t), 0);
        res++;
      }
      return res;
    };
    // dest(2)
    if ( idx >= 0 && (r->type == R_predicate || r->type == R_enum) ) {
      const render_named *rn = (const render_named *)r;
      const nv_eattr *ea = find_ea(p.first, rn->name);
#ifdef DEBUG
 printf("%lX idx %d %s\n", off, idx, rn->name);
#endif
      if ( !ea ) {
#ifdef DEBUG
  printf("%lX cannot find %s\n", off, rn->name);
#endif
        idx++;
        continue;
      }
      if ( ea->ignore ) continue;
      auto kvi = p.second.find(rn->name);
      if ( kvi == p.second.end() ) continue;
#ifdef DEBUG
printf("%lX idx %d reg %ld %s\n", off, idx, kvi->second, rn->name);
#endif
      if ( is_pred(ea, kvi) )
       {
         fill_tab_chains(p, s_tkey_pred, rtdb->rpred(kvi->second, off, 0), 1);
         res++; idx++;
         continue;
       }
      if ( is_upred(ea, kvi) )
       {
         fill_tab_chains(p, s_tkey_upred, rtdb->rupred(kvi->second, off, 0), 1);
         res++; idx++;
         continue;
       }
      if ( is_reg(ea, kvi) )
      {
        if ( is_sv(d_sv, rn->name) ) {
         if ( d_size <= 32 )
          { fill_tab_chains(p, s_tkey_gpr, rtdb->wgpr(kvi->second, off, 0, t1), 0); res++; }
         else res += gpr_multi(d_size, kvi, t1);
        } else if ( is_sv(d2_sv, rn->name) ) {
         if ( d2_size <= 32 )
         { fill_tab_chains(p, s_tkey_gpr, rtdb->wgpr(kvi->second, off, 0, t2), 0); res++; }
         else res += gpr_multi(d2_size, kvi, t2);
        } else if ( !strcmp(rn->name, "Rd") ) {
         if ( d_size <= 32 )
          { fill_tab_chains(p, s_tkey_gpr, rtdb->wgpr(kvi->second, off, 0, t1), 0); res++; }
         else res += gpr_multi(d_size, kvi, t1);
        } else if ( !strcmp(rn->name, "Rd2") ) {
         if ( d2_size <= 32 )
          { fill_tab_chains(p, s_tkey_gpr, rtdb->wgpr(kvi->second, off, 0, t2), 0); res++; }
         else res += gpr_multi(d2_size, kvi, t2);
        } else {
         if ( a_size > 32 && is_sv2(a_sv, rn->name, "Ra") )
          res += rgpr_multi(a_size, kvi, ISRC_A, t_a);
         else if ( b_size > 32 && is_sv2(b_sv, rn->name, "Rb") )
          res += rgpr_multi(b_size, kvi, ISRC_B, t_b);
         else if ( b2_size > 32 && !strcmp(rn->name, "Rb2") )
          res += rgpr_multi(b2_size, kvi, ISRC_B, t_b);
         else if ( c_size > 32 && is_sv2(c_sv, rn->name, "Rc") )
          res += rgpr_multi(c_size, kvi, ISRC_C, t_c);
         else if ( e_size > 32 && is_sv2(e_sv, rn->name, "Re") )
          res += rgpr_multi(e_size, kvi, ISRC_E, t_e);
         else if ( h_size > 32 && is_sv2(h_sv, rn->name, "Rh") )
          res += rgpr_multi(h_size, kvi, ISRC_H, t_h);
         else if ( i_size > 32 && is_sv2(nullptr, rn->name, "Ri") )
          res += rgpr_multi(i_size, kvi, 0);
         else
         {
           if ( is_sv2(a_sv, rn->name, "Ra") )
             fill_tab_chains(p, s_tkey_gpr, rtdb->rgpr(kvi->second, off, 0, ISRC_A, t_a), 1);
           else if ( is_sv2(b_sv, rn->name, "Rb") )
             fill_tab_chains(p, s_tkey_gpr, rtdb->rgpr(kvi->second, off, 0, ISRC_B, t_b), 1);
           else if ( !strcmp(rn->name, "Rb2") )
             fill_tab_chains(p, s_tkey_gpr, rtdb->rgpr(kvi->second, off, 0, ISRC_B, t_b), 1);
           else if ( is_sv2(c_sv, rn->name, "Rc") )
             fill_tab_chains(p, s_tkey_gpr, rtdb->rgpr(kvi->second, off, 0, ISRC_C, t_c), 1);
           else if ( is_sv2(e_sv, rn->name, "Re") )
             fill_tab_chains(p, s_tkey_gpr, rtdb->rgpr(kvi->second, off, 0, ISRC_E, t_e), 1);
           else if ( is_sv2(h_sv, rn->name, "Rh") ) // NOTE - ISRC_H_SIZE is always 32bit at time of writing this
             fill_tab_chains(p, s_tkey_gpr, rtdb->rgpr(kvi->second, off, 0, ISRC_H, t_h), 1);
           else fill_tab_chains(p, s_tkey_gpr, rtdb->rgpr(kvi->second, off, 0, 0), 1);
           res++;
         }
        }
      } else if ( is_ureg(ea, kvi) )
      {
        if ( is_sv(d_sv, rn->name) ) {
         if ( d_size <= 32 )
          { fill_tab_chains(p, s_tkey_ugpr, rtdb->wugpr(kvi->second, off, 0, t1), 0); res++; }
          else res += ugpr_multi(d_size, kvi, t1);
        } else if ( is_sv(d2_sv, rn->name) ) {
         if ( d2_size <= 32 )
           { fill_tab_chains(p, s_tkey_ugpr, rtdb->wugpr(kvi->second, off, 0, t2), 0); res++; }
         else res += ugpr_multi(d2_size, kvi, t2);
        } else if ( !strcmp(rn->name, "URd") ) {
         if ( d_size <= 32 )
          { fill_tab_chains(p, s_tkey_ugpr, rtdb->wugpr(kvi->second, off, 0, t1), 0); res++; }
         else res += ugpr_multi(d_size, kvi, t1);
        } else if ( !strcmp(rn->name, "URd2") ) {
         if ( d2_size <= 32 )
          { fill_tab_chains(p, s_tkey_ugpr, rtdb->wugpr(kvi->second, off, 0, t2), 0); res++; }
         else res += ugpr_multi(d2_size, kvi, t2);
        } else {
         if ( a_size > 32 && is_sv2(a_sv, rn->name, "URa") )
          res += rugpr_multi(a_size, kvi, ISRC_A, t_a);
         else if ( b_size > 32 && is_sv2(b_sv, rn->name, "URb") )
          res += rugpr_multi(b_size, kvi, ISRC_B, t_b);
         else if ( c_size > 32 && is_sv2(c_sv, rn->name, "URc") )
          res += rugpr_multi(c_size, kvi, ISRC_C, t_c);
         else if ( e_size > 32 && is_sv2(e_sv, rn->name, "URe") )
          res += rugpr_multi(e_size, kvi, ISRC_E, t_e);
         else if ( i_size > 32 && is_sv2(nullptr, rn->name, "URi") )
          res += rgpr_multi(i_size, kvi, 0);
         else
         {
           if ( is_sv2(a_sv, rn->name, "URa") )
             fill_tab_chains(p, s_tkey_ugpr, rtdb->rugpr(kvi->second, off, 0, ISRC_A, t_a), 1);
           else if ( is_sv2(a_sv, rn->name, "URb") )
             fill_tab_chains(p, s_tkey_ugpr, rtdb->rugpr(kvi->second, off, 0, ISRC_B, t_b), 1);
           else if ( is_sv2(a_sv, rn->name, "URc") )
             fill_tab_chains(p, s_tkey_ugpr, rtdb->rugpr(kvi->second, off, 0, ISRC_C, t_c), 1);
           else if ( is_sv2(a_sv, rn->name, "URe") )
             fill_tab_chains(p, s_tkey_ugpr, rtdb->rugpr(kvi->second, off, 0, ISRC_E, t_e), 1);
           else if ( is_sv2(nullptr, rn->name, "URi") )
             fill_tab_chains(p, s_tkey_ugpr, rtdb->rugpr(kvi->second, off, 0, 0), 1);
           else fill_tab_chains(p, s_tkey_ugpr, rtdb->rugpr(kvi->second, off, 0, 0), 1);
           res++;
         }
        }
      }
      idx++;
      continue;
    }
#ifdef DEBUG
printf("%lX idx %d rtype %d\n", off, idx, r->type);
#endif
    // we have something compound, size in out_size
    auto ve_type = [&](const ve_base &ve, int &out_size) -> NVP_type {
      auto len = strlen(ve.arg);
      if ( len < 2 ) return GENERIC;
      if ( len > 7 && !strcmp(ve.arg + len - 7, "_offset") ) len -= 7;
      if ( ve.arg[len - 2] != 'R' ) return GENERIC;
      if ( ve.arg[len - 1] == 'a' ) { out_size = a_size; return t_a; }
      if ( ve.arg[len - 1] == 'b' ) { out_size = b_size; return t_b; }
      if ( ve.arg[len - 1] == 'c' ) { out_size = c_size; return t_c; }
      if ( ve.arg[len - 1] == 'e' ) { out_size = e_size; return t_e; }
      if ( ve.arg[len - 1] == 'h' ) { out_size = h_size; return t_h; }
      return GENERIC;
    };
    auto check_ve_t = [&](const ve_base &ve, reg_history::RH what, const nv_eattr *ea, const NV_Prop *pr) {
        if ( ve.type == R_value ) return 0;
        auto kvi = p.second.find(ve.arg);
        if ( kvi == p.second.end() ) return 0;
        // check what we have
        if ( is_reg(ea, kvi) )
        {
          int psize = 0;
          if ( !strcmp(ve.arg, "Ra") ) psize = a_size;
          else if ( !strcmp(ve.arg,"Rb") ) psize = b_size;
          else if ( !strcmp(ve.arg,"Rb2") ) psize = b2_size;
          else if ( !strcmp(ve.arg,"Rc") ) psize = c_size;
          else if ( !strcmp(ve.arg,"Re") ) psize = e_size;
          else if ( !strcmp(ve.arg, "Rh") ) psize = h_size;
          auto type = pr ? pr->t : ve_type(ve, psize);
#ifdef DEBUG
printf("check_ve %s %d\n", ve.arg, psize);
#endif
          if ( pr ) what |= 1 + pr->op;
          auto li = labels.find(ve.arg);
          if ( li != labels.end() && li->second > 32 )
            rgpr_multi(li->second, kvi, pr ? pr->op : 0, type);
          else {
            if ( psize > 32 )
              rgpr_multi(psize, kvi, pr ? pr->op : 0, type);
            else
              fill_tab_chains(p, s_tkey_gpr, rtdb->rgpr(kvi->second, off, what | reg_history::comp, pr ? pr->op : 0, type), 1);
          }
          return 1;
        }
        if ( is_ureg(ea, kvi) ) {
          int psize = 0;
          if ( !strcmp(ve.arg,"URa") ) psize = a_size;
          else if ( !strcmp(ve.arg,"URb") ) psize = b_size;
          else if ( !strcmp(ve.arg, "URc") ) psize = c_size;
          else if ( !strcmp(ve.arg, "URe") ) psize = e_size;
          else if ( !strcmp(ve.arg, "URh") ) psize = h_size;
          auto type = pr ? pr->t : ve_type(ve, psize);
          if ( pr ) what |= 1 + pr->op;
          auto li = labels.find(ve.arg);
          if ( li != labels.end() && li->second > 32 )
            rugpr_multi(li->second, kvi, pr ? pr->op : 0, type);
          else {
            if ( psize > 32 )
              rugpr_multi(psize, kvi, pr ? pr->op : 0, type);
            else
              fill_tab_chains(p, s_tkey_ugpr, rtdb->rugpr(kvi->second, off, what | reg_history::comp, pr ? pr->op : 0, type), 1);
          }
          return 1;
        }
        // do we really can have predicates inside compound render items?
        if ( is_pred(ea, kvi) )
        { fill_tab_chains(p, s_tkey_pred, rtdb->rpred(kvi->second, off, what), 1); return 1; }
        if ( is_upred(ea, kvi) )
        { fill_tab_chains(p, s_tkey_upred, rtdb->rupred(kvi->second, off, what), 1); return 1; }
        return 0;
    };
    auto check_ve = [&](const ve_base &ve, reg_history::RH what, const NV_Prop *pr) {
        if ( ve.type == R_value ) return 0;
        const nv_eattr *ea = find_ea(p.first, ve.arg);
        if ( !ea ) return 0;
        return check_ve_t(ve, what, ea, pr);
    };
    auto check_ve_list = [&](const std::list<ve_base> &l, reg_history::RH what, const NV_Prop *pr) {
        int res = 0;
        for ( auto &ve: l ) {
          if ( ve.type == R_value ) continue;
          const nv_eattr *ea = find_ea(p.first, ve.arg);
          if ( !ea ) continue;
          if ( ea->ignore ) continue;
          res += check_ve_t(ve, what, ea, pr);
        }
        return res;
    };
#ifdef DEBUG
 fprintf(m_out, "@%lX: r->type %d\n", off, r->type);
#endif
    // AGALIARETPH - HELL
    if ( r->type == R_C || r->type == R_CX ) {
      const render_C *rn = (const render_C *)r;
      auto ctype = find_compound_prop(p.first, rn);
      res += check_ve(rn->left, 0, ctype);
      res += check_ve_list(rn->right, reg_history::in_list, ctype);
    } else if ( r->type == R_desc ) {
      const render_desc *rd = (const render_desc *)r;
      auto ctype = find_compound_prop(p.first, rd);
      res += check_ve(rd->left, 0, ctype);
      res += check_ve_list(rd->right, reg_history::in_list, ctype);
    } else if ( r->type == R_mem ) {
      const render_mem *rm = (const render_mem *)r;
      auto ctype = find_compound_prop(p.first, rm);
      res += check_ve_list(rm->right, reg_history::in_list, ctype);
    } else if ( r->type == R_TTU ) {
      const render_TTU *rt = (const render_TTU *)r;
      auto ctype = find_compound_prop(p.first, rt);
      res += check_ve(rt->left, 0, ctype);
    } else if ( r->type == R_M1 ) {
      const render_M1 *rt = (const render_M1 *)r;
      auto ctype = find_compound_prop(p.first, rt);
      res += check_ve(rt->left, 0, ctype);
    }
    idx++;
  }
  // track CC - must be last after filling pred_mask
  if ( p.first->predicated ) {
    auto cci = p.first->predicated->find(s_cc_prop);
    if ( cci != p.first->predicated->end() ) {
      int read_cc = cci->second(p.second);
      if ( read_cc ) fill_tab_chain_CC(p, rtdb->rcc(off), 1);
    }
  }
  // track writeCC
  auto ccki = p.second.find("writeCC");
  if ( ccki != p.second.end() && ccki->second )
    fill_tab_chain_CC(p, rtdb->wcc(off), 0);

  return res;
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
 if ( !rtdb->cc.empty() )
  std::sort(rtdb->cc.begin(), rtdb->cc.end(), srt);
 if ( !rtdb->cbs.empty() ) {
  std::sort(rtdb->cbs.begin(), rtdb->cbs.end(), [](const cbank_history &a, const cbank_history &b) { return a.off < b.off; });
 }
}

void NV_renderer::dump_rchains(const RegTabChains &rc, int is_col) const {
  if ( rc.empty() ) return;
  const NV_tab *old_tab = nullptr;
  for ( auto &pair: rc ) {
    if ( old_tab != pair.first ) {
      // dump table
      fprintf(m_out, "\ttab_%s(%s)", pair.first->name, pair.first->connection);
      old_tab = pair.first;
    }
    fprintf(m_out, " %c%d(%s)", is_col ? 'c' : 'r', pair.second,
     get_it(is_col ? pair.first->cols : pair.first->rows, pair.second).first);
  }
}

void NV_renderer::dump_rt(reg_pad *rtdb, int rc) const {
  if ( !rtdb ) return;
  if ( !rtdb->gpr.empty() ) {
    fprintf(m_out, ";;; %ld GPR\n", rtdb->gpr.size());
    dump_trset(rtdb->gpr, "R", rc);
  }
  if ( !rtdb->ugpr.empty() ) {
    fprintf(m_out, ";;; %ld UGPR\n", rtdb->ugpr.size());
    dump_trset(rtdb->ugpr, "UR", rc);
  }
  if ( !rtdb->pred.empty() ) {
    fprintf(m_out, ";;; %ld PRED\n", rtdb->pred.size());
    dump_rset(rtdb->pred, "P", rc);
  }
  if ( !rtdb->upred.empty() ) {
    fprintf(m_out, ";;; %ld UPRED\n", rtdb->upred.size());
    dump_rset(rtdb->upred, "UP", rc);
  }
  if ( !rtdb->cc.empty() ) {
   fprintf(m_out, ";;; %ld CC\n", rtdb->cc.size());
   constexpr int mask = (1 << 11) - 1;
   for ( auto &c: rtdb->cc ) {
      // truncated version of dump_rset
      int pred = 0;
      bool is_pred = c.has_pred(pred);
      if ( c.kind & 0x8000 )
      {
        if ( is_pred )
          fprintf(m_out, " ;   %lX <- %X %d", c.off, c.kind & mask, pred);
        else
          fprintf(m_out, " ;   %lX <- %X", c.off, c.kind & mask);
      } else {
        if ( is_pred )
          fprintf(m_out, " ;   %lX %X %d", c.off, c.kind & mask, pred);
        else
          fprintf(m_out, " ;   %lX %X", c.off, c.kind & mask);
      }
      if ( rc ) dump_rchains(c.tab_chain, !(c.kind & 0x8000));
      fputc('\n', m_out);
   }
  }
  if ( !rtdb->cbs.empty() ) {
   fprintf(m_out, ";;; %ld CBanks\n", rtdb->cbs.size());
   for ( auto &c: rtdb->cbs )
     fprintf(m_out, " ;   %lX: %X %lX size %d\n", c.off, c.cb_num, c.cb_off, c.kind & 0xf);
  }
}

void NV_renderer::dump_rset(const reg_pad::RSet &rs, const char *pfx, int rc) const
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
          fprintf(m_out, " ;   %lX <- %X %s%d", tr.off, tr.kind & mask, tr.kind & 0x4000 ? "UP" : "P", pred);
        else
          fprintf(m_out, " ;   %lX <- %X", tr.off, tr.kind & mask);
      } else {
        if ( is_pred )
          fprintf(m_out, " ;   %lX %X %s%d", tr.off, tr.kind & mask, tr.kind & 0x4000 ? "UP" : "P", pred);
        else
          fprintf(m_out, " ;   %lX %X", tr.off, tr.kind & mask);
      }
      if ( rc ) dump_rchains(tr.tab_chain, !(tr.kind & 0x8000));
      fputc('\n', m_out);
    }
  }
}

void NV_renderer::dump_trset(const reg_pad::TRSet &rs, const char *pfx, int rc) const
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
      if ( tr.is_reuse() ) fprintf(m_out, " reuse");
      if ( tname ) fprintf(m_out, " %s", tname);
      if ( rc ) dump_rchains(tr.tab_chain, !(tr.kind & 0x8000));
      fputc('\n', m_out);
    }
  }
}
