#include <dlfcn.h>
#include <fp16.h>
#include "nv_rend.h"

extern int opt_m;

int NV_renderer::load(std::string &sm_name)
{
     void *dh = dlopen(sm_name.c_str(), RTLD_NOW);
     if ( !dh ) {
      fprintf(stderr, "cannot load %s, errno %d (%s)\n", sm_name.c_str(), errno, strerror(errno));
       return 0;
     }
     m_vq = (Dvq_name)dlsym(dh, "get_vq_name");
     if ( !m_vq )
       fprintf(stderr, "cannot find get_vq_nam(%s), errno %d (%s)\n", sm_name.c_str(), errno, strerror(errno));
     Dproto fn = (Dproto)dlsym(dh, "get_sm");
     if ( !fn ) {
      fprintf(stderr, "cannot find get_sm(%s), errno %d (%s)\n", sm_name.c_str(), errno, strerror(errno));
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

int NV_renderer::cmp(const std::string_view &sv, const char *s) const
{
  size_t i = 0;
  for ( auto c = sv.cbegin(); c != sv.cend(); ++c, ++i ) {
    if ( *c != s[i] ) return 0;
  }
  return 1;
}

bool NV_renderer::contain(const std::string_view &sv, char sym) const
{
  return sv.find(sym) != std::string::npos;
}

void NV_renderer::dump_value(const struct nv_instr *ins, const NV_extracted &kv, const std::string_view &var_name,
  std::string &res, const nv_vattr &a, uint64_t v) const
{
  char buf[128];
  uint32_t f32;
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
  switch(kind)
  {
    case NV_F64Imm:
      snprintf(buf, 127, "%f", *(double *)&v);
     break;
    case NV_F16Imm:
      f32 = fp16_ieee_to_fp32_bits((uint16_t)v);
      snprintf(buf, 127, "%f", *(float *)&f32);
     break;
    case NV_F32Imm:
      snprintf(buf, 127, "%f", *(float *)&v);
     break;
    default:
      if ( !v ) { res += '0'; return; }
      snprintf(buf, 127, "0x%lX", v);
  }
  buf[127] = 0;
  res += buf;
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

int NV_renderer::calc_miss(const struct nv_instr *ins, const NV_extracted &kv, int rz) const
{
  int res = 0;
  for ( auto ki: kv ) {
    const nv_eattr *ea = nullptr;
    auto kiter = find(ins->eas, ki.first);
    if ( kiter ) { ea = kiter->ea; }
    else { ea = try_by_ename(ins, ki.first); }
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
  auto ei = find(i->eas, ve.arg);
  if ( !ei ) return 1;
  const nv_eattr *ea = ei->ea;
  auto kvi = kv.find(ve.arg);
  if ( kvi == kv.end() ) { if ( opt_m ) m_missed.insert(ve.arg); return 1; }
  auto eid = ea->em->find(kvi->second);
  if ( eid != ea->em->end() )
    res += eid->second;
  else return 1;
  return 0;
}

// render right []
int NV_renderer::render_ve_list(const std::list<ve_base> &l, const struct nv_instr *i, const NV_extracted &kv, std::string &res) const
{
  auto size = l.size();
  if ( 1 == size )
    return render_ve(*l.begin(), i, kv, res);
  int missed = 0;
  int idx = 0;
  for ( auto ve: l ) {
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
      else if ( idx ) res += '+';
      res += tmp;
      idx++;
      continue;
    }
    // this is (optional) enum
    const nv_eattr *ea = nullptr;
    auto ei = find(i->eas, ve.arg);
    if ( ei ) { ea = ei->ea; }
    else { ea = try_by_ename(i, ve.arg); }
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
    if ( ea->has_def_value && ea->def_value == (int)kvi->second ) {
      if ( ea->ignore && !ea->print ) continue;
      // ignore zero register even without ea->ignore
      if ( !strcmp(ea->ename, "ZeroRegister") ) continue;
    }
    if ( !ea->ignore ) idx++;
    if ( ea->ignore ) res += '.';
    else {
      if ( ve.pfx ) res += ve.pfx;
      else if ( idx > 1 ) res += " + ";
    }
    auto eid = ea->em->find(kvi->second);
    if ( eid != ea->em->end() )
       res += eid->second;
    else {
       missed++;
       continue;
    }
  }
  return missed;
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
  auto wi = find(i->vwidth, kvi->first);
  if ( !wi ) return false;
  // yes, this is some imm for branch, check if it negative
  if ( kvi->second & (1L << (wi->w - 1)) )
    res = kvi->second - (1L << wi->w);
  else
    res = (long)kvi->second;
  return true;
}

int NV_renderer::render(const NV_rlist *rl, std::string &res, const struct nv_instr *i, const NV_extracted &kv, NV_labels *l) const
{
  int idx = 0;
  int missed = 0;
  int was_bs = 0; // seems that scheduling args always starts with BITSET req_xx
  int prev = -1;  // workaround to fix op, bcs testcc is missed
  for ( auto ri: *rl ) {
    std::string tmp;
    switch(ri->type)
    {
      case R_opcode:
       res += i->name;
       break;

      case R_value: {
        const render_named *rn = (const render_named *)ri;
        auto kvi = kv.find(rn->name);
        if ( kvi == kv.end() ) {
          if ( opt_m ) m_missed.insert(rn->name);
          missed++;
          break;
        }
        auto vi = find(i->vas, rn->name);
        if ( !vi ) { missed++; break; }
        if ( vi->kind == NV_BITSET && !strncmp(rn->name, "req_", 4) ) was_bs = 1;
        long branch_off = 0;
        if ( check_branch(i, kvi, branch_off) ) {
          char buf[128];
          snprintf(buf, 127, "%ld", branch_off);
          tmp = buf;
          // make (LABEL_xxx)
          snprintf(buf, 127, " (LABEL_%lX)", branch_off + m_dis->off_next());
          if ( l ) l->insert(branch_off + m_dis->off_next());
          tmp += buf;
        } else
          dump_value(i, kv, rn->name, tmp, *vi, kvi->second);
        if ( rn->pfx ) { if ( prev != R_opcode ) res += rn->pfx; res += ' '; }
        else if ( was_bs ) res += " &";
        res += tmp;
       } break;

      case R_enum: {
         const render_named *rn = (const render_named *)ri;
         const nv_eattr *ea = nullptr;
         auto ei = find(i->eas, rn->name);
         if ( ei ) { ea = ei->ea; }
         else { ea = try_by_ename(i, rn->name); }
         if ( !ea ) {
           missed++;
           idx++;
           continue;
         }
         auto kvi = kv.find(rn->name);
         if ( kvi == kv.end() ) {
           kvi = kv.find(ea->ename);
           if ( kvi == kv.end() ) {
             if ( opt_m ) m_missed.insert(rn->name);
             missed++;
             idx++;
             continue;
           }
         }
         // now we have enum attr in ea and value in kvi
         // we have 2 cases - if this attr has ignore and !print and value == def_value - we should skip it
         if ( ea->has_def_value && ea->def_value == (int)kvi->second && ea->ignore && !ea->print ) {
           idx++; continue;
         }
         if ( ea->ignore ) res += '.';
         else {
           if ( rn->pfx ) {
             if ( '?' == rn->pfx ) res += ' ';
             res += rn->pfx;
           } else res += ' ';
           // check mod
           if ( rn->mod ) check_mod(rn->mod, kv, rn->name, res);
         }
         auto eid = ea->em->find(kvi->second);
         if ( eid != ea->em->end() )
           res += eid->second;
         else {
           missed++;
           break;
         }
         if ( ea->ignore ) {
           idx++; continue;
         }
       } break;

      case R_predicate: { // like enum but can be ignored if has default value
         const render_named *rn = (const render_named *)ri;
         auto ei = find(i->eas, rn->name);
         if ( !ei ) {
           missed++;
           break;
         }
         const nv_eattr *ea = ei->ea;
         auto kvi = kv.find(rn->name);
         if ( kvi == kv.end() ) {
           if ( opt_m ) m_missed.insert(rn->name);
           missed++;
           break;
         }
         if ( ea->def_value == (int)kvi->second ) break;
         if ( rn->pfx ) res += rn->pfx;
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
         if ( rn->pfx ) res += rn->pfx;
         else res += ' ';
         if ( rn->mod ) check_mod(rn->mod, kv, rn->name, res);
         res += "c:[";
         missed += render_ve(rn->left, i, kv, res);
         res += "][";
         missed += render_ve_list(rn->right, i, kv, res);
         res += ']';
       } break;

      case R_TTU: {
         const render_TTU *rt = (const render_TTU *)ri;
         if ( rt->pfx ) res += rt->pfx;
         else res += ' ';
         res += "ttu:[";
         missed += render_ve(rt->left, i, kv, res);
         res += ']';
       } break;

      case R_M1: {
         const render_M1 *rt = (const render_M1 *)ri;
         if ( rt->pfx ) res += rt->pfx;
         else res += ' ';
         res += rt->name;
         res += ":[";
         missed += render_ve(rt->left, i, kv, res);
         res += ']';
       } break;

      case R_desc: {
         const render_desc *rt = (const render_desc *)ri;
         if ( rt->pfx ) res += rt->pfx;
         else res += ' ';
         res += "desc:[";
         missed += render_ve(rt->left, i, kv, res);
         res += "],[";
         missed += render_ve_list(rt->right, i, kv, res);
         res += ']';
       } break;

      case R_mem: {
         const render_mem *rt = (const render_mem *)ri;
         if ( rt->pfx ) res += rt->pfx;
         else res += ' ';
         res += "[";
         missed += render_ve_list(rt->right, i, kv, res);
         res += ']';
       } break;

      default: fprintf(stderr, "unknown rend type %d at index %d for inst %s\n", ri->type, idx, i->name);
    }
    prev = ri->type;
    idx++;
  }
  return missed;
}

void NV_renderer::dump_predicates(const struct nv_instr *i, const NV_extracted &kv) const
{
  if ( !i->predicated ) return;
  for ( auto &pred: *i->predicated ) {
    fputs("P> ", m_out);
    dump_sv(pred.first);
    int res = pred.second(kv);
    if ( res >= 0 && m_vq && cmp(pred.first, "VQ") ) {
     auto name = m_vq(res);
     if ( name )
       fprintf(m_out, ": %s (%d)", name, res);
     else
       fprintf(m_out, ": %d", res);
    } else
      fprintf(m_out, ": %d", res);
    fputc('\n', m_out);
  }
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
    const nv_eattr *ea = nullptr;
    auto ei = find(i->eas, kv1.first);
    if ( ei ) { ea = ei->ea; }
    else { ea = try_by_ename(i, kv1.first); }
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
  NV_Tabset tmp;
  return check_sched_cond(i, kv, clist, tmp);
}

bool NV_renderer::check_sched_cond(const struct nv_instr *i, const NV_extracted &kv, const NV_one_cond &clist,
 NV_Tabset &out_res)
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
    out_res[cond.first] = (int)kiter->second;
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
      if ( !check_sched_cond(i, kv, get_it(titer.tab->cols, titer.idx), row_res) ) {
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
      if ( !check_sched_cond(i, kv, get_it(titer.tab->rows, titer.idx), row_res) ) {
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

bool NV_renderer::check_dual(const NV_extracted &kv)
{
  if ( m_width != 88 ) return false;
  auto kvi = kv.find("usched_info");
  if ( kvi == kv.end() ) return false;
  return kvi->second == 0x10; // floxy2
}
