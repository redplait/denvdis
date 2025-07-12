#include "sass_parser.h"

std::regex ParseSASS::s_digits("\\d+");
std::regex ParseSASS::s_commas("\\s*,\\s*");

void ParseSASS::dump(const one_form &of) const
{
  printf("%s line %d:", of.instr->name, of.instr->line);
  for ( auto ops = of.current; ops != of.ops.end(); ++ops ) {
    std::string res;
    rend_single((*ops)->rb, res);
    printf(" %s", res.c_str());
    if ( (*ops)->rb->type == R_predicate || (*ops)->rb->type == R_enum ) {
     // check if those predicate has default
     const render_named *rn = (const render_named *)(*ops)->rb;
     auto ea = find_ea(of.instr, rn->name);
     if ( !ea ) continue;
     if ( ea->has_def_value ) printf(".D(%d):%s", ea->def_value, ea->ename);
     else printf(".E:%s", ea->ename);
    }
  }
  fputc('\n', stdout);
}

void ParseSASS::dump(const form_list *fl, const nv_instr *instr) const
{
  printf("%s line %d:", instr->name, instr->line);
  std::string res;
  rend_single(fl->rb, res);
  printf("%s\n", res.c_str());
}

int ParseSASS::_extract_ve(NV_extracted &res, const one_form *of, const ve_base &vb)
{
  if ( vb.type != R_enum ) return 1;
  auto ea = find_ea(of->instr, vb.arg);
  if ( !ea ) return 0;
  if ( !ea->has_def_value || !ea->def_value ) return 1;
  res[vb.arg] = ea->def_value;
  return 1;
}

int ParseSASS::_extract_vel(NV_extracted &res, const one_form *of, const std::list<ve_base> &rl)
{
  for ( auto &r: rl )
    if ( !_extract_ve(res, of, r) ) return 0;
  return 1;
}

int ParseSASS::extract_full(NV_extracted &res)
{
  if ( m_forms.empty() ) return 0;
  // if there are several - no difference which to use so let it be first
  const one_form *of = &m_forms.at(0);
  return _extract_full(res, of);
}

int ParseSASS::_extract_full(NV_extracted &res, const one_form *of)
{
  int retval = _extract(res, of);
  if ( !retval ) return retval;
  // lets fill all non-zero defaults
  for ( auto r: *(of->rend) ) {
    if ( r->type == R_enum || r->type == R_predicate ) {
     const render_named *rn = (const render_named *)r;
     auto ea = find_ea(of->instr, rn->name);
     if ( !ea ) return 0;
     if ( !ea->has_def_value || !ea->def_value ) continue;
     // add this field
     res[rn->name] = ea->def_value;
    } else if ( r->type == R_mem ) {
      const render_mem *rm = (const render_mem *)r;
      if ( !_extract_vel(res, of, rm->right) ) return 0;
    } else if ( r->type == R_C || r->type == R_CX ) {
      const render_C *rn = (const render_C *)r;
      if ( !_extract_ve(res, of, rn->left) || !_extract_vel(res, of, rn->right) ) return 0;
    } else if ( r->type == R_desc ) {
      const render_desc *rd = (const render_desc *)r;
      if ( !_extract_ve(res, of, rd->left) || !_extract_vel(res, of, rd->right) ) return 0;
    } else if ( r->type == R_TTU ) {
      const render_TTU *rt = (const render_TTU *)r;
      if ( !_extract_ve(res, of, rt->left) ) return 0;
    } else if ( r->type == R_M1 ) {
      const render_M1 *rt = (const render_M1 *)r;
      if ( !_extract_ve(res, of, rt->left) ) return 0;
    }
  }
  return retval;
}

int ParseSASS::extract(NV_extracted &res)
{
  if ( m_forms.empty() ) return 0;
  // if there are several - no difference which to use so let it be first
  const one_form *of = &m_forms.at(0);
  return _extract(res, of);
}

int ParseSASS::_extract(NV_extracted &res, const one_form *of)
{
  // add locals
  res.insert(of->l_kv.begin(), of->l_kv.end());
  // check if we have predicate
  if ( has_pred() ) {
    auto first_r = *of->rend->begin();
    if ( first_r->type != R_predicate ) return 0;
    // map name in m_pred to key
    const render_named *rn = (const render_named *)first_r;
    auto ea = find_ea(of->instr, rn->name);
    if ( !ea ) return 0;
    // check if it has enum in s
    auto en = m_renums->find(ea->ename);
    if ( en == m_renums->end() ) return 0;
    auto aiter = en->second->find(m_pred);
    if ( aiter == en->second->end() ) {
      fprintf(stderr, "cannot map predicate %s (%s)\n", m_pred.c_str(), ea->ename);
      return 0;
    }
    res[rn->name] = aiter->second;
    if ( has_ast ) {
      std::string ast = rn->name;
      ast += "@not";
      auto pair = m_kv.emplace(std::make_pair(ast, 1));
      res[pair.first->first] = 1;
    }
  }
  // merge m_kv
  for ( auto &pkv: m_kv )
    res[pkv.first] = pkv.second;
  return 1;
}

int ParseSASS::parse_pred(int idx, const std::string &s)
{
  if ( s.at(idx) != '@' ) return 0;
  int res = idx + 1;
  reset_pred();
  if ( s.at(res) == '!' ) {
    has_ast = 1;
    res++;
  }
  while ( !isspace(s.at(res)) ) { m_pred.push_back(s.at(res)); res++; }
  for ( ; res < (int)s.size(); res++ ) if ( !isspace(s.at(res)) ) break; // skip spaces after predicate
  return res;
}

// return len
int ParseSASS::parse_digit(const char *s, int &v)
{
  char *end;
  if ( s[0] == '0' && s[1] == 'x' ) {
    v = strtol(s + 2, &end, 16);
    return end - s;
  }
  v = strtol(s, &end, 10);
  return end - s;
}

// return new idx in s
template <typename C>
int cut_lspaces(int idx, const C &s)
{
  for ( ; idx < (int)s.size(); ++idx ) {
    char c = s.at(idx);
    if ( !isspace(c) ) break;
  }
  return idx;
}

template <typename C>
int ParseSASS::parse_float_tail(int idx, const C &s)
{
  char *end;
  const char *start = s.data() + idx;
  m_d = strtod(start, &end);
  if ( m_minus ) m_d = -m_d;
  m_numv = NumV::fp;
  int diff = int(end - start);
  idx += diff;
  if ( idx >= (int)s.size() ) return idx;
  // skip trailing spaces
  return cut_lspaces(idx, s);
}

template <typename C>
int ParseSASS::parse_hex_tail(int idx, const C &s, int radix)
{
  char *end;
  const char *start = s.data() + idx;
  m_v = strtol(start, &end, radix);
  m_numv = NumV::num;
  int diff = int(end - start);
  idx += diff;
  if ( idx >= (int)s.size() ) return idx;
  // skip trailing spaces
  return cut_lspaces(idx, s);
}

// parse {list,of,digits}, assign result to m_v for next call of reduce_value
int ParseSASS::parse_bitset(int idx, const std::string_view &s)
{
  m_v = 0;
  m_numv = NumV::num;
  for ( int i = idx; i < (int)s.size(); i++ ) {
    char c = s.at(i);
    if ( c == '}' ) return 1;
    if ( isdigit(c) ) { m_v |= 1 << (c - '0'); continue; }
    if ( c != ',' && !isspace(c) ) break;
  }
  return 0;
}

int ParseSASS::parse_req(const char *s)
{
  int i = 0;
  for ( ; s[i] != '}'; i++ ) ;
  // ripped from https://stackoverflow.com/questions/10058606/splitting-a-string-by-a-character
  int req = 0;
  std::cregex_token_iterator begin(s, s + i, s_digits), end;
  std::for_each(begin, end, [s,&req]( const std::string &ss ) {
    int v = atoi(ss.c_str());
    if ( v > 5 ) fprintf(stderr, "bad req index %d in %s\n", v, s);
    else req |= 1 << v;
  });
  // push into kv
  m_kv["req_bit_set"] = req;
  return i + 1;
}

template <typename C>
std::string ParseSASS::extract_label(int idx, const C &s)
{
  std::string res;
  // skip spaces
  int i = idx;
  char c;
  for ( ; i < (int)s.size(); i++ ) {
    c = s.at(i);
    if ( !isspace(c) ) break;
  }
  for ( ; i < (int)s.size(); i++ ) {
    c = s.at(i);
    if ( c == ')' || c == '"' ) break;
    res.push_back(c);
  }
  return res;
}

int ParseSASS::mark_label(int t, std::string &s)
{
  int res = 0;
  std::for_each(m_forms.begin(), m_forms.end(), [&](one_form &of) {
    res++;
    of.ltype = t;
    of.lname = s; // don't use std::move here bcs we don't know size of remained items in m_forms
   });
  return res;
}

int ParseSASS::reduce(int kind)
{
  auto cl = [kind](const render_base *rb) { return rb->type == kind; };
  return apply_kind(m_forms, cl);
}

int ParseSASS::set_num_value(const nv_vattr *vas, const char *name, one_form &of)
{
  if ( !vas ) return 0;
  if ( vas->kind == NV_SImm || vas->kind == NV_SSImm || vas->kind == NV_RSImm ) {
    long l = (long)m_v;
    if ( m_minus ) l = -l;
    of.l_kv[name] = l;
  } else if ( vas->kind == NV_BITSET || vas->kind == NV_UImm )
    of.l_kv[name] = m_v;
  // for cases when float number didn't contained '.' and so was readed in parse_hex_tail into m_v
  // also need to take into account m_minus here
  else if ( vas->kind == NV_F64Imm )
  {
    double d = (double)this->m_v;
    if ( m_minus ) d = -d;
    of.l_kv[name] = *(uint64_t *)&d;
  } else if ( vas->kind == NV_F32Imm ) {
    float fl = (float)this->m_v;
    if ( m_minus ) fl = -fl;
    uint64_t v;
    *(float *)&v = fl;
    of.l_kv[name] = v;
  } else if ( vas->kind == NV_F16Imm ) {
    uint64_t v = fp16_ieee_from_fp32_value(float(m_minus ? -m_v : m_v));
    of.l_kv[name] = v;
  }
  return 1;
}

int ParseSASS::reduce_value()
{
  if ( m_numv == NumV::num ) {
    return apply_kind(m_forms, [&](const render_base *rb, one_form &f) {
      if ( rb->type != R_value ) return 0;
      if ( f.instr->vas ) {
        const render_named *rn = (const render_named *)rb;
        set_num_value(find(f.instr->vas, rn->name), rn->name, f);
      }
      return 1;
    });
  } else if ( m_numv == NumV::fp ) {
    return apply_kind(m_forms, [&](const render_base *rb, one_form &f) {
      if ( rb->type != R_value ) return 0;
      if ( f.instr->vas ) {
        const render_named *rn = (const render_named *)rb;
        auto vas = find(f.instr->vas, rn->name);
        if ( vas ) {
          uint64_t v;
          if ( vas->kind == NV_F64Imm ) {
            f.l_kv[rn->name] = *(uint64_t *)&this->m_d;
          } else if ( vas->kind == NV_F32Imm ) {
            float fl = (float)this->m_d;
            *(float *)&v = fl;
            f.l_kv[rn->name] = v;
          } else if ( vas->kind == NV_F16Imm ) {
            v = fp16_ieee_from_fp32_value(float(this->m_d));
            f.l_kv[rn->name] = v;
          }
        }
      }
      return 1;
    });
  } else return reduce(R_value);
}

// type - type of value in form_list
// ltype - type of label
// s - name of label, don't move it bcs it can be assigned to multiply of one_forms
int ParseSASS::reduce_label(int type, int ltype, std::string &s)
{
  auto apply_label = [&](one_form &f, auto &ci) {
    f.ltype = ltype;
    f.lname = s;
    f.lop = ci;
  };
  return apply_op2(m_forms, [&](form_list *fl, one_form &f, auto &ci) -> bool {
#ifdef DEBUG
 dump(fl, f.instr);
#endif
    // we can have R_value - and then must check name of value - it must be the same as instr->target_index
    // or we can have R_Cxx - then I don't know how to confirm if this is what I want
    if ( type == R_value && fl->rb->type == R_value )
    {
      if ( !f.instr->target_index ) {
       apply_label(f, ci);
       return 1;
      }
      // find vas
      const render_named *rn = (const render_named *)fl->rb;
      // there is strange case in RET/CALL where target_index is Ra and rn->name Ra_offset
      if ( !strcmp(f.instr->name, "RET") || !strcmp(f.instr->name, "CALL") ) {
        apply_label(f, ci);
        return 1;
      }
      int res = !strcmp(f.instr->target_index, rn->name);
      if ( res ) apply_label(f, ci);
      return res;
    }
    // const bank, perhaps I should check R_CX too?
    if ( type != R_value && fl->rb->type == R_C ) {
      apply_label(f, ci);
      return 1;
    }
    return 0;
   });
}

int ParseSASS::reduce_pred(const std::string_view &s, int exclamation)
{
#ifdef DEBUG
 printf("reduce_pred: %d ", exclamation); dump_outln(s);
#endif
  return apply_op(m_forms, [&](const form_list *fl, one_form &of) -> bool {
#ifdef DEBUG
 dump(fl, of.instr);
#endif
    if ( fl->rb->type != R_predicate && fl->rb->type != R_enum ) return 0;
    const render_named *rn = (const render_named *)fl->rb;
    auto ea = find_ea(of.instr, rn->name);
    if ( !ea ) return 0;
    // check if it has enum in s
    auto en = m_renums->find(ea->ename);
    if ( en == m_renums->end() ) return 0;
    auto aiter = en->second->find(s);
#ifdef DEBUG
 printf("en %d\n", aiter != en->second->end());
#endif
    if ( aiter == en->second->end() ) return 0;
    // store into l_kv
    of.l_kv[rn->name] = aiter->second;
    if ( exclamation ) {
      std::string not_name = rn->name;
      not_name += "@not";
      of.l_kv[not_name] = 1;
    }
    return 1;
   });
}

// fun fact - without ParseSASS:: prefix stupid gcc 12 says
//  error: ‘OFRights’ does not name a type
// wtf? it is obvious bug
template <typename C, typename F>
ParseSASS::OFRights ParseSASS::collect_rights(F &&f)
{
  OFRights res;
  std::for_each(m_forms.begin(), m_forms.end(), [&](one_form &of) {
     if ( !f((*of.current)->rb) ) return;
     const C *rc = (const C *)(*of.current)->rb;
     res.push_back( std::make_pair(&rc->right, &of) );
   });
  return res;
}

inline int is_msep(char c) {
 return isspace(c) || c == '+' || c == '.';
}

// try extract next enum after '+'
int ParseSASS::try_plus(const std::string_view &s, int start, int end, std::list<std::string_view> &elist)
{
  if ( s.at(start) == '-' ) { m_minus = 1; start++; } // [R6+-0x50] from sm_120
  if ( start + 2 < end && s.at(start) == '0' && s.at(start+1) == 'x' ) {
    parse_hex_tail(start + 2, s, 16);
    return 0; // hex
  }
  int digit = 1;
  int ti = start;
  for ( ; ti < end; ++ti ) {
    char c = s.at(ti);
    if ( c >= '0' && c <= '9' ) continue;
    digit = 0;
    if ( is_msep(c) ) {
      elist.push_back({ s.data() + start, size_t(ti - start) });
      return 1;
    }
  }
  if ( digit ) {
    parse_hex_tail(start, s, 10);
    return 0;
  }
  if ( ti == end ) {
    elist.push_back({ s.data() + start, size_t(end - start) });
    return 1;
  }
  return 0;
}

// try extract second enum ater '.'
int ParseSASS::parse_dot(const std::string_view &s, int start, int end, std::list<std::string_view> &elist)
{
  int i = start;
  for ( ; i < end; i++ )
  {
    char c = s.at(i);
     if ( is_msep(c) ) {
      elist.push_back({ s.data() + start, size_t(i - start) });
      if ( c == '+' ) try_plus(s, i + 1, end, elist);
      return 1;
     }
  }
  if ( i == end ) {
    elist.push_back({ s.data() + start, size_t(end - start) });
    return 1;
  }
  return 0;
}

template <typename C, typename F>
void ParseSASS::apply_mem_attrs(F &&f)
{
  if ( m_minus || m_abs ) {
   // store modifiers
   check_kind(m_forms, [&](const render_base *rb, one_form &of) -> bool {
     if ( !f(rb) ) return 0;
     const C *rc = (const C *)rb;
     if ( !rc->name ) return 0;
     if ( m_minus ) {
       std::string mname = rc->name;
       mname += "@negate";
       of.l_kv[mname] = 1;
     }
     if ( m_abs ) {
       std::string mname = rc->name;
       mname += "@absolute";
       of.l_kv[mname] = 1;
     }
     return 1;
   });
  }
}

// s - contains body after '[' (point by idx)
template <typename C, typename F>
int ParseSASS::parse_mem_right(int idx, const std::string_view &s, F &&f)
{
  if ( opt_d ) {
    printf("mem_right: "); dump_outln(s);
  }
  m_minus = 0;
  // find last ']' and check if we have tail like c[0x0] [0x8].H1
  int ri = idx;
  if ( s.at(idx) == '+' ) idx++;
  if ( s.at(idx) == '-' ) { m_minus = 1; idx++; }
  for ( ; ri < (int)s.size(); ++ri ) if ( s.at(ri) == ']' ) break;
  // check right part - if it contains number value
  int type = R_enum;
  std::string_view ename;
  std::list<std::string_view> enums;
  if ( s.at(idx) == '0' && s.at(idx+1) == 'x' ) {
   parse_hex_tail(idx + 2, s, 16);
   type = R_value;
  } else {
    int dig = 1;
    int ti = idx;
    for ( ; ti < ri; ++ti ) {
      char c = s.at(ti);
      if ( c >= '0' && c <= '9' ) continue;
      dig = 0;
      if ( is_msep(c) ) {
        ename = { s.data() + idx, size_t(ti - idx) };
        int need_add = 0;
        if ( c == '.' ) {
          need_add = parse_dot(s, ti + 1, ri, enums);
        } else if ( c == '+' ) {
          need_add = try_plus(s, ti + 1, ri, enums);
        }
        if ( need_add ) enums.push_front(ename);
        break;
      }
    }
    if ( dig ) {
      type = R_value;
      parse_hex_tail(idx, s, 10);
    } else if ( ti == ri ) ename = { s.data() + idx, size_t(ti - idx) };
    idx = ti;
  }
  if ( opt_d ) {
    if ( !enums.empty() ) printf("%ld enums\n", enums.size());
    else if ( type == R_enum )
    { printf("parse_mem_right enum: "); dump_outln(ename); }
  }
  // extract forms
  auto lf = collect_rights<C>(f);
  if ( opt_d ) {
    for ( auto &li: lf ) {
      std::string res;
      r_velisti(li.second->instr, *li.first, res);
      printf(" %d %s\n", li.second->instr->line, res.c_str());
    }
  }
  std::unordered_set<const nv_instr *> to_del;
  for ( auto &p: lf ) {
    int match = 0;
    if ( !enums.empty() ) {
      auto ei = enums.cbegin();
      for ( auto &vb: *p.first ) {
        if ( vb.type != R_enum ) {
          if ( ei != enums.cend() && *ei == "64"sv ) {
            if ( ++ei == enums.cend() ) match = 1;
          }
          break;
        }
        auto ea = find_ea(p.second->instr, vb.arg);
        if ( !ea ) break;
 /// printf("line %d enum %s\n", p.second->instr->line, vb.arg);
        if ( ei != enums.cend() ) {
          auto en = m_renums->find(ea->ename);
          if ( en == m_renums->end() ) break;
 /// printf("try find "); dump_outln(ename);
          auto aiter = en->second->find(*ei);
          if ( aiter != en->second->end() ) {
 /// printf("found "); dump_outln(ename);
           // store in l_kv
            p.second->l_kv[vb.arg] = aiter->second;
            if ( ++ei == enums.cend() ) match = 1;
 /// printf("next "); dump_outln(*ei);
            continue;
          }
        }
        if ( ea->has_def_value ) continue;
        match = 0;
        break;
      }
    } else if ( type == R_value ) {
      // check first item in p.first
      for ( auto &vb: *p.first ) {
       if ( vb.type == type ) {
         // patch l_kv here
         p.second->l_kv[vb.arg] = m_v;
         match = 1; break;
       }
       if ( vb.type != R_enum ) break;
       // check enum
       auto ea = find_ea(p.second->instr, vb.arg);
       if ( !ea ) break; // remove if no attr was found
       // check if this enum has default
       if ( ea->has_def_value ) continue;
       break;
     }
    } else if ( !ename.empty() )
    {
      for ( auto &vb: *p.first ) {
        if ( vb.type != R_enum ) break;
        auto ea = find_ea(p.second->instr, vb.arg);
        if ( !ea ) break;
        auto en = m_renums->find(ea->ename);
        if ( en == m_renums->end() ) break;
// printf("try find "); dump_outln(ename);
        auto aiter = en->second->find(ename);
        if ( aiter != en->second->end() ) {
          match = 1;
          // store in l_kv
          p.second->l_kv[vb.arg] = aiter->second;
          continue;
        }
        if ( ea->has_def_value ) continue;
        match = 0;
        break;
      }
    }
#ifdef DEBUG
 if ( !match ) {
  std::string rs; r_velisti(p.second->instr, *p.first, rs);
  printf("line %d type %d %d: %s\n", p.second->instr->line, type, match, rs.c_str());
 }
#endif
    if ( !match ) to_del.insert(p.second->instr);
  }
  if ( !to_del.empty() )
  {
    std::erase_if(m_forms, [&to_del](one_form &of) {
      auto di = to_del.find(of.instr);
      return di != to_del.end();
    });
    if ( m_forms.empty() ) return 0;
  }
  // here we filled l_kv for single R_value or enums values
  // however when we have some value and enums list - it's still not saved in l_kv
  if ( !enums.empty() && m_v ) {
    std::for_each(m_forms.begin(), m_forms.end(), [&](one_form &of) {
     if ( !of.instr->vas ) return;
     if ( !f((*of.current)->rb) ) return;
     const C *rc = (const C *)(*of.current)->rb;
     for ( auto &rci: rc->right ) {
       if ( rci.type == R_value ) {
         if ( set_num_value(find(of.instr->vas, rci.arg), rci.arg, of) ) break;
       }
     }
    });
  }
  m_minus = 0;
  constexpr bool has_name = requires(const C& t) {
    t.name;
  };
  if constexpr ( has_name ) {
    apply_mem_attrs<C>(f);
  }
  // check attrs - ri is pos of ']'
  if ( ri + 2 < (int)s.size() && s.at(ri + 1) == '.' )
    return process_tail_attr(ri + 1, s, m_forms);
  return !m_forms.empty();
}

// f - predicate for render_base filtering
// C - real type of current render
// idx - start of const bank after 'c['
template <typename C, typename F>
int ParseSASS::parse_c_left(int idx, const std::string &s, F &&f)
{
  int l_minus = 0;
  if ( s.at(idx) == '-' ) { l_minus = 1; idx++; }
  // find ]
  int li = idx;
  for ( ; li < (int)s.size(); ++li ) if ( s.at(li) == ']' ) break;
  // pre-classify what is it
  int type = R_enum;
  if ( s.at(idx) == '0' && s.at(idx+1) == 'x' ) {
   type = R_value;
   parse_hex_tail(idx + 2, s, 16);
  } else {
    int dig = 1;
    for ( auto ti = idx; ti < li; ++ti ) {
      char c = s.at(ti);
      if ( c >= '0' && c <= '9' ) continue;
      dig = 0;
      break;
    }
    if ( dig ) {
     type = R_value;
     parse_hex_tail(idx, s, 10);
    }
  }
#ifdef DEBUG
 printf("type %d idx %d %s\n", type, idx, s.c_str() + idx);
#endif
  std::string_view ename;
  if ( type == R_enum ) {
    ename = { s.c_str() + idx, (size_t)(li - idx) };
    if ( opt_d ) {
      printf("c_left: "); dump_outln(ename);
    }
  }
  auto my_cl = [&](one_form &of) -> bool
   {
     if ( !f((*of.current)->rb) ) return 1;
     const C *rc = (const C *)(*of.current)->rb;
     if ( rc->left.type != type ) return 1;
     if ( type == R_value ) {
       // it's safe to fill l_kv here bcs non-matched forms will be just erased
       long l = m_v;
       if ( l_minus ) l = -l;
       of.l_kv[rc->left.arg] = l;
     }
     if ( type != R_enum ) return 0;
     // check enum
     auto ea = find_ea(of.instr, rc->left.arg);
     if ( !ea ) return 1; // remove if no attr was found
     auto en = m_renums->find(ea->ename);
     if ( en == m_renums->end() ) return 1;
     auto aiter = en->second->find(ename);
     if ( aiter != en->second->end() ) {
       of.l_kv[rc->left.arg] = aiter->second;
       return 0;
     }
     return 1; // enum not found
   };
  std::erase_if(m_forms, my_cl);
  if ( m_forms.empty() ) return 0;
  // check if render of type C has field 'right'
  // from https://stackoverflow.com/questions/257288/how-can-you-check-whether-a-templated-class-has-a-member-function
  constexpr bool has_right = requires(const C& t) {
    t.right;
  };
  constexpr bool has_name = requires(const C& t) {
    t.name;
  };
  if constexpr ( has_name ) {
    apply_mem_attrs<C>(f);
  }
  // reset modifiers
  m_minus = m_abs = 0;
  std::string_view tail;
  if constexpr ( has_right )
  {
    // find right part
    for ( ; li < (int)s.size(); ++li ) if ( s.at(li) == '[' ) break;
    if ( li >= (int)s.size() ) return 1; // bad format?
    tail = { s.c_str() + li + 1, (size_t)(s.size() - li - 1) };
    return parse_mem_right<C>(0, tail, f);
  } else {
    // check if we have attributes, li holds pos of closing ']'
    if ( li + 2 < (int)s.size() && s.at(li + 1) == '.' )
    {
      tail = { s.c_str() + li + 1, (size_t)(s.size() - li - 1) };
      if ( opt_d ) {
        printf("enum tail: "); dump_outln(tail);
      }
      if ( !tail.empty() ) return process_tail_attr(0, tail, m_forms);
    }
  }
  return !m_forms.empty();
}

static const std::string_view s_bt = "(*\"BRANCH_TARGETS"sv;
static const std::string_view s_ic = "(*\"INDIRECT_CALL\"*)"sv;
static std::string s_empty = "";

// main horror - try to detect what dis op is
// sass grammar is not pure regular - some operands separated by space - especially labels, like
//  BRX R2 -0x110 (*"INDIRECT_CALL"*)
// here first call will be apply_enum
// then enum_tail calls classify_op again - this time it will classify op -0x110 as hex and will call classify_op
// yet one time after parse_hex_tail
// So this method is recursive with max depth (hopefully) 3
// Also to avoid operands copying use string_view - they are much cheaper than std::string
int ParseSASS::classify_op(int op_idx, const std::string_view &os)
{
  reset_v();
  auto spaces = os.find_first_not_of(' ');
  std::string s{ spaces == std::string::npos ? os.begin() : os.begin() + spaces, os.end() };
#ifdef DEBUG
 printf("op %d %s\n", op_idx, s.c_str());
 dump_forms();
#endif
  int idx = 0;
  char c = s.at(idx);
  if ( c == '-' ) { m_minus = 1; c = s.at(++idx); }
  else if ( c == '+' ) c = s.at(++idx);
  else if ( c == '~' ) { m_tilda = 1; c = s.at(++idx); };
  std::string_view tmp{ s.c_str() + idx, s.size() - idx};
  if ( tmp == "INF"sv || tmp == "inf"sv ) { m_numv = NumV::inf; return reduce(R_value); }
  if ( tmp == "QNAN"sv || tmp == "nan"sv ) { m_numv = NumV::nan; return reduce(R_value); }
  auto cl = [](const render_base *rb) { return rb->type == R_C || rb->type == R_CX; };
  if ( tmp.starts_with("desc["sv) ) {
    auto dcl = [](const render_base *rb) { return rb->type == R_desc; };
    int kres = apply_kind(m_forms, dcl);
    if ( !kres ) return 0;
    return parse_c_left<render_desc>(idx + 5, s, dcl);
  }
  if ( tmp.starts_with("a["sv) ) {
    auto da = [](const render_base *rb) { return rb->type == R_mem; };
    int kres = apply_kind(m_forms, da);
    if ( !kres ) return 0;
    return parse_mem_right<render_mem>(idx + 2, s, da);
  }
  if ( tmp.starts_with("ttu["sv) ) {
    auto dttu = [](const render_base *rb) { return rb->type == R_TTU; };
    int kres = apply_kind(m_forms, dttu);
    if ( !kres ) return 0;
    return parse_c_left<render_TTU>(idx + 4, s, dttu);
  }
  if ( tmp.starts_with("c["sv) ) {
    int kres = apply_kind(m_forms, cl);
    if ( !kres ) return 0;
    return parse_c_left<render_C>(idx + 2, s, cl);
  }
  if ( tmp.starts_with("cx["sv) ) {
    int kres = apply_kind(m_forms, cl);
    if ( !kres ) return 0;
    return parse_c_left<render_C>(idx + 3, s, cl);
  }
  if ( tmp.starts_with("0x"sv) ) {
   // hex value + possible tail for label
   idx = parse_hex_tail(2, tmp, 16);
   if ( !reduce_value() ) return 0;
   if ( idx < (int)tmp.size() ) {
     std::string_view next{ tmp.data() + idx, size_t(tmp.size() - idx) };
     return classify_op(op_idx + 1, next);
   }
   return 1;
  }
  if ( tmp.starts_with(s_bt) ) {
    auto bt_name = extract_label(s_bt.size(), tmp);
    if ( has_target(&m_forms) )
      return reduce_label(R_value, BRANCH_TARGET, bt_name);
    else
      return mark_label(BRANCH_TARGET, bt_name);
  }
  switch(c) {
    case '`': if ( s.at(idx+1) != '(' ) {
       fprintf(stderr, "unknown op %d: %s\n", op_idx, s.c_str());
       return 0;
     } else {
       auto lname = extract_label(idx + 2, s);
       if ( has_target(&m_forms) ) {
         if ( opt_d ) printf("` has targets, try R_value\n");
       } else {
         if ( opt_d ) printf("unknown target operand %d: %s\n", op_idx, s.c_str());
       }
       return reduce_label(R_value, LABEL, lname);
     }
     break;
    case '!': return reduce_pred({ s.c_str() + idx + 1, s.size() - 1 - idx}, 1);
    case '|':
     m_abs = 1;
     if ( !tmp.ends_with("|") ) {
       // surprise - there can be ops like |R13|.reuse
       // so try to find second |
       int ip = idx + 1;
       for ( ; ip < (int)tmp.size(); ip++ ) if ( tmp.at(ip) == '|' ) break;
       if ( ip == (int)tmp.size() ) {
         fprintf(stderr, "bad operand %d: %s\n", op_idx, s.c_str());
         return 0;
       }
       // remained attributes start at s + ip + 1
       std::string_view abs{ s.c_str() + idx + 1, size_t(ip - idx - 1)};
#ifdef DEBUG
 printf("piped: len %d ", ip - idx - 1); dump_outln(abs);
#endif
       int eres = apply_enum(abs);
       if ( !eres ) return eres;
       if ( ip + 1 < (int)s.size() ) {
         std::string tmp{ s.begin() + ip + 1, s.end() };
#ifdef DEBUG
 printf("after | %s\n", tmp.c_str());
#endif
         eres = enum_tail(0, tmp);
       }
       return eres;
     } else {
       // check what is dis
       std::string_view abs{ s.c_str() + idx + 1, tmp.size() - 2};
       if ( abs.starts_with("c["sv) ) {
         int kres = apply_kind(m_forms, cl);
         if ( !kres ) return 0;
         return parse_c_left<render_C>(idx + 3, s, cl); // 1 - | + 2 - c[
       } else return apply_enum(abs);
     }
    case '{': // hopefully this is bitset for DEPBAR
     if ( parse_bitset(idx + 1, tmp) ) return reduce_value();
     return reduce(R_value);
    case '[': {
      auto dm = [](const render_base *rb) { return rb->type == R_mem; };
      int kres = apply_kind(m_forms, dm);
      if ( !kres ) return 0;
      return parse_mem_right<render_mem>(idx + 1, s, dm);
    }
  }
  // INDIRECT_CALL
  if ( tmp.starts_with(s_ic) ) return mark_label(INDIRECT_CALL, s_empty);
  // 32@lo( & 32@hi
  if ( tmp.starts_with("32@lo("sv) ) {
    c = tmp.at(6);
    auto lname = extract_label(c == '(' ? 7 : 6, tmp);
    return reduce_label(R_value, L32, lname);
  }
  if ( tmp.starts_with("32@hi("sv) ) {
    c = tmp.at(6);
    auto lname = extract_label(c == '(' ? 7: 6, tmp);
    return reduce_label(R_value, H32, lname);
  }
  // check for digit
  int dig = 1, cnt = 0, was_dot = 0;
  for ( auto ti = tmp.cbegin(); ti != tmp.cend(); ++ti ) {
    c = *ti;
    if ( c >= '0' && c <= '9' ) { cnt++; continue; }
    if ( c == '.' ) { if ( !was_dot ) { ++was_dot; continue; } }
    if ( cnt && c == 'e' ) break;
    dig = 0;
    break;
  }
  if ( dig && !was_dot ) {
   // decimal value + possible tail for label
   idx = parse_hex_tail(0, tmp, 10);
   if ( !reduce_value() ) return 0;
   if ( idx < (int)tmp.size() ) {
     std::string_view next{ tmp.data() + idx, size_t(tmp.size() - idx) };
     return classify_op(op_idx + 1, next);
   }
   return 1;
  }
  if ( dig ) {
   idx = parse_float_tail(0, tmp);
   if ( !reduce_value() ) return 0;
   if ( idx < (int)tmp.size() ) {
     std::string_view next{ tmp.data() + idx, size_t(tmp.size() - idx) };
     return classify_op(op_idx + 1, next);
   }
   return 1;
  }
  // check for some unknown prefix for memory
  for ( int pi = idx + 1; pi < (int)tmp.size(); ++pi ) {
    if ( '[' == tmp.at(pi) ) {
      fprintf(stderr, "[!] unknown memory prefix: "); dump_outln(tmp, stderr);
      return 0;
    }
  }
  // will hope this is enum
  return apply_enum(tmp);
}

std::string ParseSASS::process_tail(int idx, const std::string &s, NV_Forms &f)
{
  std::string res;
  int state = 0;
  for ( int i = idx; i < (int)s.size(); ) {
    auto c = s.at(i);
#ifdef DEBUG
 printf("state %d i %d %c\n", state, i, c);
#endif
    if ( !state ) {
      if ( c == '&' ) state = 1;
      else if ( c == '?' ) state = 2;
      else {
        res.push_back(c); i++; continue;
      }
      i++;
    }
    if ( 2 == state ) {
      if ( !usched ) break;
      // check if this enum exists
      std::string ename;
      std::copy_if( s.begin() + i, s.end(), std::back_inserter(ename), [](char c) { return !isspace(c) && c != '}';  });
      auto ei = usched->find(ename);
      if ( ei == usched->end() ) {
        printf("[!] unknown sched %s\n", ename.c_str());
        break;
      }
      // update kv
      m_kv[c_usched_name] = ei->second;
      break; // bcs ?usched is always last
    }
    // check &something=
    if ( 1 == state ) {
      int value = 0;
      std::string_view tmp{ s.c_str() + i, s.size() - idx };
      if ( tmp.starts_with("req={") ) {
        i += 5 + parse_req(s.c_str() + 5 + i);
      } else if ( tmp.starts_with("wr=") ) {
        i += 3 + parse_digit(s.c_str() + 3 + i, value);
        m_kv["dist_wr_sb"] = value;
      } else if ( tmp.starts_with("rd=") ) {
        i += 3 + parse_digit(s.c_str() + 3 + i, value);
        m_kv["src_rel_sb"] = value;
      }
      else {
        printf("unknown tail %s\n", s.c_str() + i);
        break;
      }
      state = 3;
      continue;
    }
    // check symbol at tail
    if ( c == '&' ) state = 1;
    else if ( c == '?' ) state = 2;
    else if ( !isspace(c) ) {
       printf("unknown symbol '%c' in tail %s\n", c, s.c_str() + i);
       break;
    }
    i++;
  }
  rstrip(res);
  return res;
}

template <typename T>
int ParseSASS::try_dotted(int idx, T &s, std::string_view &dotted, int &dotted_last)
{
  int last;
  dotted_last = 0;
  for ( last = idx; last < (int)s.size(); ++last ) {
    auto c = s.at(last);
    if ( isspace(c) ) break;
    if ( c == '.' ) {
      if ( !m_dotted )
        break;
      // check if this constant contains '.'
      int len = last - idx + 1;
      std::string_view tmp( s.data() + idx, len );
      auto di = m_dotted->lower_bound(tmp);
      if ( di == m_dotted->end() ) break;
      if ( !(*di).starts_with(tmp) ) break;
#ifdef DEBUG
dump_out(tmp); printf(" -> "); dump_outln(*di);
#endif
      int i2 = 1 + last;
      for ( ; i2 < (int)s.size(); ++i2, ++len ) {
        auto c = s.at(i2);
        if ( isspace(c) || c == '.' ) break;
      }
      // check in dotted
      dotted = { s.data() + idx, (size_t)len };
// fputc('>', stdout); dump_outln(dotted);
      di = m_dotted->find(dotted);
      if ( di != m_dotted->end() ) {
        dotted_last = i2;
// dump_out(dotted); printf(" %d-> ", last); dump_outln(*di);
      }
      break;
    }
  }
  return last;
}

int ParseSASS::apply_enum(const std::string_view &s)
{
  int last, dotted_last = 0;
  std::string_view dotted;
  last = try_dotted(0, s, dotted, dotted_last);
  std::string_view ename(s.begin(), last);

  if ( opt_d ) {
    printf("apply_enum "); dump_out(ename); printf(" last %d dlast %d\n", last, dotted_last);
  }
  if ( dotted_last ) {
    if ( check_op(m_forms, [&](const form_list *fl, const nv_instr *instr) -> bool {
      if ( fl->rb->type != R_predicate && fl->rb->type != R_enum ) return 0;
      const render_named *rn = (const render_named *)fl->rb;
      auto ea = find_ea(instr, rn->name);
      if ( !ea ) return 0;
      // check if it has enum in s
      auto en = m_renums->find(ea->ename);
      if ( en == m_renums->end() ) return 0;
      auto aiter = en->second->find(dotted);
      return aiter != en->second->end();
     }) ) {
     ename = dotted;
     last = dotted_last;
     if ( opt_d ) {
       printf("found dotted "); dump_outln(ename);
     }
   }
  }
  if ( opt_d ) dump_forms();
  int res = apply_op(m_forms, [&](const form_list *fl, one_form &of) -> bool {
    if ( opt_d ) {
      std::string res;
      rend_single(fl->rb, res); printf(" %s\n", res.c_str());
    }
    if ( fl->rb->type != R_predicate && fl->rb->type != R_enum ) return 0;
    const render_named *rn = (const render_named *)fl->rb;
    auto ea = find_ea(of.instr, rn->name);
    if ( !ea ) return 0;
#ifdef DEBUG
  printf("found ei %s", ea->ename);
#endif
    // check if it has enum in s
    auto en = m_renums->find(ea->ename);
    if ( en == m_renums->end() ) return 0;
#ifdef DEBUG
  printf("en "); dump_outln(ename);
#endif
    auto aiter = en->second->find(ename);
    if ( aiter == en->second->end() ) return 0;
    // store in local kv
    of.l_kv[rn->name] = aiter->second;
    // and attributes
    if ( m_tilda ) {
      std::string tname = rn->name;
      tname += "@invert";
      of.l_kv[tname] = 1;
    }
    if ( m_abs ) {
      std::string tname = rn->name;
      tname += "@absolute";
      of.l_kv[tname] = 1;
    }
    if ( m_minus ) {
      std::string tname = rn->name;
      tname += "@negate";
      of.l_kv[tname] = 1;
    }
    return 1;
  });
  if ( !res ) return res;
  // reset modifiers
  m_tilda = m_abs = m_minus = 0;
  // tail
  if ( last < (int)s.size() ) res = enum_tail(last, s);
  return res;
}

// idx - index of '.' at start of attr
int ParseSASS::process_tail_attr(int idx, const std::string_view &s, NV_Forms &f)
{
  int last, dotted_last = 0;
  std::string_view dotted;
  last = try_dotted(++idx, s, dotted, dotted_last);
  std::string_view ename(s.data() + idx, last - idx);
  int found = 0;
  if ( dotted_last ) {
    // if we have some enum with '.' - check it but don't remove forms
    std::for_each(f.begin(), f.end(), [&](const one_form &of) {
    if ( (*of.current)->empty() ) return;
    for ( auto &a: (*of.current)->lr ) {
      if ( !a.en ) continue;
      auto aiter = a.en->find(dotted);
      if ( aiter != a.en->end() ) { found++; return; }
    } });
    if ( found ) {
      // yes, we can proceed with dotted enum
      last = dotted_last;
      ename = dotted;
#ifdef DEBUG
 printf("%d last %d>", found, last); dump_outln(ename);
#endif
      found = 0;
    }
  }
  // iterate on all remained forms and try to find this attr at theirs current operand
  std::erase_if(f, [&](one_form &of) {
    if ( (*of.current)->empty() ) return 1;
    for ( auto &a: (*of.current)->lr ) {
      if ( !a.en ) continue;
      const render_named *rn = (const render_named *)a.first;
      auto ki = of.l_kv.find(rn->name);
      if ( ki != of.l_kv.end() ) continue;
      auto aiter = a.en->find(ename);
      if ( aiter != a.en->end() ) {
       // insert name of this enum into m_kv
       of.l_kv[rn->name] = aiter->second;
       return 0;
      }
    }
    return 1;
  });
  return last;
}

// idx - index of '.' at start of attr
int ParseSASS::process_attr(int idx, const std::string &s, NV_Forms &f)
{
  int last, dotted_last = 0;
  std::string_view dotted;
  last = try_dotted(++idx, s, dotted, dotted_last);
  std::string_view ename(s.c_str() + idx, last - idx);
#ifdef DEBUG
 printf("attr %s len %d\n", s.c_str() + idx, last - idx);
#endif
  if ( !dotted_last && pseudo ) {
    auto pi = pseudo->find(ename);
    if ( pi != pseudo->end() ) return last;
  }
  int found = 0;
  if ( dotted_last ) {
    // if we have some enum with '.' - check it but don't remove forms
    std::for_each(f.begin(), f.end(), [&](const one_form &of) {
    if ( (*of.current)->empty() ) return;
    for ( auto &a: (*of.current)->lr ) {
      if ( !a.en ) continue;
      auto aiter = a.en->find(dotted);
      if ( aiter != a.en->end() ) { found++; return; }
    } });
    if ( found ) {
      // yes, we can proceed with dotted enum
      last = dotted_last;
      ename = dotted;
#ifdef DEBUG
 printf("%d last %d>", found, last); dump_outln(ename);
#endif
      found = 0;
    }
  }
  // iterate on all remained forms and try to find this attr at thers current operand
  std::erase_if(f, [&](one_form &of) {
    if ( (*of.current)->empty() ) return 1;
    for ( auto &a: (*of.current)->lr ) {
      if ( !a.en ) continue;
      const render_named *rn = (const render_named *)a.first;
      auto ki = of.l_kv.find(rn->name);
      if ( ki != of.l_kv.end() ) continue;
      auto aiter = a.en->find(ename);
      if ( aiter != a.en->end() ) {
       // insert name of this enum into m_kv
       of.l_kv[rn->name] = aiter->second;
       return 0;
      }
    }
    return 1;
  });
  return last;
}

int ParseSASS::fill_forms(NV_Forms &forms, const std::vector<const nv_instr *> &mv)
{
  int res = 0;
  for ( auto ins: mv ) {
    auto r = m_dis->get_rend(ins->n);
    if ( !r ) continue;
    auto ri = r->begin();
    if ( has_pred() && (*ri)->type != R_predicate ) continue;
    if ( (*ri)->type == R_predicate ) ++ri;
    if ( (*ri)->type != R_opcode ) continue;
    one_form of(ins, r);
    // dissect render - ri holds R_opcode and pushed into of
    of.ops.push_back( new form_list(*ri) );
    for ( ++ri; ri != r->end(); ++ri ) {
      const render_named *rn = nullptr;
      switch((*ri)->type) {
        case R_value: { // check for bitmap
          rn = (const render_named *)*ri;
          auto vi = find(ins->vas, rn->name);
          // need to check name of field too bcs we can have DEPBAR with dep_scbd having type NV_BITSET too
          if ( is_tail(vi, rn) ) goto out;
          of.ops.push_back( new form_list(*ri) );
        }
        break;
        case R_enum: {
          rn = (const render_named *)*ri;
          const nv_eattr *ea = find_ea(ins, rn->name);
          if ( ea && ea->ignore ) { // push it into last
            auto en = m_renums->find(ea->ename);
            LTuple l{ *ri, ea, en != m_renums->end() ? en->second : nullptr };
            of.ops.back()->lr.push_back( l );
            break;
          }
        } // notice - no break here
        default:
         of.ops.push_back( new form_list(*ri) );
      }
    }
out:
    // finally put into forms
    forms.push_back(std::move(of));
    res++;
  }
  std::for_each( forms.begin(), forms.end(), [](one_form &of) { of.current = of.ops.begin(); });
  return res;
}

int ParseSASS::enum_tail(int idx, const std::string_view &head)
{
  if ( opt_d ) {
    printf("enum_tail: "); for ( int i = idx; i < (int)head.size(); i++ ) fputc(head.at(i), stdout); fputc('\n', stdout);
  }
  for( int a_n = 0; idx < (int)head.size(); ++a_n )
  {
    auto c = head.at(idx);
    if ( c == '.' ) {
      idx = process_tail_attr(idx, head, m_forms);
      if ( m_forms.empty() ) {
       std::string_view tmp{ head.data() + idx, size_t(head.size() - idx) };
       printf("[!] unknown attr "); dump_out(tmp); printf(" after enum_tail %d\n", a_n);
       return 0;
      }
    } else if ( c == ' ' ) {
      if ( !next(m_forms) ) return 0;
      std::string_view tmp{ head.data() + idx + 1, size_t(head.size() - idx - 1) };
      if ( opt_d ) {
        printf("call classify_op "); dump_outln(tmp);
        dump_forms();
      }
      return classify_op(0, tmp);
    } else {
      std::string_view tmp{ head.data() + idx, size_t(head.size() - idx) };
      printf("[!] unknown tail "); dump_out(tmp); printf(" after enum_tail %d\n", a_n);
      return 0;
    }
  }
  return 1;
}

int ParseSASS::add(const std::string &s, int idx)
{
  int ares = add_internal(s, idx);
  if ( !ares ) return 0;
  if ( skip_final_cut ) return ares;
  // final cut
  // there is big problem - when remained forms set is too small (or contains IMAD) - usually it removes
  // all bcs some enums linked to opcode not presents in l_kv
  // so trick is to try first non-relaxed mode and if there are no forms - set relax and do real removing
  bool relax = 0;
  int matched = 0;
  auto try_upto = [&](one_form &of) {
     // check opcode operand
     for ( auto cold = of.ops.begin(); ; ++cold ) {
      if ( !(*cold)->empty() ) {
       for ( auto &a: (*cold)->lr ) {
         if ( a.second->has_def_value ) continue;
         // special case - check if a.second->en has exactly single value
         // first pass relax is false so em->size is not even called
         if ( relax && a.second->em->size() == 1 ) continue;
         const render_named *rn = (const render_named *)a.first;
         auto ki = of.l_kv.find(rn->name);
         if ( ki == of.l_kv.end() ) {
//           dump(of); printf("%d no %s\n", of.instr->line, rn->name);
           return 1;
         }
       }
      }
      if ( cold == of.current ) break;
    }
    // if we here - this form was perfectly matched
    matched++;
    return 0;
  };
  // check matches
  std::for_each(m_forms.begin(), m_forms.end(), try_upto);
  // if no matches - set relax
  if ( !matched ) relax = true;
  // now perform real erasing
  std::erase_if(m_forms, [&](one_form &of) {
     if ( try_upto(of) ) return 1;
     if ( of.current == of.ops.end() ) return 0;
     auto ci = of.current;
     // check if we have some operands without default values behind last processed
     for ( ++ci; ci != of.ops.end(); ++ci ) {
       if ( (*ci)->rb->type == R_value ) continue;
       if ( (*ci)->rb->type == R_predicate || (*ci)->rb->type == R_enum ) {
         // check if those predicate has default
         const render_named *rn = (const render_named *)(*ci)->rb;
         auto ea = find_ea(of.instr, rn->name);
         if ( ea && ea->has_def_value ) continue;
       }
       return 1;
     }
     return 0;
   });
  return !m_forms.empty();
}

int ParseSASS::add_internal(const std::string &s, int idx)
{
  reset_pred();
  // skip initial spaces
  for ( ; idx < (int)s.size(); idx++ ) if ( !isspace(s.at(idx)) ) break;
  // check { for dual-issued instructions
  if ( m_width == 88 && s.at(idx) == '{' ) {
    m_kv[c_usched_name] = 0x10; // see https://redplait.blogspot.com/2025/04/nvidia-sass-disassembler-part-7-dual.html
    for ( ++idx; idx < (int)s.size(); idx++ ) if ( !isspace(s.at(idx)) ) break;
  }
  // check predicate
  if ( s.at(idx) == '@' ) idx = parse_pred(idx, s);
  // extract mnemonic
  std::string mnem;
  for ( ; idx < (int)s.size(); idx++ ) {
    auto c = s.at(idx);
    if ( isspace(c) || c == '.' || c == ',' ) break;
    mnem.push_back(c);
  }
  // try to find mnemonic
  auto mv = std::lower_bound( m_sorted->begin(), m_sorted->end(), mnem, [](const auto &pair, const std::string &w) {
    return pair.first < w;
   });
  if ( mv == m_sorted->end() ) {
    printf("[!] cannot find mnemonic %s\n", mnem.c_str());
    return 0;
  }
  // ok, lets construct forms array
  m_forms.clear();
  fill_forms(m_forms, mv->second);
  if ( m_forms.empty() ) return 0;
  if ( idx >= (int)s.size() ) return 1;
  // process first tail with rd/wr etc
  std::string head = process_tail(idx, s, m_forms);
  if ( head.empty() ) {
    if ( !m_forms.empty() ) return 1;
    printf("[!] unknown form %s after process_tail\n", s.c_str());
    return 0;
  }
  idx = 0;
  if ( opt_d ) {
    printf("before operands processing:\n");
    dump_forms();
  }
  for( int attr_n = 0; idx < (int)head.size(); ++attr_n )
  {
    auto c = head.at(idx);
    if ( c == '.' ) {
    // some attr
    idx = process_attr(idx, head, m_forms);
    if ( m_forms.empty() ) {
      // surprise - there is mnemonics like UIADD.64
      if ( !attr_n ) {
        std::string second_mnem(head, idx);
        mnem += second_mnem;
        mv = std::lower_bound( m_sorted->begin(), m_sorted->end(), mnem, [](const auto &pair, const std::string &w) {
         return pair.first < w;
        });
        if ( mv != m_sorted->end() ) {
          if ( fill_forms(m_forms, mv->second) ) continue;
        }
      }
      printf("[!] unknown form %s after process_attr %d\n", head.c_str(), attr_n);
      return 0;
     }
    } else if ( c == ' ' ) {
      if ( !next(m_forms) ) return 0;
      idx++; break;
    } else {
      printf("[!] cannot parse %s\n", head.c_str() + idx);
      return 0;
    }
  }
  // we have set of opcodes in head
  if ( !skip_op_parsing ) {
    std::cregex_token_iterator begin(head.c_str() + idx, head.c_str() + head.size(), s_commas, -1), end;
    int op_idx = 0;
    for ( auto op = begin; op != end; ++op, ++op_idx ) {
      auto s = *op;
      if ( !s.length() ) continue;
      if ( op_idx ) { // first next was issued in head processing after first space
        if ( !next(m_forms) ) return 0;
      }
      std::string_view tmp{ s.first, s.second };
      classify_op(op_idx, tmp);
      if ( m_forms.empty() ) {
        printf("[!] empty after %d op: %s\n", op_idx, head.c_str() + idx);
        break;
      }
    }
  }
  if ( m_forms.empty() ) return 0;
  return 1;
}

int ParseSASS::next(NV_Forms &f) const
{
  std::erase_if(f, [](one_form &of) { return ++of.current == of.ops.end(); });
  return !f.empty();
}

int ParseSASS::init_guts()
{
  m_sorted = m_dis->get_instrs();
  m_renums = m_dis->get_renums();
  if ( !m_renums ) {
    fprintf(stderr, "get_renums failed\n");
    return 0;
  }
  auto ri = m_renums->find("USCHED_INFO");
  if ( ri != m_renums->end() ) usched = ri->second;
  else {
    fprintf(stderr, "cannot find usched_info enum\n");
  }
  ri = m_renums->find("PSEUDO_OPCODE");
  if ( ri != m_renums->end() ) pseudo = ri->second;
  else {
    fprintf(stderr, "cannot find pseudo_opcode enum\n");
  }
  m_dotted = m_dis->get_dotted();
  if ( !m_dotted )
    fprintf(stderr, "cannot find dotted enums\n");
  return 1;
}
