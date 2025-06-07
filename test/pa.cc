#include <fstream>
#include "nv_rend.h"
#include <regex>
#include <unistd.h>

int opt_m = 0,
    opt_s = 0,
    opt_o = 0,
    opt_v = 0;

// for sv literals
using namespace std::string_view_literals;

class ParseSASS: public NV_renderer
{
  public:
   ParseSASS(): NV_renderer()
   { }
   int init(const std::string &s);
   int add(const std::string &s);
   int print_fsummary(FILE *) const;
  protected:
   struct form_list {
     form_list(const render_base *_rb) {
       rb = _rb;
     }
     inline bool empty() const {
       return lr.empty();
     }
     const render_base *rb = nullptr;
     std::list<std::pair<const render_base *, const nv_eattr *> > lr;
   };
   struct one_form
   {
     const nv_instr *instr;
     const NV_rlist *rend;
     std::list<form_list *> ops;
     std::list<form_list *>::iterator current;
     one_form(const nv_instr *_i, const NV_rlist *_r) {
       instr = _i;
       rend = _r;
     }
     ~one_form() {
       for ( auto o: ops ) delete o;
     }
     one_form& operator=(one_form&& other) = default;
     one_form(one_form&& other) = default;
   };
   void dump(const one_form &);
   typedef std::vector<one_form> NV_Forms;
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
   int check_kind(NV_Forms &forms, F pred) {
     int res = 0;
     for ( auto &f: forms )
     {
       for ( auto ci = f.current; ci != f.ops.end(); ++ci )
       {
         if ( pred((*ci)->rb) ) { res++; break; }
         if ( (*ci)->rb->type == R_predicate ) {
           // check if those predicate has default
           const render_named *rn = (const render_named *)(*ci)->rb;
           auto ei = find(f.instr->eas, rn->name);
           if ( !ei ) break;
           if ( !ei->ea->has_def_value ) break;
           continue;
         }
         break;
       }
     }
     return res;
   }
   template <typename F>
   int check_op(NV_Forms &forms, F pred) {
     int res = 0;
     for ( auto &f: forms )
     {
       for ( auto ci = f.current; ci != f.ops.end(); ++ci )
       {
         if ( pred((*ci), f.instr) ) { res++; break; }
         if ( (*ci)->rb->type == R_predicate ) {
           // check if those predicate has default
           const render_named *rn = (const render_named *)(*ci)->rb;
           auto ei = find(f.instr->eas, rn->name);
           if ( !ei ) break;
           if ( !ei->ea->has_def_value ) break;
           continue;
         }
         break;
       }
     }
     return res;
   }
   template <typename F>
   int apply_kind(NV_Forms &f, F pred) {
     std::erase_if(f, [&](one_form &f) {
       for ( auto ci = f.current; ci != f.ops.end(); ci++ )
       {
         if ( pred((*ci)->rb) ) { f.current = ci; return 0; }
         if ( (*ci)->rb->type == R_predicate ) {
           // check if those predicate has default
           const render_named *rn = (const render_named *)(*ci)->rb;
           auto ei = find(f.instr->eas, rn->name);
           if ( !ei ) break;
           if ( !ei->ea->has_def_value ) break;
           continue;
         }
         return 1;
       }
       return 1;
     });
     return !f.empty();
   }
   template <typename F>
   int apply_op(NV_Forms &f, F pred) {
     std::erase_if(f, [&](one_form &f) {
       for ( auto ci = f.current; ci != f.ops.end(); ci++ )
       {
         if ( pred((*ci), f.instr) ) { f.current = ci; return 0; }
         if ( (*ci)->rb->type == R_predicate ) {
           // check if those predicate has default
           const render_named *rn = (const render_named *)(*ci)->rb;
           auto ei = find(f.instr->eas, rn->name);
           if ( !ei ) break;
           if ( !ei->ea->has_def_value ) break;
           continue;
         }
         return 1;
       }
       return 1;
     });
     return !f.empty();
   }

   int parse_req(const char *s);
   int parse_digit(const char *s, int &v);
   int parse_pred(const std::string &s);
   std::string process_tail(int idx, const std::string &s, NV_Forms &);
   int process_attr(int idx, const std::string &s, NV_Forms &);
   template <typename T>
   int try_dotted(int, T &, std::string_view &dotted, int &dotted_last);
   int classify_op(int idx, const std::string &s);
   int reduce(int);
   int reduce_enum(const std::string_view &);
   int reduce_pred(const std::string_view &);
   int apply_enum(const std::string_view &);
   NV_Forms m_forms;
   // currently kv
   NV_extracted m_kv;
   static std::regex s_digits;
   static std::regex s_commas;
   static constexpr auto c_usched_name = "usched_info";
   const NV_sorted *m_sorted = nullptr;
   const NV_Renums *m_renums = nullptr;
   const NV_Renum *usched = nullptr;
   const NV_Renum *pseudo = nullptr;
   const NV_dotted *m_dotted = nullptr;
};

std::regex ParseSASS::s_digits("\\d+");
std::regex ParseSASS::s_commas("\\s*,\\s*");

void ParseSASS::dump(const one_form &of)
{
  printf("%s line %d:", of.instr->name, of.instr->line);
  for ( auto ops = of.current; ops != of.ops.end(); ++ops ) {
    std::string res;
    rend_single(*(*ops)->rb, res);
    printf(" %s", res.c_str());
    if ( (*ops)->rb->type == R_predicate || (*ops)->rb->type == R_enum ) {
     // check if those predicate has default
     const render_named *rn = (const render_named *)(*ops)->rb;
     auto ei = find(of.instr->eas, rn->name);
     if ( !ei ) continue;
     if ( ei->ea->has_def_value ) printf(".D(%d):%s", ei->ea->def_value, ei->ea->ename);
    }
  }
  fputc('\n', stdout);
}

int ParseSASS::parse_pred(const std::string &s)
{
  if ( s.at(0) != '@' ) return 0;
  int res = 1;
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

int ParseSASS::reduce(int kind)
{
  auto cl = [kind](const render_base *rb) { return rb->type == kind; };
  return apply_kind(m_forms, cl);
}

int ParseSASS::reduce_pred(const std::string_view &s)
{
  return apply_op(m_forms, [&](const form_list *fl, const nv_instr *instr) -> bool {
    if ( fl->rb->type != R_predicate ) return 0;
    const render_named *rn = (const render_named *)fl->rb;
    auto ei = find(instr->eas, rn->name);
    if ( !ei ) return 0;
    // check if it has enum in s
    auto en = m_renums->find(ei->name);
    if ( en == m_renums->end() ) return 0;
    auto aiter = en->second->find(s);
    return aiter != en->second->end();
   });
}

// main horror - try to detect what dis op is
int ParseSASS::classify_op(int op_idx, const std::string &s)
{
  int idx = 0, minus = 0;
  char c = s.at(idx);
  if ( c == '-' ) { minus = 1; idx++; }
  else if ( c == '+' ) idx++;
  std::string_view tmp{ s.c_str() + idx, s.size() - idx};
  if ( tmp == "INF"sv ) return reduce(R_value);
  auto cl = [](const render_base *rb) { return rb->type == R_C || rb->type == R_CX; };
  if ( tmp.starts_with("desc[") ) return reduce(R_desc);
  if ( tmp.starts_with("c[") ) {
    return apply_kind(m_forms, cl);
  }
  if ( tmp.starts_with("0x") ) return reduce(R_value);
  if ( tmp.starts_with("(*\"BRANCH_TARGETS") ) return reduce(R_value);
  switch(c) {
    case '`': if ( s.at(1) != '(' ) {
       fprintf(stderr, "unknown op %d: %s\n", op_idx, s.c_str());
       return 0;
     }
     return reduce(R_value);
     break;
    case '!': return reduce_pred({ s.c_str() + idx, s.size() - idx});
    case '|': if ( !tmp.ends_with("|") ) {
       fprintf(stderr, "bad operand %d: %s\n", op_idx, s.c_str());
       return 0;
     } else {
       // check what is dis
       std::string_view abs{ s.c_str() + idx, tmp.size() - 1};
       if ( abs.starts_with("c[") ) return apply_kind(m_forms, cl);
       else return apply_enum(abs);
     }
    case '~':
     return reduce(R_enum);
    case '[': return reduce(R_mem);
  }
  // check for digit
  int dig = 1, was_dot = 0;
  for ( auto ti = tmp.cbegin(); ti != tmp.cend(); ++ti ) {
    c = *ti;
    if ( c >= '0' && c <= '9' ) continue;
    if ( c == '.' ) { if ( !was_dot ) { ++was_dot; continue; } }
    dig = 0;
    break;
  }
  if ( dig ) return reduce(R_value);
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
      std::copy_if( s.begin() + i, s.end(), std::back_inserter(ename), [](char c) { return !isspace(c); });
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
dump_out(tmp); printf(" -> "); dump_out(*di); fputc('\n', stdout);
#endif
      int i2 = 1 + last;
      for ( ; i2 < (int)s.size(); ++i2, ++len ) {
        auto c = s.at(i2);
        if ( isspace(c) || c == '.' ) break;
      }
      // check in dotted
      dotted = { s.data() + idx, (size_t)len };
// fputc('>', stdout); dump_out(dotted); fputc('\n', stdout);
      di = m_dotted->find(dotted);
      if ( di != m_dotted->end() ) {
        dotted_last = i2;
// dump_out(dotted); printf(" %d-> ", last); dump_out(*di); fputc('\n', stdout);
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
// #ifdef DEBUG
 printf("apply_enum "); dump_out(s); printf(" last %d dlast %d\n", last, dotted_last);
// #endif

  std::string_view ename(s.begin(), last);
  if ( dotted_last ) {
    if ( check_op(m_forms, [&](const form_list *fl, const nv_instr *instr) -> bool {
    if ( fl->rb->type != R_predicate && fl->rb->type != R_enum ) return 0;
    const render_named *rn = (const render_named *)fl->rb;
    auto ei = find(instr->eas, rn->name);
    if ( !ei ) return 0;
    // check if it has enum in s
    auto en = m_renums->find(ei->name);
    if ( en == m_renums->end() ) return 0;
    auto aiter = en->second->find(dotted);
    return aiter != en->second->end();
   }) ) {
    ename = dotted;
printf("found dotted "); dump_out(ename); fputc('\n', stdout);
   }
  }
  return apply_op(m_forms, [&](const form_list *fl, const nv_instr *instr) -> bool {
    if ( fl->rb->type != R_predicate && fl->rb->type != R_enum ) return 0;
    const render_named *rn = (const render_named *)fl->rb;
    auto ei = find(instr->eas, rn->name);
    if ( !ei ) return 0;
    // check if it has enum in s
    auto en = m_renums->find(ei->name);
    if ( en == m_renums->end() ) return 0;
    auto aiter = en->second->find(ename);
    return aiter != en->second->end();
  });
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
      auto en = m_renums->find(a.second->ename);
      if ( en == m_renums->end() ) continue;
      auto aiter = en->second->find(dotted);
      if ( aiter != en->second->end() ) { found++; return; }
    } });
    if ( found ) {
      // yes, we can proceed with dotted enum
      last = dotted_last;
      ename = dotted;
#ifdef DEBUG
 printf("%d last %d>", found, last); dump_out(ename); fputc('\n', stdout);
#endif
      found = 0;
    }
  }
  // iterate on all remained forms and try to find this attr at thers current operand
  std::erase_if(f, [&](one_form &of) {
    if ( (*of.current)->empty() ) return 1;
    for ( auto &a: (*of.current)->lr ) {
      auto en = m_renums->find(a.second->ename);
      if ( en == m_renums->end() ) continue;
      auto aiter = en->second->find(ename);
      if ( aiter != en->second->end() ) { found++; return 0; }
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
          if ( vi && vi->kind == NV_BITSET ) goto out;
          of.ops.push_back( new form_list(*ri) );
        }
        break;
        case R_enum: {
          rn = (const render_named *)*ri;
          const nv_eattr *ea = nullptr;
          auto ei = find(ins->eas, rn->name);
          if ( ei ) { ea = ei->ea; }
          else { ea = try_by_ename(ins, rn->name); }
          if ( ea && ea->ignore ) { // push it into last
            of.ops.back()->lr.push_back( std::make_pair(*ri, ea) );
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

int ParseSASS::add(const std::string &s)
{
  reset_pred();
  int idx = 0;
  // check predicate
  if ( s.at(0) == '@' ) idx = parse_pred(s);
  // check { for dual-issued instructions
  if ( m_width == 88 && s.at(0) == '{' ) {
    m_kv[c_usched_name] = 0x10; // see https://redplait.blogspot.com/2025/04/nvidia-sass-disassembler-part-7-dual.html
    for ( idx = 1; idx < (int)s.size(); idx++ ) if ( !isspace(s.at(idx)) ) break;
  }
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
  while( idx < (int)head.size() )
  {
    int old_idx = idx;
    auto c = head.at(idx);
    if ( c == '.' ) {
    // some attr
    idx = process_attr(idx, head, m_forms);
    if ( m_forms.empty() ) {
      // surprise - there is mnemonics like UIADD.64
      if ( !old_idx ) {
        std::string second_mnem(head, idx);
        mnem += second_mnem;
        mv = std::lower_bound( m_sorted->begin(), m_sorted->end(), mnem, [](const auto &pair, const std::string &w) {
         return pair.first < w;
        });
        if ( mv != m_sorted->end() ) {
          if ( fill_forms(m_forms, mv->second) ) continue;
        }
      }
      printf("[!] unknown form %s after process_attr\n", head.c_str());
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
  if ( !opt_o ) {
    std::cregex_token_iterator begin(head.c_str() + idx, head.c_str() + head.size(), s_commas, -1), end;
    int op_idx = 0;
    for ( auto op = begin; op != end; ++op, ++op_idx ) {
      auto s = *op;
      if ( !s.length() ) continue;
      if ( op_idx ) { // first next was issued in head processing after first space
        if ( !next(m_forms) ) return 0;
      }
      classify_op(op_idx, *op);
      if ( m_forms.empty() ) {
        printf("[!] empty after %d op: %s\n", op_idx, head.c_str());
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

int ParseSASS::init(const std::string &s)
{
#ifdef DEBUG
 printf("init %s\n", s.c_str());
#endif
  // try to find in tale s_sms
  std::string sm = "sm";
  sm += s;
  std::string found;
  for ( auto &si: s_sms ) {
    if ( si.second.first == sm ) { found = si.second.second ? si.second.second : sm; break; }
  }
  if ( found.empty() ) {
    fprintf(stderr, "unknown %s\n", sm.c_str());
    return 0;
  }
  sm = "./";
  sm += found;
  sm += ".so";
  if ( opt_v ) printf("try load %s\n", sm.c_str());
  if ( !load(sm) ) return 0;
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

int ParseSASS::print_fsummary(FILE *fp) const
{
  auto fsize = m_forms.size();
  if ( !fsize ) {
    fprintf(fp, "[!] no forms\n");
    return 0;
  }
  fprintf(fp, "%ld forms:", fsize);
  for ( auto &f: m_forms ) fprintf(fp, " %d", f.instr->line);
  fputc('\n', fp);
  return 1;
}

//
// main
//
void usage(const char *prog)
{
  printf("usage: %s [options] input.asm\n", prog);
  printf("Options:\n");
  printf(" -m - dump missed fields\n");
  printf(" -o - skip operands parsing\n");
  printf(" -s - print forms summary\n");
  printf(" -S - print stat\n");
  printf(" -v - verbose mode\n");
  exit(6);
}

int main(int argc, char **argv)
{
  int c, opt_S = 0;
  while(1) {
    c = getopt(argc, argv, "mosSv");
    if ( c == -1 ) break;
    switch(c) {
      case 'm': opt_m = 1; break;
      case 'o': opt_o = 1; break;
      case 's': opt_s = 1; break;
      case 'S': opt_S = 1; break;
      case 'v': opt_v = 1; break;
      default: usage(argv[0]);
    }
  }
  if ( argc == optind ) {
    printf("where is input file?\n");
    usage(argv[0]);
    return 6;
  }
  std::ifstream fs(argv[optind]);
  if ( !fs.is_open() ) {
    printf("cannot open %s\n", argv[optind]);
    return 1;
  }
  ParseSASS pa;
  int ln = 1, state = 0;
  std::string s;
  // there are two type for target sm
  // in new nvdisasm: .target sm_120
  // in old something like: .headerflags @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM50 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM50)"
  // make 1 capture group in each for version digits
  // yeh, in perl it looks much better
  std::regex tgt("\\.target\\s+sm_(\\d+)");
  std::regex hdr("\\.headerflags\\s+.*EF_CUDA_SM(\\d+)");
  std::regex cmt("^\\s*\\/\\/");
  std::regex section("^\\s+\\.section\\s+\\.(\\w+)");
  std::regex code("^\\s*\\/\\*.*\\*\\/\\s+(.*)\\s*;");
  unsigned long total = 0,
   succ = 0;
  for( ; std::getline(fs, s); ++ln ) {
    std::smatch matches;
    if ( !state ) {
      if ( std::regex_search(s, matches, tgt) ) {
        if ( !pa.init(matches[1].str()) ) return 2; else state = 1;
      } else if ( std::regex_search(s, matches, hdr) ) {
        if ( !pa.init(matches[1].str()) ) return 2; else state = 1;
      }
      continue;
    }
    // check .text sections
    if ( std::regex_search(s, cmt) ) continue;
    if ( std::regex_search(s, matches, section) ) {
      auto pfx = matches[1].str();
      if ( pfx.starts_with("text") ) state = 2;
      continue;
    }
    if ( 2 == state ) {
      if ( std::regex_search(s, matches, code) ) {
        auto what = matches[1].str();
        if ( opt_v ) printf("%d %s\n", ln, what.c_str());
        total++;
        if ( !pa.add(what) ) printf("[!] %d %s\n", ln, what.c_str());
        else {
          succ++;
          if ( opt_s ) pa.print_fsummary(stdout);
        }
      }
    }
  }
  if ( opt_S ) {
    printf("total %ld succ %ld rate %f\n", total, succ, (double)succ / (double)total);
  }
}