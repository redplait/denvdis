#include <fstream>
#include "nv_rend.h"
#include <regex>
#include <unistd.h>

int opt_m = 0,
    opt_v = 0;

class ParseSASS: public NV_renderer
{
  public:
   ParseSASS(): NV_renderer()
   { }
   int init(const std::string &s);
   int add(const std::string &s);
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
   int parse_req(const char *s);
   int parse_digit(const char *s, int &v);
   int parse_pred(const std::string &s);
   std::string process_tail(int idx, const std::string &s, NV_Forms &);
   int process_attr(int idx, const std::string &s, NV_Forms &);
   // currently kv
   NV_extracted m_kv;
   static std::regex s_digits;
   static constexpr auto c_usched_name = "usched_info";
   const NV_sorted *m_sorted = nullptr;
   const NV_Renums *m_renums = nullptr;
   const NV_Renum *usched = nullptr;
};

std::regex ParseSASS::s_digits("\\d+");

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

// idx - index of '.' at start of attr
int ParseSASS::process_attr(int idx, const std::string &s, NV_Forms &f)
{
  int last;
  for ( last = ++idx; last < (int)s.size(); ++last ) {
    auto c = s.at(last);
    if ( isspace(c) || c == '.' ) break;
  }
  std::string_view ename(s.c_str() + idx, last - idx);
#ifdef DEBUG
 printf("attr %s len %d\n", s.c_str() + idx, last - idx);
#endif
  int found = 0;
  // iterate on all remained forms and try to find this attr at thers current operand
  std::erase_if(f, [&](one_form &of) {
    if ( (*of.current)->empty() ) return 1;
    for ( auto &a: (*of.current)->lr ) {
      auto en = m_renums->find(a.second->ename);
#ifdef DEBUG
 printf("check in %s\n", a.second->ename);
#endif
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
  NV_Forms forms;
  fill_forms(forms, mv->second);
  if ( forms.empty() ) return 0;
  if ( idx >= (int)s.size() ) return 1;
  // process first tail with rd/wr etc
  std::string head = process_tail(idx, s, forms);
  if ( head.empty() ) {
    if ( !forms.empty() ) return 1;
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
    idx = process_attr(idx, head, forms);
    if ( forms.empty() ) {
      // surprise - there is mnemonics like UIADD.64
      if ( !old_idx ) {
        std::string second_mnem(head, idx);
        mnem += second_mnem;
        mv = std::lower_bound( m_sorted->begin(), m_sorted->end(), mnem, [](const auto &pair, const std::string &w) {
         return pair.first < w;
        });
        if ( mv != m_sorted->end() ) {
          if ( fill_forms(forms, mv->second) ) continue;
        }
      }
      printf("[!] unknown form %s after process_attr\n", head.c_str());
      return 0;
     }
    } else if ( c == ' ' ) {
      if ( !next(forms) ) return 0;
      idx++; break;
    } else {
      printf("[!] cannot parse %s\n", head.c_str() + idx);
      return 0;
    }
  }
  return 1;
}

int ParseSASS::next(NV_Forms &f) const
{
  std::for_each( f.begin(), f.end(), [](one_form &of) { of.current = of.ops.begin(); });
  std::erase_if(f, [](one_form &of) { return of.current == of.ops.end(); });
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
  return 1;
}

void usage(const char *prog)
{
  printf("usage: %s [options] input.asm\n", prog);
  printf("Options:\n");
  printf(" -m - dump missed fields\n");
  printf(" -v - verbose mode\n");
  exit(6);
}

int main(int argc, char **argv)
{
  int c;
  while(1) {
    c = getopt(argc, argv, "mv");
    if ( c == -1 ) break;
    switch(c) {
      case 'm': opt_m = 1; break;
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
        if ( !pa.add(what) ) printf("[!] %d %s\n", ln, what.c_str());
      }
    }
  }
}