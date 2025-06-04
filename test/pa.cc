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
   int parse_pred(const std::string &s);
   // currently kv
   NV_extracted m_kv;
   const NV_sorted *m_sorted = nullptr;
   const NV_Renums *m_renums = nullptr;
   const NV_Renum *usched = nullptr;
};

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

int ParseSASS::add(const std::string &s)
{
  reset_pred();
  int idx = 0;
  // check predicate
  if ( s.at(0) == '@' ) idx = parse_pred(s);
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
  for ( auto ins: mv->second ) {
    auto r = m_dis->get_rend(ins->n);
    if ( !r ) continue;
    auto ri = r->begin();
    if ( has_pred() && (*ri)->type != R_predicate ) continue;
    if ( (*ri)->type == R_predicate ) ++ri;
    if ( (*ri)->type != R_opcode ) continue;
    one_form of(ins, r);
    // dissect render - ri holds R_opcode and pushed into of
    of.ops.push_back( new form_list(*ri) );
    for ( ; ri != r->end(); ++ri ) {
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
  }
  if ( forms.empty() ) return 0;
  std::for_each( forms.begin(), forms.end(), [](one_form &of) { of.current = of.ops.begin(); });
  if ( idx >= (int)s.size() ) return 1;
  auto c = s.at(idx);
  if ( s.at(idx) == '.' ) {
    // some attr
  } else if ( c == ' ' ) {
    if ( !next(forms) ) return 0;
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