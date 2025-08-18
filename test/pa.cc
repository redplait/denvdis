#include <unistd.h>
#include "sass_parser.h"

int opt_d = 0,
    skip_final_cut = 0,
    opt_k = 0,
    opt_m = 0,
    opt_s = 0,
    skip_op_parsing = 0,
    opt_T = 0,
    opt_v = 0;

class MyParseSASS: public ParseSASS
{
  public:
   MyParseSASS(): ParseSASS()
   {
     if ( opt_T ) m_rtdb = new reg_pad;
   }
   virtual ~MyParseSASS() {
     if ( m_rtdb ) delete m_rtdb;
   }
   virtual int init(const std::string &s) override;
   int print_fsummary(FILE *) const;
   void dump_rt(const std::string &pfx) {
     if ( !m_rtdb || m_rtdb->empty() ) return;
     finalize_rt(m_rtdb);
     fprintf(m_out, "; %s\n", pfx.c_str());
     ParseSASS::dump_rt(m_rtdb);
     m_rtdb->clear();
   }
   int add_with_rt(const std::string &s, unsigned long off) {
     int res = add(s);
     if ( res ) {
       const one_form *of = &m_forms.at(0);
       NV_pair p;
       p.first = of->instr;
       if ( extract(p.second) )
         track_regs(m_rtdb, of->rend, p, off);
     }
     return res;
   }
   int verify(unsigned long off) {
     NV_extracted ex;
     if ( m_forms.empty() ) return 0;
     // if there are several - no difference which to use so let it be first
     const one_form *of = &m_forms.at(0);
     if ( !_extract_full(ex, of) ) {
       fprintf(m_out, "[!] %lX - extract_full failed\n", off);
       return 0;
     }
     if ( !validate_tabs(of->instr, ex) ) {
       fprintf(m_out, "[!] %lX - validate_tabs failed\n", off);
       return 0;
     }
     return 1;
   }
  protected:
   // regs track db
   reg_pad *m_rtdb = nullptr;
};

int MyParseSASS::init(const std::string &s)
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
    Err("unknown %s\n", sm.c_str());
    return 0;
  }
  // check SM_DIR env
  char *sm_dir = getenv("SM_DIR");
  if ( sm_dir ) {
    sm = sm_dir;
    if ( !sm.ends_with("/") ) sm += '/';
  } else {
    sm = "./";
  }
  sm += found;
  sm += ".so";
  if ( opt_v ) printf("try load %s\n", sm.c_str());
  if ( !load(sm) ) return 0;
  return init_guts();
}

int MyParseSASS::print_fsummary(FILE *fp) const
{
  auto fsize = m_forms.size();
  if ( !fsize ) {
    fprintf(fp, "[!] no forms\n");
    return 0;
  }
  auto dump_kv_item = [&](const nv_instr *instr, auto &&ki) {
    if ( instr->vas ) {
      auto vas = find(instr->vas, ki.first);
      if ( vas ) {
        fprintf(fp, " %s", s_fmts[vas->kind]);
        std::string res;
        dump_value(*vas, ki.second, vas->kind, res);
        fprintf(fp, ": %s\n", res.c_str());
        return;
      }
    }
    fprintf(fp, ": %ld\n", ki.second);
  };
  if ( opt_k && !m_kv.empty() ) {
    auto ins = m_forms[0].instr;
    for ( auto &ki: m_kv ) {
      fputc(' ', fp); dump_out(ki.first, fp);
      dump_kv_item(ins, ki);
    }
  }
  if ( 1 == fsize )
    fprintf(fp, "%ld form:\n", fsize);
  else
    fprintf(fp, "%ld forms:\n", fsize);
  for ( auto &f: m_forms ) {
    if ( opt_k ) {
     for ( auto &ki: f.l_kv ) {
       fprintf(fp, " %s", ki.first.c_str());
       dump_kv_item(f.instr, ki);
     }
    }
    fprintf(fp, " %d", f.instr->line);
    std::string res;
    rend_rendererE(f.instr, f.rend, res);
    fprintf(fp, " %s", res.c_str());
    if ( f.has_label() ) {
      fprintf(fp, " ; LABEL %s %s", s_labels[f.ltype], f.lname.c_str());
    }
    fputc('\n', fp);
  }
  return 1;
}

//
// main
//
void usage(const char *prog)
{
  printf("usage: %s [options] input.asm\n", prog);
  printf("Options:\n");
  printf(" -d - debug mode\n");
  printf(" -e - skip final cut\n");
  printf(" -k - dump kv\n");
  printf(" -m - dump missed fields\n");
  printf(" -o - skip operands parsing\n");
  printf(" -s - print forms summary\n");
  printf(" -S - print stat\n");
  printf(" -T - track registers\n");
  printf(" -v - verbose mode\n");
  printf(" -V - verify instructions, very slow\n");
  exit(6);
}

int main(int argc, char **argv)
{
  int c, opt_S = 0, opt_V = 0;
  while(1) {
    c = getopt(argc, argv, "dekmosSTvV");
    if ( c == -1 ) break;
    switch(c) {
      case 'd': opt_d = 1; break;
      case 'e': skip_final_cut = 1; break;
      case 'k': opt_k = 1; break;
      case 'm': opt_m = 1; break;
      case 'o': skip_op_parsing = 1; break;
      case 's': opt_s = 1; break;
      case 'S': opt_S = 1; break;
      case 'T': opt_T = 1; break;
      case 'v': opt_v = 1; break;
      case 'V': opt_V = 1; break;
      case '?':
      default: usage(argv[0]);
    }
  }
  ParseSASS::Istr *is = ParseSASS::try_open(argc == optind ? nullptr : argv[optind]);
  if ( !is )
     return 1;
  MyParseSASS pa;
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
  std::regex section("^\\s+\\.section\\s+\\.([\\w_@\\.]+)");
  std::regex code("^\\s*(?:\\[.+\\]\\s+)?\\/\\*(.*)\\*\\/\\s+(.*)\\s*;");
  unsigned long total = 0,
   succ = 0,
   forms = 0;
  std::string text_sname;
  for( ; std::getline(*is->is, s); ++ln ) {
    std::smatch matches;
// printf("line %d state %d\n", ln, state);
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
      if ( opt_T ) {
        pa.dump_rt(text_sname);
      }
      auto pfx = matches[1].str();
      if ( pfx.starts_with("text") ) { text_sname = pfx; state = 2; }
      continue;
    }
    if ( 2 == state ) {
      if ( std::regex_search(s, matches, code) ) {
        auto what = matches[2].str();
        if ( opt_v ) printf("%d %s\n", ln, what.c_str());
        total++;
        int add_res = 0;
        unsigned long off = 0;
        if ( opt_T || opt_V ) {
          // parse offset in matches[1]
          char *end;
          off = strtoul(matches[1].str().c_str(), &end, 16);
        }
        if ( opt_T )
          add_res = pa.add_with_rt(what, off);
        else
          add_res = pa.add(what);
        if ( !add_res ) printf("[!] %d %s\n", ln, what.c_str());
        else {
          succ++;
          forms += pa.fsize();
          if ( opt_V ) pa.verify(off);
          if ( opt_s ) pa.print_fsummary(stdout);
        }
      }
    }
  }
  if ( opt_T )
    pa.dump_rt(text_sname);
  delete is;
  if ( opt_S && total ) {
    printf("total %ld succ %ld forms %ld rate %f avg %f\n", total, succ, forms,
     (double)succ / (double)total, (double)forms / (double)total);
  }
}