#include <fstream>
#include <unistd.h>
#include <iostream>
#include "sass_parser.h"

int opt_d = 0,
    skip_final_cut = 0,
    opt_k = 0,
    opt_m = 0,
    opt_s = 0,
    skip_op_parsing = 0,
    opt_v = 0;

// for sv literals
using namespace std::string_view_literals;

class MyParseSASS: public ParseSASS
{
  public:
   MyParseSASS(): ParseSASS()
   { }
   virtual int init(const std::string &s) override;
   int print_fsummary(FILE *) const;
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
    fprintf(stderr, "unknown %s\n", sm.c_str());
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
  if ( 1 == fsize )
    fprintf(fp, "%ld form:\n", fsize);
  else
    fprintf(fp, "%ld forms:\n", fsize);
  for ( auto &f: m_forms ) {
    if ( opt_k ) {
     for ( auto &ki: f.l_kv ) {
       fprintf(fp, " %s", ki.first.c_str());
       if ( f.instr->vas ) {
         auto vas = find(f.instr->vas, ki.first);
         if ( vas ) {
           fprintf(fp, " %s", s_fmts[vas->kind]);
           std::string res;
           dump_value(*vas, ki.second, vas->kind, res);
           fprintf(fp, ": %s\n", res.c_str());
           continue;
         }
       }
       fprintf(fp, ": %ld\n", ki.second);
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
  printf(" -v - verbose mode\n");
  exit(6);
}

class istr
{
  public:
   std::istream *is = nullptr;
   virtual ~istr() {}
};

class Istr: public istr
{
  public:
   Istr() {
     is = &std::cin;
   }
};

class Fstr: public istr
{
  public:
   virtual ~Fstr() {}
   Fstr(const char *fname): m_f(fname) {
     is = &m_f;
   }
   bool is_open() const {
     return m_f.is_open();
   }
  protected:
   std::ifstream m_f;
};

int main(int argc, char **argv)
{
  int c, opt_S = 0;
  while(1) {
    c = getopt(argc, argv, "dekmosSv");
    if ( c == -1 ) break;
    switch(c) {
      case 'd': opt_d = 1; break;
      case 'e': skip_final_cut = 1; break;
      case 'k': opt_k = 1; break;
      case 'm': opt_m = 1; break;
      case 'o': skip_op_parsing = 1; break;
      case 's': opt_s = 1; break;
      case 'S': opt_S = 1; break;
      case 'v': opt_v = 1; break;
      case '?':
      default: usage(argv[0]);
    }
  }
  istr *is;
  if ( argc == optind ) {
    is = new Istr();
  } else {
    auto fs = new Fstr(argv[optind]);
    if ( !fs->is_open() ) {
      printf("cannot open %s\n", argv[optind]);
      delete fs;
      return 1;
    }
    is = fs;
  }
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
  std::regex section("^\\s+\\.section\\s+\\.(\\w+)");
  std::regex code("^\\s*(?:\\[.+\\]\\s+)?\\/\\*.*\\*\\/\\s+(.*)\\s*;");
  unsigned long total = 0,
   succ = 0,
   forms = 0;
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
          forms += pa.fsize();
          if ( opt_s ) pa.print_fsummary(stdout);
        }
      }
    }
  }
  delete is;
  if ( opt_S && total ) {
    printf("total %ld succ %ld forms %ld rate %f avg %f\n", total, succ, forms,
     (double)succ / (double)total, (double)forms / (double)total);
  }
}