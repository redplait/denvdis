#include <readline/history.h>
#include <readline/readline.h>
#include "nv_rend.h"
// for stat & getopt
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

int opt_m = 0,
    opt_v = 0;

static const NV_sorted *g_sorted = nullptr;
static int g_sorted_idx = -1;
static const std::vector<const nv_instr *> *g_found;
// mess of globals for readline interface
static std::string g_prompt, s_opcode;

// completiton logic stolen from https://prateek.page/post/gnu-readline-for-tab-autocomplete-and-bash-like-history/
char *null_generator(const char *text, int state) {
  return nullptr;
};

char *instr_generator(const char *text, int state) {
  if ( -1 == g_sorted_idx || g_sorted_idx >= (int)g_sorted->size() ) return nullptr;
  std::string textstr(text);
  std::transform(textstr.begin(), textstr.end(), textstr.begin(), ::toupper);
  auto row = &g_sorted->at(g_sorted_idx);
  if ( !row->first.starts_with(textstr) ) {
    g_sorted_idx = -1;
    return nullptr;
  }
  auto sv_size = row->first.size();
  char *res = (char *)malloc(sv_size + 1);
  memcpy(res, row->first.data(), sv_size);
  res[sv_size] = 0;
  g_sorted_idx++;
  return res;
}

static char **instr_completion(const char *text, int start, int end) {
  rl_attempted_completion_over = 1;
  g_sorted_idx = -1;
  std::string what( text + start, text + end);
  std::transform(what.begin(), what.end(), what.begin(), ::toupper);
#ifdef DEBUG
  printf("instr_completion: %s\n", what.c_str());
#endif
  // try to find in g_sorted name with what prefix
  auto lb = std::lower_bound( g_sorted->begin(), g_sorted->end(), what, [](const auto &pair, const std::string &w) {
    return pair.first < w;
   });
  if ( lb == g_sorted->end() )
    return rl_completion_matches(text, null_generator);
  if ( !lb->first.starts_with(what) )
    return rl_completion_matches(text, null_generator);
  g_sorted_idx = lb - g_sorted->begin();
#ifdef DEBUG
  printf("start idx: %d\n", g_sorted_idx);
#endif
 return rl_completion_matches(text, instr_generator);
}

struct Apply {
  virtual Apply *next() = 0;
};
typedef Apply *ptrApply;

struct INA: public NV_renderer {
  ~INA() {
    if ( ibuf ) free(ibuf);
    if ( ifp ) fclose(ifp);
  }
  INA(): NV_renderer()
  { }
  typedef std::function<ptrApply()> FApply;
  struct inapply: public Apply {
    FApply clos;
    inapply(FApply cl): clos(cl) {}
    virtual Apply *next() override { return clos(); }
  };
  inapply mnem_name{ [&]() -> ptrApply {
    g_prompt = "> ";
    rl_attempted_completion_function = instr_completion;
    char *buf;
    while( nullptr != (buf = readline(g_prompt.c_str())) ) {
      if ( !strcmp(buf, "q") ) { free(buf); return nullptr; } // q - quit
      // check if we have such instruction
      std::string what(buf);
      free(buf);
      std::transform(what.begin(), what.end(), what.begin(), ::toupper);
      g_found = find_il( g_sorted, what );
      if ( !g_found ) continue;
      g_prompt = s_opcode = what;
      g_prompt += " ";
      if ( g_found->size() > 1 ) return &mnem_idx;
      auto ins = g_found->at(0);
      printf("ins %d\n", ins->n);
    }
    return nullptr;
  } };
  inapply mnem_idx{ [&]() -> ptrApply {
    // dump renderers
    int rsize = (int)g_found->size();
    printf("%d forms:\n", rsize);
    for ( int i = 0; i < rsize; i++ ) {
      auto ins = g_found->at(i);
      if ( !ins ) continue;
      auto rend = m_dis->get_rend(ins->n);
      if ( !rend ) continue;
      std::string form;
      if ( rend_renderer( rend, s_opcode, form ) ) {
        printf("%d) %s\n", 1+i, form.c_str());
      }
    }
    char *buf;
    rl_attempted_completion_function = nullptr;
    while( nullptr != (buf = readline(g_prompt.c_str())) ) {
      if ( !strcmp(buf, "q") ) { free(buf); return nullptr; } // q - quit
      if ( !strcmp(buf, "b") ) { free(buf); g_found = nullptr; return &mnem_name; } // b - back to instruction selection
    }
    return nullptr;
  } };
  int init(int dump);
  int open_binary(const char *);
  int process_binary(const char *);
  // 64bit - 8 + 7 * 8 = 64 bytes
  // 88bit - 8 + 3 * 8 = 32 bytes
  // 128bit - just 16 bytes
  unsigned char buf[64];
  size_t block_size = 0;
 protected:
  void r_ve(const ve_base &, std::string &res);
  void r_velist(const std::list<ve_base> &l, std::string &res);
  int rend_renderer(const NV_rlist *, const std::string &opcode, std::string &res);
  void process_buf();
  void dump_ins(const NV_pair &p, size_t off);
  FILE *ifp = nullptr;
  unsigned char *ibuf = nullptr;
  NV_extracted m_kv;
} g_ina;

int INA::rend_renderer(const NV_rlist *rlist, const std::string &opcode, std::string &res)
{
  for ( auto r: *rlist ) {
    switch(r->type) {
      case R_value:
      case R_predicate: {
        const render_named *rn = (const render_named *)r;
        if ( r->pfx ) res += r->pfx;
        res += rn->name;
       }
       break;
      case R_enum:{
        const render_named *rn = (const render_named *)r;
        res += "E:";
        res += rn->name;
       }
       break;
      case R_opcode:
        res += opcode;
       break;
      case R_C:
      case R_CX: {
         const render_C *rn = (const render_C *)r;
         res += "c:[";
         r_ve(rn->left, res);
         res += "][";
         r_velist(rn->right, res);
         res += ']';
       } break;
       case R_TTU: {
         const render_TTU *rt = (const render_TTU *)r;
         if ( rt->pfx ) res += rt->pfx;
         else res += ' ';
         res += "ttu:[";
         r_ve(rt->left, res);
         res += ']';
       }
       break;
     case R_M1: {
         const render_M1 *rt = (const render_M1 *)r;
         if ( rt->pfx ) res += rt->pfx;
         res += rt->name;
         res += ":[";
         r_ve(rt->left, res);
         res += ']';
       } break;

      case R_desc: {
         const render_desc *rt = (const render_desc *)r;
         if ( rt->pfx ) res += rt->pfx;
         res += "desc:[";
         r_ve(rt->left, res);
         res += "],[";
         r_velist(rt->right, res);
         res += ']';
       } break;

      case R_mem: {
         const render_mem *rt = (const render_mem *)r;
         if ( rt->pfx ) res += rt->pfx;
         res += "[";
         r_velist(rt->right, res);
         res += ']';
       } break;

//      default: fprintf(stderr, "unknown rend type %d at index %d for inst %s\n", r->type, idx, opcode.c_str());
    }
    res += ' ';
  }
  res.pop_back(); // remove last space
  return !res.empty();
}

void INA::r_ve(const ve_base &ve, std::string &res)
{
  if ( ve.type == R_enum ) res += "E:";
  res += ve.arg;
}

void INA::r_velist(const std::list<ve_base> &l, std::string &res)
{
  auto size = l.size();
  if ( 1 == size ) {
    r_ve(*l.begin(), res);
    return;
  }
  int idx = 0;
  for ( auto ve: l ) {
    if ( ve.type == R_value )
    {
      if ( ve.pfx ) res += ve.pfx;
      else if ( idx ) res += '+';
      res += ve.arg;
      idx++;
      continue;
    }
    // enum
    res += "E:";
    res += ve.arg;
    res += " ";
  }
  if ( res.back() == ' ' ) res.pop_back();
}

int INA::init(int dump)
{
  memset(buf, 0, 64);
  g_sorted = m_dis->get_instrs();
  if ( !g_sorted ) return -1;
  switch(m_width) {
    case 64: block_size = 64; break;
    case 88: block_size = 32; break;
    case 128: block_size = 16; break;
    default:
     fprintf(stderr, "Unknown width %d\n", m_width);
     return -2;
  }
  m_dis->init(buf, block_size);
  if ( dump )
    printf("width %d, %ld instructions\n", m_width, g_sorted->size());
  return 0;
}

int INA::open_binary(const char *fname)
{
  if ( ifp ) { fclose(ifp); ifp = nullptr; }
  ifp = fopen(fname, "wb");
  if ( !ifp ) {
    fprintf(stderr, "cannot open %s, error %d (%s)\n", fname, errno, strerror(errno));
    return 0;
  }
  return 1;
}

int INA::process_binary(const char *fname)
{
  // get size of file
  struct stat fs;
  if ( stat(fname, &fs) ) {
    fprintf(stderr, "cannot stat %s, error %d (%s)\n", fname, errno, strerror(errno));
    return 1;
  }
  // check if size is multiple of block_size
  if ( !fs.st_size ) {
    fprintf(stderr, "file %s is empty\n", fname);
    return 2;
  }
  auto rem = fs.st_size % block_size;
  if ( rem ) {
    fprintf(stderr, "size of %s %ld is not multiple of width %ld\n", fname, fs.st_size, block_size);
    return 3;
  }
  // open file
  ifp = fopen(fname, "rb");
  if ( !ifp ) {
    fprintf(stderr, "cannot open %s, error %d (%s)\n", fname, errno, strerror(errno));
    return 1;
  }
  // alloc enough mem
  ibuf = (unsigned char *)malloc(fs.st_size);
  if ( !ibuf ) {
    fprintf(stderr, "cannot alloc %ld bytes\n", fs.st_size);
    return 4;
  }
  // read
  if ( 1 != fread(ibuf, fs.st_size, 1, ifp) ) {
    fprintf(stderr, "cannot read %s, error %d (%s)\n", fname, errno, strerror(errno));
    return 1;
  }
  fclose(ifp); ifp = nullptr;
  m_dis->init(ibuf, fs.st_size);
  // process buffer
  process_buf();
  if ( opt_v )
    dis_stat();
  return 0;
}

void INA::process_buf()
{
  dual_first = dual_last = false;
  while(1) {
    NV_res res;
    int get_res = m_dis->get(res);
    if ( -1 == get_res ) { fprintf(m_out, "stop at %lX\n", m_dis->offset()); break; }
    dis_total++;
    if ( !get_res ) {
      dis_notfound++;
      fprintf(m_out, "Not found at %lX\n", m_dis->offset());
      break;
    }
    int res_idx = 0;
    if ( res.size() > 1 ) res_idx = calc_index(res, m_dis->rz);
    auto off = m_dis->offset();
    if ( res_idx == -1 ) {
      dis_dups++;
      dual_first = dual_last = false;
      fprintf(m_out, "%lX: DUPS\n", off);
    } else {
      if ( m_width == 88 && !dual_first && !dual_last )
        dual_first = check_dual(res[res_idx].second);
      dump_ins(res[res_idx], off);
    }
    // reset dual
    if ( dual_first ) {
      dual_first = false;
      dual_last = true;
    } else if ( dual_last )
      dual_last = false;
  }
}

void INA::dump_ins(const NV_pair &p, size_t off)
{
  m_missed.clear();
  auto rend = m_dis->get_rend(p.first->n);
  if ( rend ) {
    std::string r;
    int miss = render(rend, r, p.first, p.second, nullptr);
    fprintf(m_out, "%lX: ", off);
    if ( dual_first ) fputs(" {", m_out);
    else if ( dual_last ) fputs("  ", m_out);
    fprintf(m_out, " %s", r.c_str());
    if ( dual_last ) fputs(" }", m_out);
    if ( miss ) {
      fprintf(m_out, " // %d missed", miss);
      if ( opt_m ) {
        fputc(':', m_out);
        for ( auto &ms: m_missed ) fprintf(m_out, " %s", ms.c_str());
      }
    }
    fputc('\n', m_out);
  }
}

void usage(const char *prog)
{
  printf("usage: %s [options] smXX.so\n", prog);
  printf("Options:\n");
  printf(" -i input binary\n");
  printf(" -o output binary\n");
  printf(" -m - dump missed fields\n");
  printf(" -v - verbose mode\n");
  exit(6);
}

int main(int argc, char **argv)
{
  int c;
  const char *o_fname = nullptr, *i_fname = nullptr;
  while(1) {
    c = getopt(argc, argv, "mvi:o:");
    if ( c == -1 ) break;
    switch(c) {
      case 'm': opt_m = 1; break;
      case 'o': o_fname = optarg; break;
      case 'i': i_fname = optarg; break;
      case 'v': opt_v = 1; break;
      default: usage(argv[0]);
    }
  }
  if ( argc == optind ) {
    usage(argv[0]);
    return 6;
  }
  // try load
  if ( !g_ina.load(argv[optind]) ) {
    fprintf(stderr, "unable to load %s\n", argv[optind]);
    return 1;
  }
  // dump summary
  if ( g_ina.init(opt_v) ) {
    fprintf(stderr, "unable to init %s\n", argv[optind]);
    return 1;
  }
  if ( i_fname ) {
    return g_ina.process_binary(i_fname);
  }
  if ( o_fname && !g_ina.open_binary(o_fname) )
  {
    fprintf(stderr, "unable to open %s\n", o_fname);
    return 1;
  }
  Apply *next = &g_ina.mnem_name;
  while( next ) next = next->next();
  return 0;
}