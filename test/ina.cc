#include <readline/history.h>
#include <readline/readline.h>
#include "nv_rend.h"
#include <numeric>
#include <fp16.h>
// for stat & getopt
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

int opt_m = 0,
    opt_v = 0;

enum kv_type {
  KV_FIELD,
  KV_TAB,
  KV_CBANK,
  KV_CTRL,
};

struct kv_field {
  kv_type type;
  union {
    const NV_field *f;
    const NV_tab_fields *t;
    const NV_cbank *cb;
  };
  int tab_idx = 0, field_idx = 0;
  const nv_eattr *ea = nullptr;
  const nv_vattr *va = nullptr;
  // constructors
  kv_field(const NV_field *_f) {
    type = KV_FIELD;
    f = _f;
  }
  kv_field(const NV_tab_fields *_t, int idx, int fidx) {
    type = KV_TAB;
    t = _t; tab_idx = idx; field_idx = fidx;
  }
  kv_field(const NV_cbank *_cb, int idx) {
    type = KV_CBANK;
    cb = _cb; tab_idx = idx;
  }
  kv_field(kv_type k) { // mostly for KV_CTRL
    type = k;
  }
  inline int has_format(int &v) const {
    if ( !va ) return 0;
    v = va->kind;
    return 1;
  }
  int mask_len(const std::pair<short, short> *m, int size) const
  {
    if ( !m ) return 0;
    return std::accumulate(m, m + size, 0, [](int res, const std::pair<short, short> &item)
     { return res + item.second; } );
  }
  template <typename T>
  int mask_len(const T *what) const
  {
    return std::accumulate(what->mask, what->mask + what->mask_size, 0, [](int res, const std::pair<short, short> &item)
     { return res + item.second; } );
  }
  int mask_len() const
  {
    switch(type) {
      case KV_FIELD: return mask_len(f);
      case KV_TAB: return mask_len(t);
      case KV_CBANK: return mask_len(cb->mask1, cb->mask1_size) + mask_len(cb->mask2, cb->mask2_size) + mask_len(cb->mask3, cb->mask3_size);
      default: return 0;
    }
    return 0;
  }
  int patch(uint64_t v, INV_disasm *dis) const
  {
    switch(type) {
      case KV_FIELD: return dis->put(f->mask, f->mask_size, v);
      case KV_TAB:   return dis->put(t->mask, t->mask_size, v);
      case KV_CTRL:  return dis->put_ctrl((uint8_t)v);
      case KV_CBANK:
        if ( tab_idx == 1 ) {
          if ( cb->scale ) v /= cb->scale;
          return 3 == cb_size() ? dis->put(cb->mask3, cb->mask3_size, v) : dis->put(cb->mask2, cb->mask2_size, v);
        }
        if ( !tab_idx ) {
          auto msize = cb_size(); // calc count of masks
          if ( 2 == msize ) return dis->put(cb->mask1, cb->mask1_size, v);
          if ( 3 == msize ) { // BankLo | (BankHi << 4), masks: BankHi - 1, BankLo - 2
            auto lo = v & 0xf;
            auto hi = (v >> 4) & 0xf;
            if ( !dis->put(cb->mask1, cb->mask1_size, hi) ) return 0;
            return dis->put(cb->mask2, cb->mask2_size, lo);
          }
        }
    }
    return 0;
  }
  // for inserting to std::map
  kv_field& operator=(kv_field&& other) = default;
  kv_field(kv_field&& other) = default;
  kv_field& operator=(kv_field& other) = default;
  kv_field(const kv_field& other) = default;

  int cb_size() const {
    if ( type != KV_CBANK ) return 0;
    if ( cb->mask3 ) return 3;
    return 2;
  }
};

static const NV_sorted *g_sorted = nullptr;
// mess of globals for readline interface
static int g_sorted_idx = -1;
static const std::vector<const nv_instr *> *g_found;
static std::string g_prompt, s_opcode;
static const nv_instr *g_instr = nullptr;
static std::map<std::string_view, kv_field> s_fields;
static std::map<std::string_view, kv_field>::const_iterator s_fields_iter;

// completiton logic stolen from https://prateek.page/post/gnu-readline-for-tab-autocomplete-and-bash-like-history/
char *null_generator(const char *text, int state) {
  return nullptr;
};

// string helpers
static std::string& rstrip(std::string &s)
{
  while(!s.empty()) {
    auto c = s.back();
    if ( !isspace(c) ) break;
    s.pop_back();
  }
  return s;
}

static std::string& to_up(std::string &s)
{
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  return s;
}

static char *rstrip(char *s)
{
  auto size = strlen(s);
  if ( !size ) return s;
  for ( char *p = s + size - 1; p >= s; ++p )
    if ( !isspace(*p) ) break;
    else *p = 0;
  return s;
}

static const char *lfind(const char *s)
{
  while( *s ) {
    if ( !isspace(*s) ) break;
    ++s;
  }
  if ( !*s ) return nullptr;
  return s;
}

static const char *lspace(const char *s)
{
  while( *s ) {
    if ( isspace(*s) ) break;
    ++s;
  }
  return s;
}

template <typename T>
char *copy_str(const T &what) {
  auto sv_size = what->first.size();
  char *res = (char *)malloc(sv_size + 1);
  memcpy(res, what->first.data(), sv_size);
  res[sv_size] = 0;
  return res;
}

// readlne completitions
char *instr_generator(const char *text, int state) {
  if ( -1 == g_sorted_idx || g_sorted_idx >= (int)g_sorted->size() ) return nullptr;
  std::string textstr(text);
  to_up(textstr);
  auto row = &g_sorted->at(g_sorted_idx);
  if ( !row->first.starts_with(textstr) ) {
    g_sorted_idx = -1;
    return nullptr;
  }
  g_sorted_idx++;
  return copy_str(row);
}

static char **instr_completion(const char *text, int start, int end) {
  rl_attempted_completion_over = 1;
  g_sorted_idx = -1;
  std::string what(text + start, text + end);
  to_up(what);
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

static char *fields_generator(const char *text, int state) {
  if ( s_fields_iter == s_fields.end() )
    return nullptr;
  if ( !s_fields_iter->first.starts_with(text) ) {
    s_fields_iter = s_fields.end();
    return nullptr;
  }
  auto res = copy_str(s_fields_iter);
  ++s_fields_iter;
  return res;
}

static char **fill_completion(const char *text, int start, int end) {
  rl_attempted_completion_over = 1;
  // we can have "i field" or just field (without spaces)
  std::string what(text, text + end);
  if ( what.starts_with("i ") ) {
    // find next
    auto next = what.begin() + 2;
    while( isspace(*next) ) {
      ++next;
      if ( next == what.end() ) break;
    }
    if ( next == what.end() ) {
      s_fields_iter = s_fields.cbegin();
      rl_completion_matches("", fields_generator);
    }
    std::string name(next, what.end());
    s_fields_iter = s_fields.lower_bound(name);
    if ( s_fields_iter == s_fields.cend() ) return rl_completion_matches(text, null_generator);
    return rl_completion_matches(name.c_str(), fields_generator);
  }
  // if we don't have spaces - this is field name
  if ( std::any_of(what.cbegin(), what.cend(), isspace) ) return rl_completion_matches(text, null_generator);
  s_fields_iter = s_fields.lower_bound(what);
  if ( s_fields_iter == s_fields.cend() ) return rl_completion_matches(text, null_generator);
  return rl_completion_matches(text, fields_generator);
}

struct Apply {
  virtual Apply *next() = 0;
};
typedef Apply *ptrApply;

struct INA: public NV_renderer {
  ~INA() {
    if ( ibuf ) free(ibuf);
    if ( ifp ) {
     if ( dirty ) fwrite(buf, block_size, 1, ifp);
     fclose(ifp);
    }
  }
  INA(): NV_renderer()
  { }
  typedef std::function<ptrApply()> FApply;
  struct inapply: public Apply {
    FApply clos;
    inapply(FApply cl): clos(cl) {}
    virtual Apply *next() override { return clos(); }
  };
  /* state machine, for each q is exit
    0 - start - mnem_name
    1 - choose form of mnemonic - mnem_idx
    2 - edit fields - ops_fill
   */
  inapply mnem_name{ [&]() -> ptrApply {
    g_prompt = "> ";
    rl_attempted_completion_function = instr_completion;
    char *buf;
    while( nullptr != (buf = readline(g_prompt.c_str())) ) {
      if ( !strcmp(buf, "q") ) { free(buf); return nullptr; } // q - quit
      // check if we have such instruction
      std::string what(buf);
      to_up(rstrip(what));
      free(buf);
      g_found = find_il( g_sorted, what );
      if ( !g_found ) continue;
      g_prompt = s_opcode = what;
      g_prompt += " ";
      add_history(s_opcode.c_str());
      if ( g_found->size() > 1 ) return &mnem_idx;
      auto ins = g_found->at(0);
#ifdef DEBUG
      printf("ins %d\n", ins->n);
#endif
      if ( pre_build(ins) ) return &ops_fill;
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
        printf("%d) ", 1 + i);
        if ( opt_v ) printf("n %d line %d ", ins->n, ins->line);
        printf("%s\n", form.c_str());
      }
    }
    char *buf;
    rl_attempted_completion_function = nullptr;
    while( nullptr != (buf = readline(g_prompt.c_str())) ) {
      if ( !strcmp(buf, "q") ) { free(buf); return nullptr; } // q - quit
      if ( !strcmp(buf, "b") ) { free(buf); g_found = nullptr; return &mnem_name; } // b - back to instruction selection
      char *end = nullptr;
      auto idx = strtol(buf, &end, 10);
      if ( *end ) { printf("bad index %s\n", buf); free(buf); continue; }
      free(buf);
      if ( idx < 1 || idx > rsize ) { printf("invalid index %ld\n", idx); continue; }
      if ( pre_build( g_found->at(idx - 1) ) ) return &ops_fill;
    }
    return nullptr;
  } };
  inapply ops_fill { [&]() -> ptrApply {
    char *buf;
    rl_attempted_completion_function = fill_completion;
    while( nullptr != (buf = readline(g_prompt.c_str())) ) {
      if ( !strcmp(buf, "q") ) { free(buf); return nullptr; } // q - quit
      if ( !strcmp(buf, "b") ) { free(buf); break; } // b - back to instruction selection
      if ( !strcmp(buf, "w") ) { free(buf); if ( flush() ) break;
       fprintf(stderr, "cannot flush\n"); continue;
      }
      if ( !strcmp(buf, "r") ) { free(buf); dump_curr_rend(); continue; }
      if ( !strcmp(buf, "kv") ) { free(buf); dump_kv(); continue; }
      // i field
      if ( buf[0] == 'i' && isspace(buf[1]) ) {
        // find first non-space symbol
        const char *what = lfind(rstrip(buf + 2));
        if ( !what ) { free(buf); continue; }
        if ( dump_i(what) ) add_history(buf);
        free(buf);
        continue;
      }
      // TabXX
      if ( buf[0] == 'T' && buf[1] == 'a' && buf[2] == 'b' ) {
        auto tab = find_tab(buf + 3);
        if ( !tab ) { free(buf); continue; }
        const char *next = lspace(buf + 3);
        if ( !next ) { free(buf); continue; }
        const char *rest = lfind(next);
        if ( !rest ) {
          printf("invalid format\n"); free(buf); continue;
        }
        patch_Tab(tab, rest);
        free(buf);
        continue;
      }
      // field value - let's first try to find field
      std::string_view sv;
      char *text = rstrip(buf);
      if ( !extract_sv(text, sv) ) {
        printf("cannot extract field name\n"); free(buf); continue;
      }
      // find field by name sv
      const char *rest = lfind(text + sv.size());
      if ( !rest ) {
        printf("invalid format\n"); free(buf); continue;
      }
      auto what = s_fields.find(sv);
      if ( what == s_fields.end() ) {
        // ok, check in ins->eas
        auto ea = find(g_instr->eas, sv);
        if ( ea )
          what = s_fields.find(ea->ea->ename);
      }
      if ( what == s_fields.end() ) {
        printf("unknown field ");
        std::for_each(sv.cbegin(), sv.cend(), [&](char c){ putc(c, stdout); });
        fputc('\n', stdout);
        free(buf); continue;
      }
      if ( patch(what, rest) ) add_history(text);
      free(buf); continue;
    }
    g_instr = nullptr;
    g_found = nullptr;
    return &mnem_name;
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
  int extract_sv(const char *, std::string_view &);
  int dump_i(const char *);
  void dump_kv();
  void r_ve(const ve_base &, std::string &res);
  void r_velist(const std::list<ve_base> &l, std::string &res);
  void dump_curr_rend();
  int rend_renderer(const NV_rlist *, const std::string &opcode, std::string &res);
  // instruction builders
  const NV_tab_fields *find_tab(const char *s);
  int patch_Tab(const NV_tab_fields *, const char *);
  int check_vas(const nv_instr *, kv_field &, const std::string_view &);
  int pre_build(const nv_instr *ins);
  int re_rend();
  int patch(std::map<std::string_view, kv_field>::const_iterator, const char *);
  int patch_internal(std::map<std::string_view, kv_field>::const_iterator *, const char *, uint64_t &v);
  int parse_arg(const char *, uint64_t &v);
  int parse_signed(const char *, uint64_t &v);
  int patch_tab(std::map<std::string_view, kv_field>::const_iterator *, uint64_t v);
  // disasm input file
  void process_buf();
  void dump_ins(const NV_pair &p, size_t off);
  // flush output file, return 1 if all ok
  int flush() {
    int res = m_dis->flush();
printf("flush res %d\n", res);
    if ( !res ) { dirty = true; return 1; }
    if ( res < 0 ) return 0;
    if ( ifp ) {
      if ( 1 != fwrite(buf, block_size, 1, ifp) ) {
        fprintf(stderr, "fwrute failed, errno %d (%s)\n", errno, strerror(errno));
        return 0;
      }
    }
    // renew buffer
    memset(buf, 0, block_size);
    m_dis->init(buf, block_size);
    dirty = false;
    return 1;
  }
  // members
  FILE *ifp = nullptr;
  unsigned char *ibuf = nullptr;
  bool dirty = false;
  NV_extracted m_kv;
} g_ina;

int INA::extract_sv(const char *s, std::string_view &sv)
{
  const char *end = lspace(s);
  if ( !end ) return 0;
  sv = { s, end };
  return 1;
}

const NV_tab_fields *INA::find_tab(const char *s)
{
  // read tab idx
  auto idx = atoi(s);
  for ( auto &kfi: s_fields ) {
    if ( kfi.second.type == KV_TAB && kfi.second.tab_idx == idx ) return kfi.second.t;
  }
  printf("cannot find table %s\n", s);
  return nullptr;
}

int INA::parse_signed(const char *s, uint64_t &v)
{
  bool minus = false;
  if ( s[0] == '-' ) { minus = true; s++; }
  char *end = nullptr;
  long sv;
  // check hex
  if ( s[0] == '0' && s[1] == 'x' ) { // 0x
    sv = strtoll(s + 2, &end, 16);
    if ( *end ) {
      printf("bad hex const %s\n", s);
      return 0;
    }
  } else {
    sv = strtoll(s, &end, 10);
    if ( *end ) {
      printf("bad const %s\n", s);
      return 0;
    }
  }
  if ( minus ) v = -sv;
  else v = sv;
  return 1;
}

int INA::parse_arg(const char *s, uint64_t &v)
{
  if ( s[0] == 0 && s[1] == 'b' ) { // 0b
    v = 0;
    for ( int i = 2; s[i]; i++ ) {
      if ( s[i] == '1' ) v = (v << 1) | 1;
      else if ( s[i] == '0' ) v <<= 1;
      else {
        printf("bad binary const %s\n", s);
        return 0;
      }
    }
    return 1;
  }
  char *end = nullptr;
  // check hex
  if ( s[0] == '0' && s[1] == 'x' ) { // 0x
    v = strtoull(s + 2, &end, 16);
    if ( *end ) {
      printf("bad hex const %s\n", s);
      return 0;
    }
    return 1;
  }
  v = strtoull(s, &end, 10);
  if ( *end ) {
    printf("bad const %s\n", s);
    return 0;
  }
  return 1;
}

// ensure that we can compose valid value from all table colums
int INA::patch_tab(std::map<std::string_view, kv_field>::const_iterator *what, uint64_t v)
{
  if ( opt_v ) printf("patch_tab %ld\n", v);
  const kv_field *f = &(*what)->second;
  // make array to search for
  std::vector<unsigned short> etalon;
  unsigned short pattern = (unsigned short)v;
  int idx = 0;
  for ( auto &fn: f->t->fields ) {
    if ( idx == f->field_idx ) etalon.push_back(pattern);
    else {
      auto kviter = m_kv.find(fn);
      if ( kviter == m_kv.end() ) {
        printf("bad tab, cannot find value for index %d\n", idx);
        return 0;
      }
      etalon.push_back((unsigned short)kviter->second);
    }
    idx++;
  }
  // check if we have such row in table
  int has_value = 0;
  for ( auto &tv: *f->t->tab ) {
    auto ar = tv.second;
    if ( ar[0] != etalon.size() ) continue;
    // compare
    int cmp = 1;
    for ( int i = 0; i < (int)etalon.size(); i++ ) {
      if ( i == f->field_idx && ar[1 + i] == pattern ) has_value++;
      if ( ar[1 + i] != etalon[i] ) { cmp = 0; break; }
    }
///std::for_each(ar, ar + 1 + ar[0], [](unsigned short v) { printf("%d ", v); }); printf("cmp %d\n", cmp);
    if ( !cmp ) continue;
    // ok, we have full match - lets patch
    if ( opt_v ) {
      std::for_each(ar, ar + 1 + ar[0], [](unsigned short v) { printf("%d ", v); }); printf("res %d\n", tv.first);
    }
    f->patch(tv.first, m_dis);
    return 2;
  }
  if ( !has_value ) {
    printf("no value for %ld (idx %d) in table: ", v, f->field_idx );
    std::for_each(etalon.cbegin(), etalon.cend(), [](unsigned short v) { printf("%d ", v); });
    fputc('\n', stdout);
    return 0;
  }
  // table contains value v for this column but no valid row was found
  printf("cannot find row: ");
  std::for_each(etalon.cbegin(), etalon.cend(), [](unsigned short v) { printf("%d ", v); });
  fputc('\n', stdout);
  return 2;
}

int INA::patch(std::map<std::string_view, kv_field>::const_iterator what, const char *s)
{
  uint64_t v = 0L;
  int pres = patch_internal(&what, s, v);
  if ( !pres ) return 0;
  // add to kv
  m_kv[what->first] = v;
  // try to patch if patch_internal returned 1
  if ( pres & 1 ) {
    if ( !what->second.patch(v, m_dis) ) {
      printf("patch failed\n");
      return 0;
    }
  }
  return re_rend();
}

int INA::re_rend()
{
  // get new render
  NV_res res;
  auto gres = m_dis->get(res, 0);
  if ( gres < 1 ) {
    std::string curr_mask;
    m_dis->gen_mask(curr_mask);
    printf("get failed: %d, mask %s\n", gres, curr_mask.c_str());
    return 0;
  }
  for ( auto &p: res ) {
   if ( p.first == g_instr )
   {
     auto rend = m_dis->get_rend(g_instr->n);
     if ( !rend ) {
       printf("rend failed\n");
       return 0;
     }
     std::string res;
#ifdef DEBUG
 // dump batch_t & usched_info
 auto b1 = p.second.find("batch_t");
 if ( b1 != p.second.end() ) printf("batch_t %ld\n", b1->second);
 b1 = p.second.find("usched_info");
 if ( b1 != p.second.end() ) printf("usched_info %ld\n", b1->second);
#endif
     render(rend, res, g_instr, p.second, nullptr);
     g_prompt = res;
     g_prompt += " ";
     return 1;
   }
  }
  printf("cannot update prompt\n");
  return 0;
}

// return: 1 if need to patch & add to kv store
//   2 if need just to add to kv store
//   0 if fails
int INA::patch_internal(std::map<std::string_view, kv_field>::const_iterator *what, const char *s, uint64_t &v)
{
  const kv_field *f = &(*what)->second;
  int mask_size = f->mask_len();
  if ( mask_size == 1 && f->type == KV_FIELD && !f->ea && !f->va ) {
    if ( *s == '0' ) v = 0;
    else v = 1;
    return 1;
  }
  if ( f->ea ) {
    // check that value exists for this enum
    if ( !parse_arg(s, v) ) return 0;
    auto ei = f->ea->em->find(v);
    if ( ei == f->ea->em->end() ) {
      printf("bad value %ld for enum %s\n", v, f->ea->ename);
      return 0;
    }
    if ( f->type == KV_TAB ) return patch_tab(what, v);
    if ( !f->va ) return 1;
  }
  if ( f->type == KV_TAB ) {
    if ( !parse_arg(s, v) ) return 0;
    return patch_tab(what, v);
  }
  if ( !f->va ) {
    if ( !parse_arg(s, v) ) return 0;
    if ( f->type == KV_TAB ) return patch_tab(what, v);
    if ( f->type == KV_CTRL ) return 1;
    return 1;
  }
  // input depends from format in va
  double d;
  float fd;
  switch(f->va->kind) {
    case NV_BITSET:
    case NV_UImm:
     if ( !parse_arg(s, v) ) return 0;
     return 1;
    case NV_F64Imm:
     d = atof(s);
     v = *(uint64_t *)&d;
     return 1;
    case NV_F32Imm:
     d = (float)atof(s);
     v = *(uint64_t *)&d;
     return 1;
    case NV_F16Imm:
      fd = atof(s);
      v = fp16_ieee_from_fp32_value(fd);
     return 1;
    // signed int
    case NV_SImm:
    case NV_SSImm:
    case NV_RSImm:
      if ( !parse_signed(s, v) ) return 0;
      return 1;
  }
  return 0;
}

int INA::dump_i(const char *fname)
{
  auto what = s_fields.find(fname);
  if ( what == s_fields.end() ) {
    // ok, check in ins->eas
    auto ea = find(g_instr->eas, fname);
    if ( ea )
      what = s_fields.find(ea->ea->ename);
    if ( what == s_fields.end() ) {
      printf("unknown field %s (%ld)\n", fname, s_fields.size()); return 0;
    }
  }
  if ( what->second.type == KV_CTRL ) { printf("Ctrl len 8\n"); return 1; }
  printf("MaskLen %d", what->second.mask_len());
  // check what is it
  int need_nl = 1;
  if ( what->second.va ) printf(" Format %s", s_fmts[what->second.va->kind]);
  if ( what->second.type == KV_FIELD && what->second.f->scale ) printf(" scale %d", what->second.f->scale);
  if ( what->second.ea ) { // enum
   const auto ea = what->second.ea;
   if ( ea->ignore )
    printf(" .Enum %s", ea->ename);
   else
    printf(" Enum %s", ea->ename);
   if ( ea->has_def_value ) printf(" DefVal %d", ea->def_value);
   // skip too long enumes with registers
   bool skip = !strcmp(ea->ename, "NonZeroRegister") || !strcmp(ea->ename, "NonZeroUniformRegister") ||
    !strcmp(ea->ename, "RegisterFAU") || !strcmp(ea->ename, "NonZeroRegisterFAU") ||
    !strcmp(ea->ename, "Register") || !strcmp(ea->ename, "SpecialRegister");
   if ( !skip ) {
     fputs(":\n", stdout);
     need_nl = 0;
     for ( auto &ei: *ea->em ) {
       printf(" %d\t%s\n", ei.first, ei.second);
     }
   }
  }
  if ( what->second.type == KV_TAB ) { // tab
    fputs(" TAB(", stdout);
    std::for_each( what->first.cbegin(), what->first.cend(), [&](char c){ putc(c, stdout); });
    fputs("):\n\t", stdout);
    need_nl = 0;
    // make offsets of fields names
    std::vector<int> offsets;
    int prev = 8;
    for ( size_t i = 0; i < what->second.t->fields.size(); ++i ) {
      auto fn = get_it(what->second.t->fields, i);
      offsets.push_back(prev);
      prev += fn.size() + 1;
      std::for_each( fn.cbegin(), fn.cend(), [&](char c){ putc(c, stdout); });
      fputc(' ', stdout);
    }
    fputc('\n', stdout);
    // dump whole tab
    auto tab = what->second.t->tab;
    for ( auto &titer: *tab ) {
      printf(" %d\t", titer.first);
      auto ar = titer.second;
      prev = 8;
      for ( int i = 1; i <= ar[0]; ++i ) {
        for ( int p = prev; p < offsets.at(i - 1); ++p ) fputc(' ', stdout);
        prev = offsets.at(i - 1) + printf("%d", ar[i]);
      }
      fputc('\n', stdout);
    }
  }
  if ( need_nl ) fputc('\n', stdout);
  return 1;
}

void INA::dump_kv()
{
  for ( auto &p: m_kv ) {
    auto fiter = s_fields.find(p.first);
    if ( fiter == s_fields.end() ) continue;
    switch(fiter->second.type) {
      case KV_CTRL:  fputs("Ctrl  ", stdout); break;
      case KV_CBANK: fputs("CBank ", stdout); break;
      case KV_TAB: printf("Tab%d  ", fiter->second.tab_idx); break;
      default:       fputs("      ", stdout); break;
    }
    if ( fiter->second.type != KV_CTRL ) {
      std::for_each( fiter->first.cbegin(), fiter->first.cend(), [&](char c){ putc(c, stdout); });
      putc(':', stdout);
    } else fputs(": ", stdout);
    if ( fiter->second.ea ) {
      printf("%d", (int)p.second);
      auto eiter = fiter->second.ea->em->find((int)p.second);
      if ( eiter == fiter->second.ea->em->end() ) {
        fputs(" (Invalid)", stdout);
      } else {
        printf(" (%s)", eiter->second);
        if ( fiter->second.ea->has_def_value && fiter->second.ea->def_value == (int)p.second ) fputs(" DEF", stdout);
      }
      putc('\n', stdout);
      continue;
    }
    if ( fiter->second.va ) {
      std::string fs;
      dump_value(g_instr, m_kv, p.first, fs, *fiter->second.va, p.second);
      printf("%s\n", fs.c_str());
      continue;
    }
    // wtf?
    printf("%d\n", (int)p.second);
  }
}

int INA::check_vas(const nv_instr *ins, kv_field &kf, const std::string_view &s)
{
  if ( !ins->vas ) return 0;
  auto viter = find(ins->vas, s);
  if ( !viter ) return 0;
  kf.va = viter;
  return 1;
}

int INA::patch_Tab(NV_tab_fields const *t, char const *s)
{
  uint64_t v;
  auto pres = parse_arg(s, v);
  if ( !pres ) return 0;
  // try to find row in tab
  auto row = t->tab->find(v);
  if ( row == t->tab->end() ) {
    printf("no value for %ld\n", v);
    return 0;
  }
  // patch
  if ( !m_dis->put(t->mask, t->mask_size, v) ) {
    printf("patch failed\n");
    return 0;
  }
  // patch fields in kv
  auto ar = row->second;
  if ( opt_v ) printf("v %ld:", v);
  for ( int i = 0; i < ar[0]; i++ )
  {
    auto fname = get_it(t->fields, i);
    m_kv[fname] = ar[i+1];
    if ( opt_v ) printf(" %d", ar[i+1]);
  }
  if ( opt_v ) fputc('\n', stdout);
  return re_rend();
}

int INA::pre_build(const nv_instr *ins)
{
  m_kv.clear();
  s_fields.clear();
  // try apply mask
  if ( !m_dis->set_mask(ins->mask) ) {
    printf("set_mask for %d failed\n", ins->n);
    return 0;
  }
  // for all tables put first row
  if ( ins->tab_fields.size() ) {
#ifdef DEBUG
    std::string curr_mask;
    m_dis->gen_mask(curr_mask);
    printf("check tabs: %s\n", curr_mask.c_str());
#endif
    int tab_idx = 0;
    for ( auto t: ins->tab_fields ) {
      auto tab = t->tab;
      tab_idx++;
      auto first_row = tab->find(0); // first - value, second - array for fields
      if ( first_row == tab->end() ) first_row = tab->begin();
      if ( !m_dis->put(t->mask, t->mask_size, first_row->first) ) return 0;
      auto arr = first_row->second;
      if ( arr[0] != t->fields.size() ) return 0;
      int i = 1,
          f_idx = 0;
      for ( auto &f: t->fields ) {
        m_kv[f] = arr[i++];
        kv_field curr_field(t, tab_idx, f_idx++);
        check_vas(ins, curr_field, f);
        // field in tabs can be enum also
        const nv_eattr *ea = nullptr;
        auto fiter = find(ins->eas, f);
        if ( fiter ) { ea = fiter->ea; }
        else { ea = try_by_ename(ins, f); }
        if ( ea ) curr_field.ea = ea;
        s_fields.insert_or_assign(f, curr_field);
      }
    }
  }
  // for all enums with default add those default values
  if ( ins->fields.size() ) {
#ifdef DEBUG
    std::string curr_mask;
    m_dis->gen_mask(curr_mask);
    printf("check fields: %s\n", curr_mask.c_str());
#endif
    for ( auto &f: ins->fields ) {
      kv_field curr_field(&f);
      check_vas(ins, curr_field, f.name);
      const nv_eattr *ea = nullptr;
      auto fiter = find(ins->eas, f.name);
      if ( fiter ) { ea = fiter->ea; }
      else { ea = try_by_ename(ins, f.name); }
      if ( !ea ) {
        s_fields.insert_or_assign(f.name, curr_field);
        continue;
      }
      curr_field.ea = ea;
      s_fields.insert_or_assign(f.name, curr_field);
      if ( !ea->has_def_value ) continue;
      if ( !m_dis->put(f.mask, f.mask_size, ea->def_value) ) return 0;
      m_kv[f.name] = ea->def_value;
    }
  }
  // const bank
  if ( ins->cb_field ) {
    kv_field f1(ins->cb_field, 0), f2(ins->cb_field, 1);
    check_vas(ins, f1, ins->cb_field->f1);
    check_vas(ins, f2, ins->cb_field->f2);
    s_fields.insert_or_assign(ins->cb_field->f1, f1);
    s_fields.insert_or_assign(ins->cb_field->f2, f2);
  }
  // add ctrl for 64bit
  if ( m_width == 64 ) {
    kv_field f(KV_CTRL);
    s_fields.insert_or_assign("Ctrl", f);
  }
  // check if we can extract our instruction back
  NV_res res;
  auto gres = m_dis->get(res, 0);
  if ( gres < 1 ) {
    std::string curr_mask;
    m_dis->gen_mask(curr_mask);
    printf("get failed: %d, mask %s\n", gres, curr_mask.c_str());
    return 0;
  }
  for ( auto &p: res ) {
   if ( p.first == ins )
   {
     auto rend = m_dis->get_rend(ins->n);
     if ( !rend ) return 0;
     std::string res;
     render(rend, res, ins, p.second, nullptr);
     g_prompt = res;
     g_prompt += " ";
     // finally set g_instr
     g_instr = ins;
     return 1;
   }
  }
  printf("Oops: not found\n");
  return 0;
}

void INA::dump_curr_rend()
{
  auto rend = m_dis->get_rend(g_instr->n);
  if ( !rend ) return;
  std::string form;
  if ( rend_renderer( rend, s_opcode, form ) ) printf("%s\n", form.c_str());
}

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
    if ( m_width == 64 ) {
     uint8_t op = 0, ctrl = 0;
     m_dis->get_ctrl(op, ctrl);
     if ( ctrl ) fprintf(m_out, " Ctrl %X", ctrl);
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