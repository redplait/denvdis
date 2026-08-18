#pragma once
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <vector>
#include <list>

typedef std::unordered_set<std::string_view> PTXTab;
struct PTXIns {
  const char *name;
  const PTXTab *spec_tab = nullptr;
  const char *fmt, *ops;
  int ln;
  static constexpr int MaskSize = 17;
  unsigned char mask[MaskSize];
};
typedef std::vector<const PTXIns *> PTXforms;
typedef std::unordered_map<std::string_view, PTXforms> PTXOps;
// first - start index, second - body
typedef std::pair<size_t, std::string_view> PTXDot;

struct ParseRes {
  PTXforms forms; // unfortunately there can be several
  std::list<std::string_view> types; // see https://gh.evko.io/crucible-notes/ptxas/pipeline/ptx-parser.html
  // key is index in forms->mask
  std::unordered_map<int, std::string_view> attrs;
};

// main parser class
class PTXParser {
 public:
  PTXParser(FILE *fp) {
    if ( nullptr == fp ) m_log_fp = stdout;
    else m_log_fp = fp;
    m_curr = nullptr;
    reset();
  }
  ~PTXParser() {
    if ( m_curr ) delete m_curr;
  }
  void reset() {
    m_body_start = 0;
    m_curr = nullptr;
    m_tail_ops = 0;
    m_pred = {};
    m_tail = {};
    m_body.clear();
    m_attrs.clear();
    m_forms.clear();
    if ( m_curr ) {
      delete m_curr;
      m_curr = nullptr;
    }
  }
  ParseRes *parse(std::string &, int verbose = 0);
  // dump state methods
  void dump(FILE *);
  // getters
  std::string_view &pred() { return m_pred; }
  std::string_view &tail() { return m_tail; }
 protected:
  typedef std::list<int> RemList;
  inline void rem_attrs(const RemList& rem) {
    for ( auto v: rem ) m_attrs.erase(v);
  }
  int try_split(std::string &);
  int split_body();
  const PTXforms *find_instr(int verbose);
  bool is_typep(const PTXforms *) const;
  int try_types(const PTXforms *);
  template <typename T>
  int try_types_tab(T &);
  int try_type(const char *);
  int cmp_types(const std::string_view &curr, char letter, std::list<std::string_view> &res);
  // list of filtered forms in ParseRes.forms, fill ParseRes.attrs, remove matched from m_attrs
  int fill_attrs();
  size_t m_body_start;
  ParseRes *m_curr;
  // candidates
  PTXforms m_forms;
  // instruction splits into 3 parts:
  // 1) predicate started with @
  std::string_view m_pred;
  // 2) instruction with attributes splitted with '.'
  std::string m_body;
  std::unordered_map<int, PTXDot> m_attrs;
  size_t m_attrs_lim, m_tail_ops;
  // 3) tail - unchanged
  std::string_view m_tail;
  // now owning pointer
  FILE *m_log_fp;
};