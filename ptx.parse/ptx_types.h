#pragma once
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <vector>
#include <list>

// error log interface - stolen from ../test/nv_rend.h
struct NV_ELog {
  virtual void verr(const char *format, va_list *ap) = 0;
};

typedef std::unordered_set<std::string_view> PTXTab;
struct PTXIns {
  const char *name;
  const PTXTab *spec_tab = nullptr;
  const char *fmt, *ops;
  int ln;
  static constexpr int MaskSize = 17;
  unsigned char mask[MaskSize];
  // some masks presents just properties of instruction
  inline bool has_bit(int maj, int min) const {
    if ( maj < 0 || maj >= MaskSize ) return false;
    return mask[maj] & (1 << min);
  }
  // iptx.pl -kf shows 11 still unknown masks
  // some really not correspond to real table and just instruction properties
  // currently I was able to recognize 5 of them
  // WARNING: keep them sorted by mask index
  inline bool pred_dest() const {
    return has_bit(0, 6); // modify predicate register
  }
  inline bool is_wide() const {
    return has_bit(4, 4); // mul.wide & mad.wide - probably to mark IDEST2
  }
  inline bool is_signed() const {
    return has_bit(3, 5); // operands are signed
  }
  inline bool tex_addr() const {
    return has_bit(6, 2); // one of operands is texture address, highly likely cannot be correctly splitted by comma
  }
  inline bool is_txq() const {
    return has_bit(8, 5); // official doc don't show nothing special for TXQ vs txq.level
  }
};
typedef std::vector<const PTXIns *> PTXforms;
typedef std::unordered_map<std::string_view, PTXforms> PTXOps;
// first - start index, second - body
typedef std::pair<size_t, std::string_view> PTXDot;

#define PR_ATTRS_MULTIMAP

struct ParseRes {
  PTXforms forms; // unfortunately there can be several
  std::list<std::string_view> types; // see https://gh.evko.io/crucible-notes/ptxas/pipeline/ptx-parser.html
#ifdef PR_ATTRS_MULTIMAP
  std::unordered_multimap<int, std::string_view> attrs;
#else
  // first field is index in forms->mask
  std::vector<std::pair<int, std::string_view> > attrs;
#endif
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
  virtual ~PTXParser() {
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
  ParseRes *parse(std::string &, int process_tail, int verbose = 0);
  // dump state methods
  void dump(FILE *);
  // getters
  std::string_view &pred() { return m_pred; }
  const std::string &body() const { return m_body; }
  std::string_view &tail() { return m_tail; }
  const std::unordered_map<int, PTXDot> &rem_attrs() const { return m_attrs; }
  // error log interface
   NV_ELog *m_elog = nullptr;
#ifdef __GNUC__
    __attribute__ (( format( printf, 2, 3 ) ))
#endif
    void Err(const char *, ...) const;
 protected:
  bool check_op_count(const char *ops) const;
  typedef std::list<int> RemList;
  inline void rem_attrs(const RemList& rem) {
    for ( auto v: rem ) m_attrs.erase(v);
  }
  int try_split(std::string &);
  int split_body();
  const PTXforms *find_instr(int verbose);
  bool is_typep(const PTXforms *) const;
  int collect_types(const PTXforms *);
  template <typename T>
  int try_types_tab(T &);
  int try_type(const char *, int);
  int cmp_types(const std::string_view &curr, char letter, std::list<std::string_view> &res, int);
  int cmp_letter(const std::string_view &must_be, char letter);
  int cmp_type(const std::string_view &curr, char letter, const std::string_view &);
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
  // not owning pointer
  FILE *m_log_fp;
};

class PTXParser_str: public PTXParser {
 public:
   PTXParser_str(FILE *fp): PTXParser(fp) {}
   ParseRes *parse(const char *str, int process_tail, int verbose = 0) {
     if ( !str || !*str ) return nullptr;
     m_copy = str;
     return PTXParser::parse(m_copy, process_tail, verbose);
   }
 protected:
   std::string m_copy;
};
