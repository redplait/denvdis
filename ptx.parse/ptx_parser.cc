#include "ptx_types.h"
#include "ops.inc"
#include <string.h>
#include <algorithm>

// from https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces-state-spaces-tab
const static PTXTab SpaceTab = {
 "reg",
 "sreg",
 "const",
 "global",
 "local",
 "shared",
 "surf",
 "tex",
};

void PTXParser::dump(FILE *fp) {
  if ( !m_pred.empty() )
    fprintf(fp, "Pred: %.*s\n", m_pred.size(), m_pred.data());
  fprintf(fp, "body at %ld: %s\n", m_body_start, m_body.c_str());
  if ( !m_tail.empty() )
    fprintf(fp, "tail: %.*s\n", m_tail.size(), m_tail.data());
  if ( !m_attrs.empty() ) {
    fprintf(fp, "%ld attrs, lim %ld:\n", m_attrs.size(), m_attrs_lim);
    for ( auto p: m_attrs ) {
      fprintf(fp, " [%d] at %ld: %.*s\n", p.first, p.second.first, p.second.second.size(), p.second.second.data());
    }
  }
}

const PTXforms *PTXParser::find_instr(int verbose) {
  // try all forms from s_max_dot till 0
  for ( int i = s_max_dot + 1; i >= 0; --i ) {
    auto attr = m_attrs.find(i);
    if ( attr == m_attrs.end() ) continue;
    // form string_view
    std::string_view what = { m_body.c_str(), attr->second.first + attr->second.second.size() };
    if ( verbose )
      fprintf(m_log_fp, "try %d: %.*s\n", i, what.size(), what.data());
    auto ins = g_ops.find(what);
    if ( ins == g_ops.end() ) continue;
    // remove attrs from 0 till i
    for ( int j = 0; j <= i; ++j ) m_attrs.erase(j);
    return &ins->second;
  }
  return nullptr;
}

int PTXParser::split_body() {
  size_t curr = 0, prev = 0;
  int idx = 0;
  for ( ; curr < m_body.size(); ++curr ) {
    auto c = m_body.at(curr);
    if ( c == '.' ) {
      m_attrs[idx++] = { prev, { m_body.c_str() + prev, curr - prev } };
      prev = ++curr;
    }
  }
  if ( prev != curr ) // last
    m_attrs[idx++] = { prev, { m_body.c_str() + prev, curr - prev } };
  m_attrs_lim = idx;
  return !m_attrs.empty();
}

int PTXParser::try_split(std::string &s) {
  size_t curr;
  // strip initial spaces
  for ( curr = 0; curr < s.size(); ++curr ) {
    auto c = s.at(curr);
    if ( isspace(c) ) continue;
    if ( c == ';' ) return 0;
    m_body_start = curr;
    break;
  }
  if ( curr == s.size() ) return 0;
  auto c = s.at(curr);
  if ( c == '.' ) return 0;
  if ( c == '@' ) { // we have predicate
    if ( ++curr >= s.size() ) return 0;
    m_body_start = curr;
    for ( ; curr < s.size(); ++curr ) {
      c = s.at(curr);
      if ( isspace(c) ) break;
      if ( c == ';' ) return 0;
    }
    if ( curr == s.size() ) return 0;
    // store predicate
    m_pred = { s.data() + m_body_start, curr - m_body_start };
    if ( ++curr >= s.size() ) return 0;
    // strip spaces after predicate
    for ( ; curr < s.size(); ++curr ) {
      c = s.at(curr);
      if ( isspace(c) ) continue;
      if ( c == ';' ) return 0;
      m_body_start = curr;
      break;
    }
    if ( curr == s.size() ) return 0;
  }
  // fill body
  c = s.at(curr);
  if ( c == '.' ) return 0;
  for ( ; curr < s.size(); ++curr ) {
    c = s.at(curr);
    if ( isspace(c) ) break;
    if ( c == ';' ) return 1;
    m_body.push_back(tolower(c));
  }
  // strip spaces after body
  for ( ++curr ; curr < s.size(); ++curr ) {
    c = s.at(curr);
    if ( isspace(c) ) continue;
    if ( c == ';' ) return 1;
    break;
  }
  if ( curr == s.size() ) return 1;
  // store tail
  m_tail = { s.data() + curr, s.size() - curr };
  return !m_body.empty();
}

template <typename T>
int PTXParser::try_types_tab(T &tab) {
  RemList rem;
  for ( int i = 1; i < m_attrs_lim; ++i ) {
    auto ai = m_attrs.find(i);
    if ( ai == m_attrs.end() ) continue;
    auto typ = tab.find(ai->second.second);
    if ( typ == tab.end() ) continue;
    rem.push_back(i);
    m_curr->types.push_back(ai->second.second);
  }
  rem_attrs(rem);
  return !m_curr->types.empty();
}

int PTXParser::fill_attrs() {
  if ( !m_curr || m_curr->forms.empty() ) return 0;
  if ( m_attrs.empty() ) return 0;
  auto &first = m_curr->forms.at(0);
  decltype(first->mask) ored_mask;
  memcpy(ored_mask, first->mask, sizeof(ored_mask));
  for ( size_t i = 1; i < m_curr->forms.size(); ++i ) {
    auto &next = m_curr->forms.at(i);
    for ( size_t j = 0; j < PTXIns::MaskSize; ++j )
      ored_mask[j] |= next->mask[j];
  }
  using TabIdx = std::pair<int, const PTXTab *>;
  std::list<TabIdx> collected;
  // check spec_tab first
  if ( first->spec_tab && first->spec_tab != &s_tab_istypep ) {
    collected.push_back( { -1, first->spec_tab });
  }
  // always use SpaceTab
  collected.push_back( { -1, &SpaceTab } );
  // traverse tabs in non-zero masks
  for ( int i = 0; i < PTXIns::MaskSize; ++i ) {
    auto c = ored_mask[i];
    for ( int j = 0; j < 8; ++j ) {
      if ( !(c & (1 << j)) ) continue;
      int idx = i * 8 + j;
      if ( nullptr == s_tabs[idx] ) continue;
      collected.push_back( { idx, s_tabs[idx] } );
    }
  }
  RemList rem;
  int res = 0;
  // enum remained attrs
  for ( int i = 1; i < m_attrs_lim; ++i ) {
    auto ai = m_attrs.find(i);
    if ( ai == m_attrs.end() ) continue;
    for ( auto &coll: collected ) {
      auto found = coll.second->find( ai->second.second );
      if ( found == coll.second->end() ) continue;
      m_curr->attrs[ coll.first ] = *found;
      res++;
      rem.push_back(i);
      break;
    }
  }
  rem_attrs(rem);
  return res;
}

// try all remained attrs for types - from s_tab282F560
int PTXParser::collect_types(const PTXforms *flist) {
  return try_types_tab( is_typep(flist) ? s_tab_istypep : s_tab282F560);
}

// generate [ | ] variants
const char *make_vars(char letter, std::list<std::string_view> &res, const char *fmt) {
  auto prev = ++fmt;
  auto curr = prev;
  for ( ; *curr && *curr != ']'; ++curr ) {
    if ( *curr == '|' ) {
      res.push_back({ prev, curr - prev });
      prev = curr + 1;
    }
  }
  if ( *curr == ']' ) {
    res.push_back({ prev, curr - prev });
    return curr + 1;
  }
  return curr;
}

// types are case sensitive - there are n & N
// F - float
// I - integer
// B - ??, can be 128bit
// H - half float like f16x2
// P - pred
// E - bf16 ?
// T - tf32
// Q - can have size 8/16/32
int cmp_letter(const std::string_view &must_be, char letter) {
  char c = must_be.at(0);
  switch(letter) {
    case 'O': // istypep only
     return 1;
     break;
    case 'P':
     return must_be == "pred";
     break;
    case 'T':
     return must_be.starts_with("tf");
     break;
    case 'F':
     return must_be.at(0) == 'f';
     break;
    case 'E':
     return must_be == "bf16";
     break;
    case 'H':
      return must_be.ends_with("x2");
     break;
    case 'B':
      return c == 'b';
     break;
    case 'I':
      return c == 's' || c == 'u';
     break;
    case 'Q':
      return c == 'e' || must_be.starts_with("ue");
     break;
    default:
     fprintf(stderr, "unknown Letter %c\n", letter);
  }
  return 0;
}

int cmp_type(const std::string_view &must_be, char letter, const std::string_view &what) {
  std::string one_type;
  switch(letter) {
    case 'B':
    case 'F':
       one_type.push_back(tolower(letter));
       one_type += what;
       return one_type == must_be;
     break;
    case 'I':
      // check s & u
      one_type = "s";
      one_type += what;
      if ( one_type == must_be ) return 1;
      one_type = "u";
      one_type += what;
      if ( one_type == must_be ) return 1;
     break;
    case 'H':
      if ( what == "32" ) {
        return must_be == "f16x2" || must_be == "bf16x2" || must_be == "u16x2" || must_be == "s16x2";
      } else
       fprintf(stderr, "unknown H size %.*s\n", what.size(), what.data());
     break;
    case 'T':
      if ( what == "32" )
        return must_be == "tf32";
      fprintf(stderr, "unknown T size %.*s\n", what.size(), what.data());
     break;
    case 'E':
      if ( what == "16" ) {
        return must_be == "bf16";
      } else if ( what == "32" ) {
        return must_be == "tf32";
      } else
       fprintf(stderr, "unknown E size %.*s\n", what.size(), what.data());
     break;
    case 'Q':
      if ( what == "8" ) {
        return must_be == "e4m3" || must_be == "e5m2" || must_be == "e3m4" || must_be == "e2m3" || must_be == "e3m2" ||
          must_be == "ue8m0" || must_be == "ue4m3";
      } else if ( what == "16" ) {
        return (must_be.at(0) == 'e' || must_be.starts_with("ue")) && must_be.ends_with("x2");
      } else if ( what == "32" ) {
        return (must_be.at(0) == 'e' || must_be.starts_with("ue")) && must_be.ends_with("x4");
      } else
       fprintf(stderr, "unknown Q size %.*s\n", what.size(), what.data());
     break;
    default:
     fprintf(stderr, "unknown letter %c, %.*s\n", letter, what.size(), what.data());
  }
  return 0;
}

int PTXParser::cmp_types(const std::string_view &curr, char letter, std::list<std::string_view> &res, int verb) {
  if ( verb ) fprintf(m_log_fp, "cmp_types: %.*s\n", curr.size(), curr.data());
  for ( auto sv: res ) {
    if ( verb & 2 ) fprintf(m_log_fp, "> %c%.*s\n", letter, sv.size(), sv.data());
    if ( cmp_type(curr, letter, sv) ) return 1;
  }
  return 0;
}

int PTXParser::try_type(const char *fmt, int verb) {
  auto ti = m_curr->types.cbegin();
  const char *curr = fmt;
  char c_fmt = *curr;
  for ( ++curr; *curr; ++curr ) {
    // check L[]
    if ( *curr == '[' ) {
      std::list<std::string_view> vars;
      curr = make_vars(c_fmt, vars, curr);
      if ( !cmp_types(*ti, c_fmt, vars, verb) ) return 0;
      ++ti;
      c_fmt = 0;
      if ( !*curr ) break;
      if ( ti == m_curr->types.cend() ) return 0;
      c_fmt = *curr;
      continue;
    }
    // check L digit(s)
    if ( isdigit(*curr) ) {
      auto start = curr;
      for ( ++curr; *curr; ++curr ) {
        if ( isdigit(*curr) ) continue;
        std::string_view dig{start, curr - start };
 if ( verb & 2 ) fprintf(m_log_fp, "L<dig> %c%.*s\n", c_fmt, dig.size(), dig.data());
        if ( !cmp_type(*ti, c_fmt, dig) ) return 0;
        ++ti;
        if ( ti == m_curr->types.cend() ) return 0;
        break;
      }
      // if this is last
      if ( !*curr ) {
        std::string_view dig{start, curr - start };
 if ( verb & 2 ) fprintf(m_log_fp, "Last<dig> %c%.*s\n", c_fmt, dig.size(), dig.data());
        if ( !cmp_type(*ti, c_fmt, dig) ) return 0;
        c_fmt = 0;
        ++ti;
        break;
      }
      c_fmt = *curr;
      continue;
    }
    // Letter like IF32
    if ( c_fmt ) {
      if ( !cmp_letter(*ti, c_fmt) ) return 0;
      ++ti;
      if ( ti == m_curr->types.cend() ) return 0;
      c_fmt = *curr;
      continue;
    }
    fprintf(stderr, "unkown fmt %c\n", *curr);
  }
  // check if we have last letter
  if ( c_fmt ) {
    if ( ti == m_curr->types.cend() ) return 0;
    if ( !cmp_letter(*ti, c_fmt) ) return 0;
    ++ti;
  }
  return ti == m_curr->types.cend();
}

bool PTXParser::is_typep(const PTXforms *flist) const {
  if ( flist->empty() ) return false;
  auto f1 = flist->at(0);
  return f1->ops != nullptr && f1->ops[0] == 'O'; // strange type for istypep only
}

ParseRes *PTXParser::parse(std::string &s, int verbose) {
  reset();
  try_split(s);
  if ( m_body.empty() ) return nullptr;
  // split body by dots
  if ( !split_body() ) return nullptr;
  if ( !m_tail.empty() )
    m_tail_ops = std::count_if(m_tail.cbegin(), m_tail.cend(), [](char c) { return c == ','; });
  // try to find instruction
  auto forms = find_instr(verbose);
  if ( !forms ) return nullptr;
  if ( verbose ) {
    fprintf(m_log_fp, "%ld forms\n", forms->size());
    for ( auto &f: *forms ) {
      fprintf(m_log_fp, " line %d:", f->ln);
      if ( f->ops ) fprintf(m_log_fp, "%s\n", f->ops);
      else fputc('\n', m_log_fp);
    }
  }
  m_curr = new ParseRes;
  collect_types(forms);
  // lets select forms
  for ( auto &f: *forms ) {
    if ( !f->ops ) {
      if ( m_curr->types.empty() ) m_curr->forms.push_back(f);
      continue;
    }
    if ( m_curr->types.empty() ) continue;
    if ( verbose ) fprintf(m_log_fp, "-- try_type %s\n", f->ops);
    if ( try_type(f->ops, verbose) ) {
      if ( verbose ) fprintf(m_log_fp, "[+] matched\n");
      m_curr->forms.push_back(f);
    }
  }
  fill_attrs();
  auto res = m_curr;
  m_curr = nullptr;
  return res;
}