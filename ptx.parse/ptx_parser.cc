#include "ptx_types.h"
#include "ops.inc"

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

// try all remained attrs for types - from s_tab282F560
int PTXParser::try_types() {
  std::list<int> rem;
  for ( int i = 1; i < m_attrs_lim; ++i ) {
    auto ai = m_attrs.find(i);
    if ( ai == m_attrs.end() ) continue;
    auto typ = s_tab282F560.find(ai->second.second);
    if ( typ == s_tab282F560.end() ) continue;
    rem.push_back(i);
    m_curr->types.push_back(ai->second.second);
  }
  for ( auto v: rem ) m_attrs.erase(v);
  return !m_curr->types.empty();
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

int cmp_type(const std::string_view &must_be, char letter, const std::string_view &what) {
  std::string one_type;
  switch(letter) {
    case 'b':
    case 'f':
       one_type.push_back(letter);
       one_type += what;
       return one_type == must_be;
     break;
    case 'i':
      // check s & u
      one_type = "s";
      one_type += what;
      if ( one_type == must_be ) return 1;
      one_type = "u";
      one_type += what;
      if ( one_type == must_be ) return 1;
     break;
    case 'h':
      if ( what == "32" ) {
        return must_be == "f16x2" || must_be == "bf16x2" || must_be == "u16x2" || must_be == "s16x2";
      } else
       fprintf(stderr, "unknown H size %.*s\n", what.size(), what.data());
     break;
    case 'e':
      if ( what == "16" ) {
        return must_be == "f16" || must_be == "bf16";
      } else if ( what == "32" ) {
        return must_be == "tf32";
      } else
       fprintf(stderr, "unknown E size %.*s\n", what.size(), what.data());
     break;
    default:
     fprintf(stderr, "unknown letter %c, %.*s\n", letter, what.size(), what.data());
  }
  return 0;
}

int PTXParser::cmp_types(const std::string_view &curr, char letter, std::list<std::string_view> &res) {
  fprintf(m_log_fp, "cmp_types: %.*s\n", curr.size(), curr.data());
  for ( auto sv: res ) {
    fprintf(m_log_fp, "> %c%.*s\n", letter, sv.size(), sv.data());
    if ( cmp_type(curr, letter, sv) ) return 1;
  }
  return 0;
}

int PTXParser::try_type(const char *fmt) {
  auto ti = m_curr->types.cbegin();
  const char *curr = fmt;
  char c_fmt = tolower(*curr);
  for ( ++curr; *curr; ++curr ) {
    // check L[]
    if ( *curr == '[' ) {
      std::list<std::string_view> vars;
      curr = make_vars(c_fmt, vars, curr);
      if ( !cmp_types(*ti, c_fmt, vars) ) return 0;
      ++ti;
      if ( !*curr ) break;
      if ( ti == m_curr->types.cend() ) return 0;
    }
    // check L digit(s)
    if ( isdigit(*curr) ) {
      auto start = curr;
      for ( ++curr; *curr; ++curr ) {
        if ( isdigit(*curr) ) continue;
        std::string_view dig{start, curr - start };
 fprintf(m_log_fp, "L<dig> %c%.*s\n", c_fmt, dig.size(), dig.data());
        if ( !cmp_type(*ti, c_fmt, dig) ) return 0;
        ++ti;
        if ( ti == m_curr->types.cend() ) return 0;
        break;
      }
      // if this is last
      if ( !*curr ) {
        std::string_view dig{start, curr - start };
 fprintf(m_log_fp, "Last<dig> %c%.*s\n", c_fmt, dig.size(), dig.data());
        if ( !cmp_type(*ti, c_fmt, dig) ) return 0;
        ++ti;
        break;
      }
      c_fmt = tolower(*curr);
    }
  }
  return ti == m_curr->types.cend();
}

ParseRes *PTXParser::parse(std::string &s, int verbose) {
  reset();
  try_split(s);
  if ( m_body.empty() ) return nullptr;
  // split body by dots
  if ( !split_body() ) return nullptr;
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
  try_types();
  // lets select forms
  for ( auto &f: *forms ) {
    if ( !f->ops ) {
      if ( m_curr->types.empty() ) m_curr->forms.push_back(f);
      continue;
    }
    if ( m_curr->types.empty() ) continue;
    if ( try_type(f->ops) ) m_curr->forms.push_back(f);
  }
  auto res = m_curr;
  m_curr = nullptr;
  return res;
}