#include <unistd.h>
#include "ced_base.h"

using namespace std::string_literals;

int opt_d = 0,
  opt_h = 0,
  opt_m = 0,
  skip_final_cut = 0,
  skip_op_parsing = 0,
  opt_t = 0,
  opt_k = 0,
  opt_v = 0;

class CEd: public CEd_base {
 public:
   int process(ParseSASS::Istr *);
   inline void summary() const {
     fprintf(m_out, "%ld reads, %ld flush\n", rdr_cnt, flush_cnt);
   }
 protected:
   int m_ln = 1; // line number
   // instruction filter
   std::string m_ifname;
   bool fresh_if = false;
   bool ifiltered = false; // if true - we should ignore patching
   inline void reset_filter() {
     if ( opt_d ) printf("reset_filter: fresh_if %d ifiltered %d\n", fresh_if, ifiltered);
     fresh_if = ifiltered = false;
   }
   bool check_filter() {
     if ( opt_d ) printf("check_filter: fresh_if %d ifiltered %d\n", fresh_if, ifiltered);
     ifiltered = false;
     if ( !fresh_if ) return false;
     // ok, check instruction name in curr_dis.first
     fresh_if = false;
     if ( m_ifname.empty() ) return false;
     ifiltered = (m_ifname != curr_dis.first->name);
     if ( opt_d ) printf("check_filter: %s vs %s ifiltered %d\n", m_ifname.c_str(), curr_dis.first->name, ifiltered);
     return ifiltered;
   }
   virtual int check_off(unsigned long off) override {
     if ( m_labels.empty() ) return 0;
     auto li = m_labels.find(off);
     if ( li == m_labels.end() ) return 0;
     fprintf(m_out, "Warning: offset %lX ", off);
     if ( li->second )
       fprintf(m_out, "has label from %s\n", s_ltypes[li->second]);
     else
       fprintf(m_out, "is branch\n");
     return 1;
   }
   virtual int check_rel(unsigned long off) override {
     m_cur_rsym = nullptr;
     m_cur_rel = nullptr;
     if ( !m_cur_srels ) return 0;
     auto si = m_cur_srels->find(off);
     if ( si == m_cur_srels->end() ) return 0;
     // ups, this offset contains reloc - make warning
     fprintf(m_out, "Warning: offset %lX has reloc %d\n", off, si->second.first);
     m_cur_rel = &si->second;
     m_cur_rsym = &m_syms[si->second.second];
     return 1;
   }
   virtual void patch_error(const char *what) override {
     Err("cannot patch %s, line %d\n", what, m_ln);
   };
   virtual void patch_error(const std::string_view &what) override {
     int w_len = int(what.length());
     Err("cannot patch %.*s, line %d\n", w_len, what.data(), m_ln);
   }
   virtual void patch_tab_error(const char *what) override {
     Err("cannot patch tab value %s, line %d\n", what, m_ln);
   }
   // parsers
   int parse_s(int idx, std::string &);
   int parse_S(int idx, std::string &);
   int parse_f(int idx, std::string &);
   int parse_if(int idx, std::string &);
   int parse_tail(int idx, std::string &);
   int verify_off(unsigned long);
   int process_p(std::string &p, int idx, std::string &tail);
   void dump_ins(unsigned long off) const;
   void dump_render() const;
};

int CEd::parse_if(int idx, std::string &s)
{
  ifiltered = false;
  rstrip(s);
  int i, s_size = (int)s.size();
  for ( i = idx; i < s_size; ++i ) {
    if ( !isspace(s.at(i)) ) {
      if ( i == idx ) {
        Err("if what? %s line %d\n", s.c_str() + idx, m_ln);
        return 0;
      }
      break;
    }
  }
  if ( i == s_size ) {
    Err("not arg for if: %s line %d\n", s.c_str() + idx, m_ln);
    return 0;
  }
  m_ifname = s.c_str() + i;
  fresh_if = true;
  return 1;
}

// fn function name
int CEd::parse_f(int idx, std::string &s)
{
  rstrip(s);
  if ( !new_state() ) return 0;
  char c = s.at(idx++);
  int s_size = (int)s.size();
  if ( c != 'n' || idx >= s_size ) {
    Err("invalid syntax: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  }
  c = s.at(idx);
  if ( !isspace(c) ) {
    Err("invalid fn syntax: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  }
  for ( ++idx; idx < s_size; idx++ )
    if ( !isspace(s.at(idx)) ) break;
  if ( idx >= s_size ) {
    Err("no function name: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  }
  Ced_named::const_iterator fiter = m_named.find({ s.c_str() + idx, s.size() - idx});
  if ( fiter == m_named.end() ) {
    Err("unknown fn: %s, line %d\n", s.c_str() + idx, m_ln);
    return 0;
  }
  return setup_f(fiter, s.c_str());
}

// S from_off to_off
int CEd::parse_S(int idx, std::string &s)
{
  rstrip(s);
  if ( s.empty() ) {
    Err("invalid S syntax: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  }
  char c = s.at(idx);
  if ( isspace(c) ) { // s index
    for ( ++idx; idx < int(s.size()); idx++ )
      if ( !isspace(s.at(idx)) ) break;
    if ( idx == int(s.size()) ) {
      Err("invalid S syntax: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
  }
  char *next = nullptr;
  unsigned long from_off = strtoul(s.c_str() + idx, &next, 16);
  // read to_off
  if ( !isspace(*next) ) {
      Err("invalid S syntax: %s, line %d\n", next, m_ln);
      return 0;
  }
  for ( ; *next; next++ )
      if ( !isspace(*next) ) break;
  if ( !*next ) {
    Err("invalid S syntax: %s, line %d\n", s.c_str(), m_ln);
      return 0;
  }
  unsigned long to_off = strtoul(next, &next, 16);
  // check if we have section/function
  if ( m_state < WantOff ) {
    Err("swap(%lX, %lX) not prepared\n", from_off, to_off);
  }
  if ( !verify_off(from_off) ) return 0;
  m_state = HasOff;
  // check if both offsets are the same
  if ( to_off == from_off ) {
    Err("swap useless\n");
    return 1;
  }
  // try swap
  return swap_with(to_off);
}

// s or sn
int CEd::parse_s(int idx, std::string &s)
{
  rstrip(s);
  if ( s.empty() ) {
    Err("invalid s syntax: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  }
  int s_idx = 0;
  if ( !new_state() ) return 0;
  char c = s.at(idx);
  if ( isspace(c) ) { // s index
    for ( ++idx; idx < int(s.size()); idx++ )
      if ( !isspace(s.at(idx)) ) break;
    if ( idx == int(s.size()) ) {
      Err("invalid s syntax: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    s_idx = atoi(s.c_str() + idx);
    auto siter = m_code_sects.find(s_idx);
    if ( siter == m_code_sects.end() ) {
      Err("section %d don't have code, %s: line %d\n", s_idx, s.c_str(), m_ln);
      return 0;
    }
  }
  else if ( c != 'n' ) { // not sn - don't know what is it
    Err("unknown keyword: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  } else { // section name
    ++idx;
    for ( ++idx; idx < int(s.size()); idx++ )
      if ( !isspace(s.at(idx)) ) break;
    if ( idx == int(s.size()) ) {
      Err("invalid sn syntax: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    auto siter = m_named_cs.find({ s.c_str() + idx, s.size() - idx});
    if ( siter == m_named_cs.end() ) {
      Err("section don't have code, %s: line %d\n", s.c_str(), m_ln);
      return 0;
    }
    s_idx = siter->second;
  }
  // index of found section in s_idx
  return setup_s(s_idx);
}

int CEd::parse_tail(int idx, std::string &s)
{
  if ( ifiltered ) {
    if ( opt_v ) printf("ignore %s, line %d bcs it is filtered\n", s.c_str() + idx, m_ln);
    return 1;
  }
  rstrip(s);
  int s_size = int(s.size());
  if ( s.empty() ) {
    Err("invalid syntax: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  }
  char c = s.at(idx);
  if ( 'r' == c ) { // 'r' for replace some instruction. parser in base class ParseSASS
    for ( idx++; idx < s_size; idx++ )
    {
      c = s.at(idx);
      if ( !isspace(c) ) break;
    }
    if ( idx >= s_size ) {
      Err("invalid r syntax: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    int add_res = add(s, idx);
    if ( !add_res || m_forms.empty() ) {
      Err("cannot parse %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    const one_form *of = &m_forms.at(0);
    if ( of->label_op ) {
      Err("instructions with labels not supported, line %d\n", m_ln);
      return 1;
    }
    NV_extracted kv;
    if ( !_extract_full(kv, of) ) {
      Err("cannot extract values for %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    if ( opt_k ) dump_ops(of->instr, kv);
    copy_tail_values(of->instr, of->rend, cex(), kv);
    if ( !generic_ins(of->instr, kv) ) return 0;
    if ( !flush_buf() ) {
      Err("instr %s flush failed\n", s.c_str());
      return 0;
    }
    m_state = WantOff;
    return 1;
  } else if ( '!' == c || '@' == c ) { // [!]@digit to patch initial predicate
    bool has_not = false;
    if ( c == '!' ) {
      has_not = true;
      c = s.at(++idx);
      if ( c != '@' ) {
        Err("invalid r syntax: %s, line %d\n", s.c_str(), m_ln);
        return 0;
      }
    }
    // parse value
    if ( idx + 1 >= s_size ) {
        Err("invalid predicate syntax: %s, line %d\n", s.c_str(), m_ln);
        return 0;
    }
    return _patch_pred(atoi(s.c_str() + idx + 1), has_not, true);
  } else if ( c == 'p' ) { // actually this is hardest part, bcs
     // fields args have different formats depending from it's type - like int/float
     // field can be part of table and current value can be bad combination - for this I postpone actual patching
     // and finally field can be in const bank
    for ( idx++; idx < s_size; idx++ )
    {
      c = s.at(idx);
      if ( !isspace(c) ) break;
    }
    if ( idx >= s_size ) {
      Err("invalid p syntax: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    // extract field name - stupid stl missed copy_while algo and take_while presents in ranges only
    std::string what;
    for ( ; idx < s_size; idx++ ) {
      c = s.at(idx);
      if ( isspace(c) ) break;
      what.push_back(c);
    }
    if ( idx >= s_size ) {
      Err("invalid p syntax: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    // and skip spaces after field name in what
    for ( idx++; idx < s_size; idx++ )
    {
      c = s.at(idx);
      if ( !isspace(c) ) break;
    }
    if ( idx >= s_size ) {
      Err("invalid p syntax - where is value?: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    m_state = HasP;
    return process_p(what, idx, s);
  }
  if ( !strcmp(s.c_str() + idx, "nop") ) { // wipe-out some instruction with NOP
    if ( !m_nop ) {
      Err("warning: cannot patch nop\n");
      return 1;
    }
    NV_extracted out_res;
    copy_tail_values(ins(), m_nop_rend, cex(), out_res);
    if ( !generic_ins(m_nop, out_res) ) return 0;
    if ( !flush_buf() ) {
      Err("nop flush failed\n");
      return 0;
    }
    m_state = WantOff;
    return 1;
  }
  Err("invalid syntax: %s, line %d\n", s.c_str(), m_ln);
  return 0;
}

int CEd::process_p(std::string &p, int idx, std::string &tail)
{
  // lets try to find field with name p
  auto in_s = ins();
  const NV_tab_fields *tab = nullptr;
  const NV_field *field = nullptr;
  const nv_eattr *ea = nullptr;
  const nv_vattr *va = nullptr;
  int cb_idx = 0, tab_idx = 0;
  bool ctr = p == "Ctrl";
  if ( ctr && m_width != 64 ) {
    Err("Ctrl not supported for 88-128 bits\n");
    return 1;
  }
  const NV_cbank *cb = is_cb_field(in_s, p, cb_idx);
  if ( !ctr && !cb ) {
    tab = is_tab_field(in_s, p, tab_idx);
    if ( !tab ) {
      field = std::lower_bound(in_s->fields.begin(), in_s->fields.end(), p,
       [](const NV_field &f, const std::string &w) {
         return f.name < w;
      });
      if ( field == in_s->fields.end() ) {
        Err("unknown field %s, line %d\n", p.c_str(), m_ln);
        return 0;
      }
      // cool, some real field
      ea = find_ea(in_s, p);
      if ( !ea && in_s->vas )
        va = find(in_s->vas, p);
    }
  }
  if ( opt_d ) {
    printf("field %s: ", p.c_str());
    if ( field && field->scale )
     printf("scale %d ", field->scale);
    if ( ea )
     printf("enum %s", ea->ename);
    else if ( va )
     printf("val %s", s_fmts[va->kind]);
    if ( cb )
     printf(" cb idx %d scale %d", cb_idx, cb->scale);
    else if ( tab )
     printf(" tab idx %d with %ld items\n", tab_idx, tab->fields.size());
    fputc('\n', stdout);
  }
  m_v = 0;
  std::string_view sv = { tail.c_str() + idx, tail.size() - idx };
  int sv_len = int(sv.size());
  // try to parse
  if ( va ) {
    if ( !parse_num(va->kind, sv) ) {
     Err("cannot parse num %.*s, line %d\n", sv_len, sv.data(), m_ln);
     return 0;
    }
  } else if ( ctr ) {
     if ( !parse_num(NV_UImm, sv) ) {
      Err("cannot parse Ctrl %.*s, line %d\n", sv_len, sv.data(), m_ln);
      return 0;
    }
  } else if ( ea ) {
    // check if tail is just number - then check if it is valid for some enum
    if ( std::regex_search(sv.begin(), sv.end(), rs_digits) ) {
      parse_num(NV_SImm, sv);
 if ( opt_d ) printf("parse_num %ld\n", m_v );
      // check if this e present in ea->em
      auto ei = ea->em->find(m_v);
      if ( ei == ea->em->end() ) {
        Err("value %.*s for field %s not in enum %s, line %d\n", sv_len, sv.data(), p.c_str(), ea->ename, m_ln);
        return 1;
      }
    } else {
      if ( !m_renums ) {
        Err("no renums for field %s, enum %s, line %d\n", p.c_str(), ea->ename, m_ln);
        return 1;
      }
      auto ed = m_renums->find(ea->ename);
      if ( ed == m_renums->end() ) {
        Err("cannot find enum %s for field %s, line %d\n", ea->ename, p.c_str(), m_ln);
        return 1;
      }
      auto edi = ed->second->find(sv);
      if ( edi == ed->second->end() ) {
        Err("cannot find %.*s in enum %s for field %s, line %d\n", sv_len, sv.data(), ea->ename, p.c_str(), m_ln);
        return 1;
      }
      m_v = edi->second;
      if ( opt_d )
        fprintf(m_out, "%.*s in %s has value %ld\n", sv_len, sv.data(), ea->ename, m_v);
    }
  }
  // check how this field should be patched
  if ( ctr ) {
    if ( opt_d ) fprintf(m_out, "write Ctrl %lX\n", m_v);
    int res = m_dis->put_ctrl(m_v);
    if ( res ) block_dirty = true;
    return res;
  }
  if ( field ) {
    if ( field->scale ) m_v /= field->scale;
    if ( opt_d ) fprintf(m_out, "write field %s %lX\n", p.c_str(), m_v);
    return patch(field, m_v, p.c_str());
  }
  if ( cb ) {
    unsigned long c1 = 0, c2 = 0;
    auto kv = ex();
    if ( !cb_idx ) {
      c1 = m_v;
      // store into current kv bcs next p can patch second cbank value
      kv[cb->f1] = m_v;
      c2 = value_or_def(ins(), cb->f2, kv);
    } else {
      c2 = m_v;
      // store into current kv bcs next p can patch second cbank value
      kv[cb->f2] = m_v;
      c1 = value_or_def(ins(), cb->f1, kv);
    }
    if ( opt_d ) fprintf(m_out, "write cbank c1 %lX c2 %lX\n", c1, c2);
    return generic_cb(ins(), c1, c2);
  }
  if ( tab ) {
    // check if provided value is valid for table
    std::vector<unsigned short> tab_row;
    if ( make_tab_row(opt_v, ins(), tab, cex(), tab_row, tab_idx) ) return 0;
    tab_row[tab_idx] = (unsigned short)m_v;
    int tab_value = 0;
    if ( !ins()->check_tab(tab->tab, tab_row, tab_value) ) {
      NV_extracted &kv = ex();
      kv[p] = m_v;
      m_inc_tabs.insert(tab);
      if ( opt_v ) {
        Err("Warning: value %ld for %s invalid in table, line %d\n", m_v, p.c_str(), m_ln);
        dump_tab_fields(tab);
      }
      return 1;
    } else
     return patch(tab, tab_value, p.c_str());
  }
  Err("dont know how to patch %s, line %d\n", p.c_str(), m_ln);
  return 0;
}

int CEd::verify_off(unsigned long off)
{
  flush_buf();
  int res = _verify_off(off);
  if ( !res ) {
    reset_filter();
    return res;
  }
  check_filter();
  // dump if need
  if ( opt_d ) dump_ins(off);
  if ( opt_k ) dump_ops(curr_dis.first, cex());
  if ( opt_v ) dump_render();
  return 1;
}

int CEd::process(ParseSASS::Istr *is)
{
  std::string s;
  std::regex off("^[0-9a-f]+\\s+(.*)\\s*$", std::regex_constants::icase);
  for( ; std::getline(*is->is, s); ++m_ln ) {
    // skip empty strings
    if ( s.empty() ) continue;
    // skip comments
    char c = s.at(0);
    if ( c == '#' ) continue;
    if ( c == 'q' ) break; // q to quit - for debugging
    // s or sn
    if ( c == 's' ) {
      reset_filter();
      if ( !parse_s(1, s) ) break;
      if ( opt_d ) printf("state %d off %lX\n", m_state, m_obj_off);
      continue;
    }
    if ( c == 'f' ) {
      reset_filter();
      if ( !parse_f(1, s) ) break;
      if ( opt_d ) printf("state %d off %lX\n", m_state, m_obj_off);
      continue;
    }
    // swap - 'S'
    if ( c == 'S' ) {
      if ( !parse_S(1, s) ) break;
    }
    // if
    if ( c == 'i' && s.size() > 3 && s.at(1) == 'f' ) {
      if ( !parse_if(2, s) ) break;
      if ( opt_d ) printf("if %s\n", m_ifname.c_str());
      continue;
    }
    std::smatch matches;
    if ( std::regex_search(s, matches, off) ) {
      if ( m_state < WantOff ) {
        Err("no section/function selected: %s, line %d\n", s.c_str(), m_ln);
        break;
      }
      // extract offset
      char *end = nullptr;
      unsigned long off = strtol(s.c_str(), &end, 16);
      if ( !isspace(*end) ) {
        Err("invalid syntax: %s, line %d\n", s.c_str(), m_ln);
        break;
      }
      if ( !verify_off(off) ) break;
      m_state = HasOff;
      auto tail = matches[1].str();
      if ( !tail.empty() ) {
#ifdef DEBUG
 printf("parse_tail %s\n", tail.c_str());
#endif
        if ( !parse_tail(0, tail) ) break;
      }
      continue;
    }
    if ( m_state == HasP && isspace(c) ) {
      if ( !parse_tail(1, s) ) break;
      continue;
    }
    Err("unknown command %s, state %d, line %d\n", s.c_str(), m_state, m_ln);
    break;
  }
  return new_state();
}

void CEd::dump_render() const
{
  std::string r;
  rend_rendererE(curr_dis.first, m_rend, r);
  printf("%s\n", r.c_str());
}

void CEd::dump_ins(unsigned long off) const
{
  std::string r;
  int miss = render(m_rend, r, curr_dis.first, cex(), nullptr, 1);
  if ( miss ) {
   fprintf(m_out, "; %d missed:", miss);
   for ( auto &ms: m_missed ) fprintf(m_out, " %s", ms.c_str());
   fputc('\n', m_out);
  }
  fprintf(m_out, " /*%lX*/ %s\n", off, r.c_str());
  dump_predicates(curr_dis.first, cex(), "P> ");
  if ( m_width < 128 ) {
    uint8_t c = 0, op = 0;
    m_dis->get_ctrl(op, c);
    if ( op ) fprintf(m_out, " Ctrl %X op %X\n", c, op);
    else fprintf(m_out, " Ctrl %X\n", c);
  }
}

//
// main
//
void usage(const char *prog)
{
  printf("usage: %s [options] cubin [script]\n", prog);
  printf("Options:\n");
  printf(" -d - debug mode\n");
  printf(" -k - dump kv\n");
  printf(" -t - dump symbols\n");
  printf(" -v - verbose mode\n");
  exit(6);
}

int main(int argc, char **argv)
{
  int c;
  while(1) {
    c = getopt(argc, argv, "dhktv");
    if ( c == -1 ) break;
    switch(c) {
      case 'd': opt_d = 1; break;
      case 'h': opt_h = 1; break;
      case 'k': opt_k = 1; break;
      case 't': opt_t = 1; break;
      case 'v': opt_v = 1; break;
      case '?':
      default: usage(argv[0]);
    }
  }
  if ( argc == optind ) usage(argv[0]);
  CEd ced;
  elfio rdr;
  // try to open
  if ( !ced.open(&rdr, argv[optind]) ) return 5;
  if ( !ced.prepare(argv[optind]) ) return 5;
  if ( opt_v ) printf("%ld symbols\n", ced.syms_size());
  // edit script
  auto is = ParseSASS::try_open(argc == optind + 1 ? nullptr : argv[optind + 1]);
  if ( !is ) return 0;
  ced.process(is);
  if ( opt_v )
    ced.summary();
  delete is;
}
