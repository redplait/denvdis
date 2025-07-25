#include <unistd.h>
#include "sass_parser.h"
#include "celf.h"

using namespace std::string_literals;

int opt_d = 0,
  opt_m = 0,
  skip_final_cut = 0,
  skip_op_parsing = 0,
  opt_t = 0,
  opt_k = 0,
  opt_v = 0;

class CEd: public CElf<ParseSASS> {
  public:
   virtual ~CEd() {
     if ( m_cubin_fp ) {
      flush_buf();
      fclose(m_cubin_fp);
     }
   }
   virtual int init(const std::string &s) override {
     return 0;
   }
   int prepare(const char *fn);
   inline size_t syms_size() const {
     return m_named.size();
   }
   int process(ParseSASS::Istr *);
   inline void summary() const {
     fprintf(m_out, "%ld reads, %ld flush\n", rdr_cnt, flush_cnt);
   }
  protected:
   /* even though this is essentially just PoC - syntax is extremely ugly and uses spaces as sign for continuation
      (to the delight of all python fans, if there are any in the wild)

      First we must select section or function (note - single section can contain several functions) to get addresses boundaries
      This can be done either
        * section index (for example obtained via readelf -S): s decimal
        * section name: sn name
        * function name: fn name
      Now we know what section to patch and state should be WantOff (with boundaries stored in m_obj_off & m_obj_size)

      Next we may patch or replace instruction at some offset - to make copy-pasting from nvd/nvdisasm easier let offsets
       always be hexadecimal number without 0x prefix
      To fully replace instruction:
       offset nop
       offset r 'body of instruction in text form' - for parsing just reuse code from ParseSASS

      Or patch only some fields (names can be gathered from corresponding MD):
       offset !@predicate - bcs almost all instruction have leading predicate
       offset p field value - patch some single field

      And here is problem - some fields can constitute table and so have only limited set of values
      Lets say we have field1 & field2
      There is chance that new_value1 + old_value2 is invalid combination
      So we should postpone validation and patching - state HasP
      Syntax for patching:
       offset p (optional field value)
       tail list in form 'leading space for continuation next field value'

      Simple sample
       fn function_name
       1234 p
        Rd R12
        some_attr value
        ...
    */
   // parser state
   enum PState {
     Fresh = 0,
     WantOff = 1, // has section or function
     HasOff = 2,
     HasP = 3, // patch something, probably not finished
   };
   PState m_state = Fresh;
   int new_state() {
     if ( m_state == HasP ) {
       auto inc_size = m_inc_tabs.size();
       if ( inc_size )
         fprintf(stderr, "Warning: %ld non-completed tables\n", inc_size);
       m_inc_tabs.clear();
       m_dis->flush();
       block_dirty = true;
       return flush_buf();
     }
     m_inc_tabs.clear();
     return 1;
   }
   int m_ln = 1; // line number
   Elf_Word m_idx = 0; // section index
   unsigned long m_obj_off = 0, // start offset of selected section (0)/function inside section
     m_obj_size = 0, // size of selected section/function
     m_file_off = 0, // offset of m_obj_off in file
     m_buf_off = -1; // offset of buf in file
   const SRels *m_cur_srels = nullptr;
   const asymbol *m_cur_rsym = nullptr;
   const NV_rel *m_cur_rel = nullptr;
   // labels from EATTRs
   NV_labels m_labels;
   int setup_labels(int idx);
   int setup_srelocs(int s_idx) {
     auto si = m_srels.find(s_idx);
     if ( si == m_srels.end() ) {
       m_cur_srels = nullptr;
       return 0;
     }
     m_cur_srels = &si->second;
     if ( opt_v ) fprintf(m_out, "%ld relocs\n", m_cur_srels->size());
     return 1;
   }
   int check_off(unsigned long off) {
     if ( m_labels.empty() ) return 0;
     auto li = m_labels.find(off);
     if ( li == m_labels.end() ) return 0;
     fprintf(m_out, "Warning: offset %lX has label from %s\n", off, s_ltypes[li->second]);
     return 1;
   }
   int check_rel(unsigned long off) {
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
   virtual const NV_rel *next_reloc(std::string_view &sv) const override {
     if ( !m_cur_rel ) return nullptr;
     sv = { m_cur_rsym->name.cbegin(), m_cur_rsym->name.cend() };
     return m_cur_rel;
   }
   // nop instruction
   const nv_instr *m_nop = nullptr;
   const NV_rlist *m_nop_rend = nullptr;
   inline bool has_nop() const {
     return m_nop && m_nop_rend;
   }
   // parsers
   int parse_s(int idx, std::string &);
   int parse_f(int idx, std::string &);
   int parse_tail(int idx, std::string &);
   int verify_off(unsigned long);
   int process_p(std::string &p, int idx, std::string &tail);
   int parse_num(NV_Format, std::string_view &);
   int patch(const NV_field *nf, unsigned long v, const std::string_view &what) {
     if ( !m_dis->put(nf->mask, nf->mask_size, v) )
     {
       fputs("cannot patch ", stderr); dump_out(what);
       fprintf(stderr, ", line %d\n", m_ln);
       return 0;
     }
     block_dirty = true;
     return 1;
   }
   int patch(const NV_field *nf, unsigned long v, const char *what) {
     if ( !m_dis->put(nf->mask, nf->mask_size, v) )
     {
       fprintf(stderr, "cannot patch %s, line %d\n", what, m_ln);
       return 0;
     }
     block_dirty = true;
     return 1;
   }
   int patch(const NV_tab_fields *tf, unsigned long v, const char *what) {
     if ( !m_dis->put(tf->mask, tf->mask_size, v) )
     {
       fprintf(stderr, "cannot patch tab value %s, line %d\n", what, m_ln);
       return 0;
     }
     m_inc_tabs.erase(tf);
     block_dirty = true;
     return 1;
   }
   // generate some ins from fresh values
   // used in noping and patch from r instruction text
   int generic_ins(const nv_instr *, NV_extracted &);
   int generic_cb(const nv_instr *, unsigned long c1, unsigned long c2);
   unsigned long value_or_def(const nv_instr *, const std::string_view &, const NV_extracted &);
   unsigned long get_def_value(const nv_instr *, const std::string_view &);
   // stored cubin name
   std::string m_cubin;
   FILE *m_cubin_fp = nullptr;
   bool block_dirty = false;
   // instr buffer
   // 64bit - 8 + 7 * 8 = 64 bytes
   // 88bit - 8 + 3 * 8 = 32 bytes
   // 128bit - just 16 bytes
   static constexpr int buf_size = 64;
   unsigned char buf[buf_size];
   size_t block_size = 0;
   int mask_size = 0;
   unsigned long flush_cnt = 0,
    rdr_cnt = 0;
   int flush_buf();
   // disasm results
   NV_pair curr_dis;
   // just wrappers to reduce repeating typing
   inline const nv_instr *ins() const { return curr_dis.first; }
   inline const NV_extracted &cex() const { return curr_dis.second; }
   inline NV_extracted &ex() { return curr_dis.second; }
   // incompleted tabs
   std::unordered_set<const NV_tab_fields *> m_inc_tabs;
   // renderer
   const NV_rlist *m_rend = nullptr;
   void dump_ins(unsigned long off) const;
   void dump_render() const;
   // named symbols
   std::unordered_map<std::string_view, const asymbol *> m_named;
   // allowed sections with code, key is .text section index and value is index of section with attributes
   std::unordered_map<int, int> m_code_sects;
   // key - section name, value - index
   std::unordered_map<std::string, int> m_named_cs;
   static std::regex rs_digits;
};

std::regex CEd::rs_digits("^\\d+");

int CEd::flush_buf()
{
  if ( !m_cubin_fp || !block_dirty ) return 1;
  fseek(m_cubin_fp, m_buf_off, SEEK_SET);
  if ( 1 != fwrite(buf, block_size, 1, m_cubin_fp) ) {
    fprintf(stderr, "fwrite at %lX failed, error %d (%s)\n", m_buf_off, errno, strerror(errno));
    return 0;
  }
  flush_cnt++;
  block_dirty = 0;
  return 1;
}

int CEd::setup_labels(int idx)
{
  if ( idx == (int)m_idx ) return 1;
  // get appr attributes section
  m_labels.clear();
  auto si = m_code_sects.find(idx);
  if ( si == m_code_sects.end() || !si->second ) return 0;
  section *sec = reader.sections[si->second];
  // similar to nv_dis::_parse_attrs
  const char *data = sec->get_data();
  const char *start = data, *end = data + sec->get_size();
  while( data < end )
  {
    if ( end - data < 2 ) {
      fprintf(stderr, "bad attrs data. section %d\n", si->second);
      return 0;
    }
    char format = data[0];
    char attr = data[1];
    unsigned short a_len;
    int ltype = 0;
    switch (format)
    {
      case 1: data += 2;
        // check align
        if ( (data - start) & 0x3 ) data += 4 - ((data - start) & 0x3);
        break;
      case 2:
        data += 3;
        // check align
        if ( (data - start) & 0x1 ) data++;
       break;
      case 3:
        data += 4;
       break;
      case 4:
        a_len = *(unsigned short *)(data + 2);
        if ( attr == 0x28 ) // EIATTR_COOP_GROUP_INSTR_OFFSETS
          ltype = NVLType::Coop_grp;
        else if ( attr == 0x1c ) // EIATTR_EXIT_INSTR_OFFSETS
          ltype = NVLType::Exit;
        else if ( attr == 0x1d ) // EIATTR_S2RCTAID_INSTR_OFFSETS
          ltype = NVLType::S2Rctaid;
        else if ( attr == 0x25 ) // EIATTR_LD_CACHEMOD_INSTR_OFFSETS
          ltype = NVLType::Ld_cachemode;
        else if ( attr == 0x31 ) // EIATTR_INT_WARP_WIDE_INSTR_OFFSETS
          ltype = NVLType::Warp_wide;
        else if ( attr == 0x39 ) // EIATTR_MBARRIER_INSTR_OFFSETS
          ltype = NVLType::MBarier;
        else if ( attr == 0x47 ) // EIATTR_SW_WAR_MEMBAR_SYS_INSTR_OFFSETS
          ltype = NVLType::War_membar;
        // read offsets
        if ( ltype ) {
          fill_eaddrs(&m_labels, ltype, data, a_len);
        }
        data += 4 + a_len;
        break;
      default: fprintf(stderr, "unknown format %d, section %d off %lX (%s)\n",
        format, idx, data - start, sec->get_name().c_str());
         return 0;
    }
  }
  return 1;
}

int CEd::prepare(const char *fn)
{
  if ( !init_guts() ) return 0;
  n_sec = reader.sections.size();
  // iterate on sections to collect section with attributes
  std::unordered_map<int, int> attrs; // key - link section, value - index of section with attributes
  for ( Elf_Half i = 0; i < n_sec; ++i ) {
    section *sec = reader.sections[i];
    auto st = sec->get_type();
    if ( st == SHT_NOBITS || !sec->get_size() ) continue;
    if ( st == 0x70000000 ) attrs[sec->get_info()] = i;
  }
  // iterate on sections to collect code sections
  for ( Elf_Half i = 0; i < n_sec; ++i ) {
   section *sec = reader.sections[i];
   if ( sec->get_type() == SHT_NOBITS || !sec->get_size() ) continue;
   auto sname = sec->get_name();
   if ( !strncmp(sname.c_str(), ".text.", 6) ) {
     auto ai = attrs.find(i);
     if ( ai == attrs.end() )
       m_code_sects[i] = 0;
     else
       m_code_sects[i] = ai->second;
     m_named_cs[sname] = i;
     if ( opt_v ) {
       if ( ai == attrs.end() )
         printf("section %d: %s, size %lX\n", i, sname.c_str(), sec->get_size());
       else
         printf("section %d: %s, size %lX attrs in section %d\n", i, sname.c_str(), sec->get_size(), ai->second);
     }
   }
  }
  if ( m_code_sects.empty() ) {
   fprintf(stderr, "cannot find code sections in %s\n", fn);
   return 0;
  }
  // init block
  switch(m_width) {
    case 64: mask_size = 6; block_size = 64; break;
    case 88: mask_size = 6; block_size = 32; break;
    case 128: mask_size = 7; block_size = 16; break;
    default:
     fprintf(stderr, "Unknown width %d\n", m_width);
     return 0;
  }
  mask_size -= 3; // align to byte
  memset(buf, 0, block_size);
  fill_rels();
  // open file
  m_cubin = fn; // store filename for logging/debugging
  m_cubin_fp = fopen(fn, "r+b");
  if ( !m_cubin_fp ) {
    fprintf(stderr, "Cannot open %s, error %d (%s)\n", fn, errno, strerror(errno));
    return 0;
  }
  // lets try to find NOP
  auto il = m_dis->get_instrs();
  auto nop = std::lower_bound(il->begin(), il->end(), "NOP"s, [](const auto &pair, const std::string &w) {
   return pair.first < w;
   });
  if ( nop == il->end() ) {
   fprintf(stderr, "Warning: cannot find NOP\n");
  } else {
    m_nop = nop->second.at(0);
    m_nop_rend = m_dis->get_rend(m_nop->n);
    if ( !m_nop_rend ) {
      fprintf(stderr, "Warning: cannot find NOP render\n");
      m_nop = nullptr;
    }
  }
  // read symbols
  return _read_symbols(opt_t, [&](asymbol &sym) {
   auto find_cs = m_code_sects.find(sym.section);
   bool add = find_cs != m_code_sects.end() && (sym.type != STT_SECTION) && (sym.type != STT_FILE);
     m_syms.push_back(std::move(sym));
     if ( add ) {
       auto *last = &m_syms.back();
       m_named[last->name] = last;
     }
   });
}

// fn function name
int CEd::parse_f(int idx, std::string &s)
{
  rstrip(s);
  if ( !new_state() ) return 0;
  char c = s.at(idx++);
  int s_size = (int)s.size();
  if ( c != 'n' || idx >= s_size ) {
    fprintf(stderr, "invalid syntax: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  }
  c = s.at(idx);
  if ( !isspace(c) ) {
    fprintf(stderr, "invalid fn syntax: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  }
  for ( ++idx; idx < s_size; idx++ )
    if ( !isspace(s.at(idx)) ) break;
  if ( idx >= s_size ) {
    fprintf(stderr, "no function name: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  }
  auto fiter = m_named.find({ s.c_str() + idx, s.size() - idx});
  if ( fiter == m_named.end() ) {
    fprintf(stderr, "unknown fn: %s, line %d\n", s.c_str() + idx, m_ln);
    return 0;
  }
  auto s_idx = fiter->second->section;
  auto siter = m_code_sects.find(s_idx);
  if ( siter == m_code_sects.end() ) {
    fprintf(stderr, "section %d don't have code, %s: line %d\n", s_idx, s.c_str(), m_ln);
    return 0;
  }
  setup_labels(s_idx);
  section *sec = reader.sections[s_idx];
  m_idx = s_idx;
  m_obj_off = fiter->second->addr;
  m_obj_size = fiter->second->size;
  m_file_off = sec->get_offset() + m_obj_off;
  setup_srelocs(s_idx);
  m_state = WantOff;
  return 1;
}

// s or sn
int CEd::parse_s(int idx, std::string &s)
{
  rstrip(s);
  if ( s.empty() ) {
    fprintf(stderr, "invalid syntax: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  }
  int s_idx = 0;
  if ( !new_state() ) return 0;
  char c = s.at(idx);
  if ( isspace(c) ) { // s index
    for ( ++idx; idx < int(s.size()); idx++ )
      if ( !isspace(s.at(idx)) ) break;
    if ( idx == int(s.size()) ) {
      fprintf(stderr, "invalid s syntax: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    s_idx = atoi(s.c_str() + idx);
    auto siter = m_code_sects.find(s_idx);
    if ( siter == m_code_sects.end() ) {
      fprintf(stderr, "section %d don't have code, %s: line %d\n", s_idx, s.c_str(), m_ln);
      return 0;
    }
  }
  else if ( c != 'n' ) { // not sn - don't know what is it
    fprintf(stderr, "unknown keyword: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  } else { // section name
    ++idx;
    for ( ++idx; idx < int(s.size()); idx++ )
      if ( !isspace(s.at(idx)) ) break;
    if ( idx == int(s.size()) ) {
      fprintf(stderr, "invalid sn syntax: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    auto siter = m_named_cs.find({ s.c_str() + idx, s.size() - idx});
    if ( siter == m_named_cs.end() ) {
      fprintf(stderr, "section don't have code, %s: line %d\n", s.c_str(), m_ln);
      return 0;
    }
    s_idx = siter->second;
  }
  // index of found section in s_idx
  setup_labels(s_idx);
  section *sec = reader.sections[s_idx];
  m_idx = s_idx;
  m_obj_off = 0;
  m_obj_size = sec->get_size();
  m_file_off = sec->get_offset();
  setup_srelocs(s_idx);
  m_state = WantOff;
  return 1;
}

int CEd::parse_tail(int idx, std::string &s)
{
  rstrip(s);
  int s_size = int(s.size());
  if ( s.empty() ) {
    fprintf(stderr, "invalid syntax: %s, line %d\n", s.c_str(), m_ln);
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
      fprintf(stderr, "invalid r syntax: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    int add_res = add(s, idx);
    if ( !add_res || m_forms.empty() ) {
      fprintf(stderr, "cannot parse %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    const one_form *of = &m_forms.at(0);
    if ( of->label_op ) {
      fprintf(stderr, "instructions with labels not supported, line %d\n", m_ln);
      return 1;
    }
    NV_extracted kv;
    if ( !_extract_full(kv, of) ) {
      fprintf(stderr, "cannot extract values for %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    if ( opt_k ) dump_ops(of->instr, kv);
    copy_tail_values(of->instr, of->rend, cex(), kv);
    if ( !generic_ins(of->instr, kv) ) return 0;
    if ( !flush_buf() ) {
      fprintf(stderr, "instr %s flush failed\n", s.c_str());
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
        fprintf(stderr, "invalid r syntax: %s, line %d\n", s.c_str(), m_ln);
        return 0;
      }
    }
    // parse value
    if ( idx + 1 >= s_size ) {
        fprintf(stderr, "invalid predicate syntax: %s, line %d\n", s.c_str(), m_ln);
        return 0;
    }
    int v = atoi(s.c_str() + idx + 1);
    auto p_name = has_predicate(m_rend);
    if ( !p_name ) {
      fprintf(stderr, "instr %d don't have predicates. ignoring\n", ins()->n);
      return 1;
    }
    auto p_field = find_field(ins(), std::string_view(p_name));
    if ( !p_field ) {
      fprintf(stderr, "instr %d don't have predicate %s. ignoring\n", ins()->n, p_name);
      return 1;
    }
    // patch predicate
    if ( !patch(p_field, v, p_name) ) return 0;
    // make pred@not and find field for it
    std::string pnot = p_name;
    pnot += "@not";
    auto pnot_field = find_field(ins(), pnot);
    if ( !pnot_field ) {
      fprintf(stderr, "instr %d don't have !predicate %s. ignoring\n", ins()->n, pnot.c_str());
    } else {
      patch(pnot_field, has_not ? 1 : 0, pnot.c_str());
    }
    m_dis->flush();
    if ( !flush_buf() ) {
      fprintf(stderr, "predicate flush failed\n");
      return 0;
    }
    m_state = WantOff;
    return 1;
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
      fprintf(stderr, "invalid p syntax: %s, line %d\n", s.c_str(), m_ln);
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
      fprintf(stderr, "invalid p syntax: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    // and skip spaces after field name in what
    for ( idx++; idx < s_size; idx++ )
    {
      c = s.at(idx);
      if ( !isspace(c) ) break;
    }
    if ( idx >= s_size ) {
      fprintf(stderr, "invalid p syntax - where is value?: %s, line %d\n", s.c_str(), m_ln);
      return 0;
    }
    m_state = HasP;
    return process_p(what, idx, s);
  }
  if ( !strcmp(s.c_str() + idx, "nop") ) { // wipe-out some instruction with NOP
    if ( !m_nop ) {
      fprintf(stderr, "warning: cannot patch nop\n");
      return 1;
    }
    NV_extracted out_res;
    copy_tail_values(ins(), m_nop_rend, cex(), out_res);
    if ( !generic_ins(ins(), out_res) ) return 0;
    if ( !flush_buf() ) {
      fprintf(stderr, "nop flush failed\n");
      return 0;
    }
    m_state = WantOff;
    return 1;
  }
  fprintf(stderr, "invalid syntax: %s, line %d\n", s.c_str(), m_ln);
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
  if ( ctr && m_width == 128 ) {
    fprintf(stderr, "Ctrl not supported for 128bit\n");
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
        fprintf(stderr, "unknown field %s, line %d\n", p.c_str(), m_ln);
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
     fprintf(stderr, "cannot parse num %.*s, line %d\n", sv_len, sv.data(), m_ln);
     return 0;
    }
  } else if ( ctr ) {
     if ( !parse_num(NV_UImm, sv) ) {
      fprintf(stderr, "cannot parse Ctrl %.*s, line %d\n", sv_len, sv.data(), m_ln);
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
        fprintf(stderr, "value %.*s for field %s not in enum %s, line %d\n", sv_len, sv.data(), p.c_str(), ea->ename, m_ln);
        return 1;
      }
    } else {
      if ( !m_renums ) {
        fprintf(stderr, "no renums for field %s, enum %s, line %d\n", p.c_str(), ea->ename, m_ln);
        return 1;
      }
      auto ed = m_renums->find(ea->ename);
      if ( ed == m_renums->end() ) {
        fprintf(stderr, "cannot find enum %s for field %s, line %d\n", ea->ename, p.c_str(), m_ln);
        return 1;
      }
      auto edi = ed->second->find(sv);
      if ( edi == ed->second->end() ) {
        fprintf(stderr, "cannot find %.*s in enum %s for field %s, line %d\n", sv_len, sv.data(), ea->ename, p.c_str(), m_ln);
        return 1;
      }
      m_v = edi->second;
      if ( opt_d )
        fprintf(m_out, "%.*s in %s has value %ld\n", sv_len, sv.data(), ea->ename, m_v);
    }
  } else {
    fprintf(stderr, "unknown field %s, line %d - ignoring\n", p.c_str(), m_ln);
    return 1;
  }
  // check how this field should be patched
  if ( ctr ) {
    if ( opt_d ) fprintf(m_out, "write Ctrl %lX\n", m_v);
    return m_dis->put_ctrl(m_v);
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
        fprintf(stderr, "Warning: value %ld for %s invalid in table, line %d\n", m_v, p.c_str(), m_ln);
        dump_tab_fields(tab);
      }
      return 1;
    } else
     return patch(tab, tab_value, p.c_str());
  } else {
    fprintf(stderr, "dont know how to patch %s, line %d\n", p.c_str(), m_ln);
    return 0;
  }
  return 0;
}

// try to reuse as much code from base ParseSASS as possible
// actual value in m_v
int CEd::parse_num(NV_Format fmt, std::string_view &tail)
{
  if ( fmt == NV_BITSET && tail.at(0) == '{' ) {
    parse_bitset(1, tail);
    return 1;
  }
  m_minus = 0;
  int idx = 0;
  if ( tail.at(0) == '-' ) {
    idx++;
    m_minus = 1;
  }
  if ( fmt < NV_F64Imm ) {
    int i = 0;
    parse_digit(tail.data() + idx, i);
    if ( m_minus ) i = -i;
    m_v = i;
    return 1;
  }
  // check for inf & nan
  int tidx = idx;
  if ( tail.at(idx) == '+' ) tidx++;;
  float fl;
  if ( !strcasecmp(tail.data() + tidx, "inf") ) {
   if ( fmt == NV_F64Imm ) {
     *(double *)&m_v = m_minus ? -INFINITY: INFINITY;
     return 1;
   }
   const uint32_t positive_infinity_f32 = uint32_t(0x7F800000);
   const uint32_t negative_infinity_f32 = uint32_t(0xFF800000);
   fl = *(float *)( m_minus ? &negative_infinity_f32 : &positive_infinity_f32 );
   if ( fmt == NV_F32Imm ) {
     *(float *)&m_v = fl;
     return 1;
   } else if ( fmt == NV_F16Imm ) {
     m_v = fp16_ieee_from_fp32_value(fl);
     return 1;
   }
  } else if ( !strcasecmp(tail.data() + tidx, "nan") || !strcasecmp(tail.data() + tidx, "qnan") ) {
   if ( fmt == NV_F64Imm ) {
     *(double *)&m_v = m_minus ? -NAN: NAN;
     return 1;
   }
    const uint32_t positive_nan_f32 = uint32_t(0x7FFFFFFF);
    const uint32_t negative_nan_f32 = uint32_t(0xFFFFFFFF);
   fl = *(float *)( m_minus ? &negative_nan_f32 : &positive_nan_f32 );
   if ( fmt == NV_F32Imm ) {
     *(float *)&m_v = fl;
     return 1;
   } else if ( fmt == NV_F16Imm ) {
     m_v = fp16_ieee_from_fp32_value(fl);
     return 1;
   }
  }
  // this is floating value
  parse_float_tail(idx, tail);
  if ( m_minus ) m_d = -m_d;
  if ( fmt == NV_F64Imm )
  {
    m_v = *(uint64_t *)&m_d;
  } else if ( fmt == NV_F32Imm ) {
    float fl = (float)this->m_d;
    *(float *)&m_v = fl;
  } else if ( fmt == NV_F16Imm ) {
    m_v = fp16_ieee_from_fp32_value(float(m_d));
  } else return 0;
  return 1;
}

int CEd::generic_ins(const nv_instr *ins, NV_extracted &kv)
{
  m_inc_tabs.clear();
  if ( !m_dis->set_mask(ins->mask) ) {
    fprintf(stderr, "set_mask for %s %d failed\n", ins->name, ins->line);
    return 0;
  }
  // enum fields
  for ( auto &f: ins->fields ) {
    unsigned long v = 0;
    auto kvi = kv.find(f.name);
    if ( kvi != kv.end() )
     v = kvi->second;
    else
     v = get_def_value(ins, f.name);
    if ( f.scale ) v /= f.scale;
    m_dis->put(f.mask, f.mask_size, v);
  }
  // tabs
  if ( ins->tab_fields.size() ) {
    std::vector<unsigned short> row;
    int row_idx = 0;
    for ( auto tf: ins->tab_fields ) {
      for ( auto &sv: tf->fields ) {
        unsigned long v;
        auto kvi = kv.find(sv);
        if ( kvi != kv.end() )
         v = kvi->second;
        else
         v = get_def_value(ins, sv);
        row.push_back((unsigned short)v);
      }
      int res_val = 0;
      if ( !ins->check_tab(tf->tab, row, res_val) ) {
        fprintf(stderr, "check_tab index %d for %s %d failed\n", row_idx, ins->name, ins->line);
        return 0;
      }
      m_dis->put(tf->mask, tf->mask_size, res_val);
      row.clear();
      row_idx++;
    }
  }
  // const bank
  if ( ins->cb_field )
  {
    unsigned long c1, c2;
    auto kvi = kv.find(ins->cb_field->f1);
    if ( kvi != kv.end() )
     c1 = kvi->second;
    else
     c1 = get_def_value(ins, ins->cb_field->f1);
    kvi = kv.find(ins->cb_field->f2);
    if ( kvi != kv.end() )
     c2 = kvi->second;
    else
     c2 = get_def_value(ins, ins->cb_field->f2);
    generic_cb(ins, c1, c2);
  }
  m_dis->flush();
  block_dirty = 1;
  return 1;
}

int CEd::generic_cb(const nv_instr *ins, unsigned long c1, unsigned long c2) {
  if ( ins->cb_field->scale )
    c2 /= ins->cb_field->scale;
  // mask can have size 2 or 3. see details in ina.cc kv_field::patch method
  if ( ins->cb_field->mask3 ) {
    auto lo = c1 & 0xf;
    auto hi = (c1 >> 4) & 0xf;
    m_dis->put(ins->cb_field->mask1, ins->cb_field->mask1_size, hi);
    m_dis->put(ins->cb_field->mask2, ins->cb_field->mask2_size, lo);
    m_dis->put(ins->cb_field->mask3, ins->cb_field->mask3_size, c2);
  } else {
    // simple 2 mask
    m_dis->put(ins->cb_field->mask1, ins->cb_field->mask1_size, c1);
    m_dis->put(ins->cb_field->mask2, ins->cb_field->mask2_size, c2);
  }
  return 1;
}

unsigned long CEd::value_or_def(const nv_instr *ins, const std::string_view &s, const NV_extracted &kv)
{
  auto kvi = kv.find(s);
  if ( kvi != kv.end() ) return kvi->second;
  return get_def_value(ins, s);
}

unsigned long CEd::get_def_value(const nv_instr *ins, const std::string_view &s)
{
  if ( ins->vas ) {
    auto va = find(ins->vas, s);
    if ( va ) return va->dval;
  }
  auto ea = find_ea(ins, s);
  if ( ea && ea->has_def_value ) return ea->def_value;
  // hz - lets return zero
  return 0;
}

int CEd::verify_off(unsigned long off)
{
  m_inc_tabs.clear();
  // check that offset is valid
  if ( off < m_obj_off || off >= (m_obj_off + m_obj_size) ) {
    fprintf(stderr, "invalid offset %lX, should be within %lX - %lX, line %d\n",
       off, m_obj_off, m_obj_off + m_obj_size, m_ln);
    return 0;
  }
  // check if offset is properly aligned
  unsigned long off_mask = (1 << mask_size) - 1;
  if ( off & off_mask ) {
    fprintf(stderr, "warning: offset %lX is not aligned on 2 ^ %d (off_mask %lX)\n", off, mask_size, off_mask);
    off &= ~off_mask;
  }
  // extract index inside block
  int block_idx = 0;
  unsigned long block_off = m_file_off + (off - m_obj_off);
  if ( m_width != 128 ) {
    auto b_off = off & ~(block_size - 1);
    if ( b_off == off ) {
      fprintf(stderr, "warning: offset %lX points to Ctrl Word, change to %lX\n", off, off + 8);
      off += 8;
      block_idx = 0;
    } else {
      block_idx = (off - 8 - b_off) / 8;
    }
    block_off = m_file_off + (b_off - m_obj_off);
    if ( opt_d )
      fprintf(m_out, "block_off %lX off %lX block_idx %d\n", b_off, off, block_idx);
  }
  // check if we have reloc on real offset
  check_rel(off);
  check_off(off);
  if ( block_off != m_buf_off ) {
    // need to read new buffer
    fseek(m_cubin_fp, block_off, SEEK_SET);
    if ( 1 != fread(buf, block_size, 1, m_cubin_fp) ) {
      fprintf(stderr, "fread at %lX failed, error %d (%s)\n", m_buf_off, errno, strerror(errno));
      return 0;
    }
    rdr_cnt++;
  }
  m_buf_off = block_off;
  if ( !m_dis->init(buf, block_size, block_idx) ) {
    fprintf(stderr, "dis init failed\n");
    return 0;
  }
  // disasm instruction at offset
  NV_res res;
  int get_res = m_dis->get(res);
  if ( -1 == get_res ) {
    fprintf(stderr, "cannot disasm at offset %lX\n", off);
    return 0;
  }
  int res_idx = 0;
  if ( res.size() > 1 ) res_idx = calc_index(res, m_dis->rz);
  if ( -1 == res_idx ) {
    fprintf(stderr, "warning: ambigious instruction at %lX, has %ld formst\n", off, res.size());
    // lets choose 1st
    res_idx = 0;
  }
  // store disasm result
  curr_dis = std::move(res[res_idx]);
  m_rend = m_dis->get_rend(curr_dis.first->n);
  if ( !m_rend ) {
    fprintf(stderr, "cannot get render at %lX, n %d\n", off, curr_dis.first->n);
    return 0;
  }
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
    if ( c == 's' ) {
      if ( !parse_s(1, s) ) break;
      if ( opt_d ) printf("state %d off %lX\n", m_state, m_obj_off);
      continue;
    }
    if ( c == 'f' ) {
      if ( !parse_f(1, s) ) break;
      if ( opt_d ) printf("state %d off %lX\n", m_state, m_obj_off);
      continue;
    }
    std::smatch matches;
    if ( std::regex_search(s, matches, off) ) {
      if ( m_state < WantOff ) {
        fprintf(stderr, "no section/function selected: %s, line %d\n", s.c_str(), m_ln);
        break;
      }
      // extract offset
      char *end = nullptr;
      unsigned long off = strtol(s.c_str(), &end, 16);
      if ( !isspace(*end) ) {
        fprintf(stderr, "invalid syntax: %s, line %d\n", s.c_str(), m_ln);
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
    fprintf(stderr, "unknown command %s, state %d, line %d\n", s.c_str(), m_state, m_ln);
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
    c = getopt(argc, argv, "dktv");
    if ( c == -1 ) break;
    switch(c) {
      case 'd': opt_d = 1; break;
      case 'k': opt_k = 1; break;
      case 't': opt_t = 1; break;
      case 'v': opt_v = 1; break;
      case '?':
      default: usage(argv[0]);
    }
  }
  if ( argc == optind ) usage(argv[0]);
  CEd ced;
  // try to open
  if ( !ced.open(argv[optind]) ) return 5;
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
