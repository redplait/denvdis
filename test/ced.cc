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
  protected:
   /* even though this is essentially just PoC - syntax is extremely ugly and uses spaces as sign for continuation
      (to the delight of all python fans, if there are any in the wild)

      First we must select section or function (note - single section can contain several functions) to get addresses boundaries
      This can be done either
        * section index (for example obtained via readelf -S): s decimal
        * section name: sn name
        * function name: fn name
      Now we know what section to patch and state should be HasOff (with boundaries stored in m_obj_off & m_obj_size)

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
   int m_ln = 1; // line number
   Elf_Word s_idx = 0;
   unsigned long m_obj_off = 0, // start offset of selected section (0)/function inside section
     m_obj_size = 0, // size of selected section/function
     m_file_off = 0, // offset of m_obj_off in file
     m_buf_off = -1; // offset of buf in file
   const SRels *m_cur_srels = nullptr;
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
   int check_rel(unsigned long off) {
     if ( !m_cur_srels ) return 0;
     auto si = m_cur_srels->find(off);
     if ( si == m_cur_srels->end() ) return 0;
     // ups, this offset contains reloc - make warning
     fprintf(m_out, "Warning: offset %lX has reloc %d\n", off, si->second.first);
     return 1;
   }
   // nop instruction
   const nv_instr *m_nop = nullptr;
   // parsers
   int parse_s(int idx, std::string &);
   int parse_f(int idx, std::string &);
   int parse_tail(int idx, std::string &);
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
   int flush_buf();
   // named symbols
   std::unordered_map<std::string_view, const asymbol *> m_named;
   // allowed sections with code
   std::unordered_set<int> m_code_sects;
   // key - section name, value - index
   std::unordered_map<std::string, int> m_named_cs;
};

int CEd::flush_buf()
{
  if ( !m_cubin_fp || !block_dirty ) return 1;
  fseek(m_cubin_fp, m_buf_off, SEEK_SET);
  if ( 1 != fwrite(buf, block_size, 1, m_cubin_fp) ) {
    fprintf(stderr, "fwrite at %lX failed, error %d (%s)\n", m_buf_off, errno, strerror(errno));
    return 0;
  }
  block_dirty = 0;
  return 1;
}

int CEd::prepare(const char *fn)
{
  if ( !init_guts() ) return 0;
  n_sec = reader.sections.size();
  // iterate on sections
  for ( Elf_Half i = 0; i < n_sec; ++i ) {
   section *sec = reader.sections[i];
   if ( sec->get_type() == SHT_NOBITS || !sec->get_size() ) continue;
   auto sname = sec->get_name();
   if ( !strncmp(sname.c_str(), ".text.", 6) ) {
     m_code_sects.insert(i);
     m_named_cs[sname] = i;
     if ( opt_v ) printf("section %d: %s, size %lX\n", i, sname.c_str(), sec->get_size());
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
  // TODO: add here state check
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
  s_idx = fiter->second->section;
  auto siter = m_code_sects.find(s_idx);
  if ( siter == m_code_sects.end() ) {
    fprintf(stderr, "section %d don't have code, %s: line %d\n", s_idx, s.c_str(), m_ln);
    return 0;
  }
  section *sec = reader.sections[s_idx];
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
  s_idx = 0;
  // TODO: add here state check
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
  section *sec = reader.sections[s_idx];
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
  if ( s.empty() ) {
    fprintf(stderr, "invalid syntax: %s, line %d\n", s.c_str(), m_ln);
    return 0;
  }
  return 1;
}

int CEd::process(ParseSASS::Istr *is)
{
  int res = 0;
  std::string s;
  std::regex off("^[0-9a-f]+\\s+(.*)\\s*$", std::regex_constants::icase);
  for( ; std::getline(*is->is, s); ++m_ln ) {
    // skip empty strings
    if ( s.empty() ) continue;
    // skip comments
    char c = s.at(0);
    if ( c == '#' ) continue;
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
      // check that offset is valid
      if ( off < m_obj_off || off >= (m_obj_off + m_obj_size) ) {
        fprintf(stderr, "invalid offset %lX, should be within %lX - %lX, line %d\n",
          off, m_obj_off, m_obj_off + m_obj_size, m_ln);
        break;
      }
      check_rel(off);
      m_state = HasOff;
      auto tail = matches[1].str();
      if ( !tail.empty() ) {
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
  return res;
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
  auto is = ParseSASS::ParseSASS::try_open(argc == optind + 1 ? nullptr : argv[optind + 1]);
  if ( !is ) return 0;
  ced.process(is);
  delete is;
}
