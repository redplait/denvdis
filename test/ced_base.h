#pragma once

#include "sass_parser.h"
#include "celf.h"

using namespace std::string_literals;

extern int opt_d, opt_h, opt_t, opt_v;

class CEd_base: public CElf<ParseSASS> {
   public:
   virtual ~CEd_base() {
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
         Err("Warning: %ld non-completed tables\n", inc_size);
       m_inc_tabs.clear();
       m_dis->flush();
       block_dirty = true;
       return flush_buf();
     }
     m_inc_tabs.clear();
     return 1;
   }
   Elf_Word m_idx = 0; // section index
   unsigned long m_obj_off = 0, // start offset of selected section (0)/function inside section
     m_obj_size = 0, // size of selected section/function
     m_file_off = 0, // offset of m_obj_off in file
     m_buf_off = -1; // offset of buf in file
   int m_bidx = 0;  // for 64 & 88 index of instruction inside block
   inline unsigned long block_offset() const {
     return m_obj_off + m_buf_off - m_file_off;
   }
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
   virtual const NV_rel *next_reloc(std::string_view &sv) const override {
     if ( !m_cur_rel ) return nullptr;
     sv = { m_cur_rsym->name.cbegin(), m_cur_rsym->name.cend() };
     return m_cur_rel;
   }
   virtual int check_off(unsigned long off) {
    return 0;
   }
   virtual int check_rel(unsigned long off) {
    return 0;
   }
   // nop instruction
   const nv_instr *m_nop = nullptr;
   const NV_rlist *m_nop_rend = nullptr;
   inline bool has_nop() const {
     return m_nop && m_nop_rend;
   }
   int _disasm(unsigned long);
   int _next_off();
   int _verify_off(unsigned long);
   int parse_num(NV_Format, std::string_view &);
   // patcher
   virtual void patch_error(const char *what) = 0;
   virtual void patch_error(const std::string_view &what) = 0;
   virtual void patch_tab_error(const char *what) = 0;
   int patch(const NV_field *nf, unsigned long v, const std::string_view &what) {
     if ( !m_dis->put(nf->mask, nf->mask_size, v) )
     {
       patch_error(what);
       return 0;
     }
     block_dirty = true;
     return 1;
   }
   int patch(const NV_field *nf, unsigned long v, const char *what) {
     if ( !m_dis->put(nf->mask, nf->mask_size, v) )
     {
       patch_error(what);
       return 0;
     }
     block_dirty = true;
     return 1;
   }
   int patch(const NV_tab_fields *tf, unsigned long v, const char *what) {
     if ( !m_dis->put(tf->mask, tf->mask_size, v) )
     {
       patch_tab_error(what);
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
   int _patch_pred(int v, bool has_not, bool flush = false);
   unsigned long value_or_def(const nv_instr *, const std::string_view &, const NV_extracted &);
   unsigned long get_def_value(const nv_instr *, const std::string_view &);
   FILE *m_cubin_fp = nullptr;
   bool block_dirty = false;
   // instr buffer
   // 64bit - 8 + 7 * 8 = 64 bytes
   // 88bit - 8 + 3 * 8 = 32 bytes
   // 128bit - just 16 bytes
   static constexpr int buf_size = 64;
   // buf better to be aligned on 8 bytes - can use dirty hack from
   // https://stackoverflow.com/questions/11558371/is-it-possible-to-align-a-particular-structure-member-in-single-byte-aligned-str
   struct {} __attribute__ ((aligned (8)));
   unsigned char buf[buf_size];
   size_t block_size = 0;
   int mask_size = 0;
   // stat for buffer write/read
   unsigned long flush_cnt = 0,
    rdr_cnt = 0;
   int flush_buf();
   // disasm results
   NV_pair curr_dis;
   void reset_ins() {
    m_rend = nullptr;
    curr_dis.first = nullptr;
    curr_dis.second.clear();
    m_inc_tabs.clear();
   }
   // just wrappers to reduce repeating typing
   inline const nv_instr *ins() const { return curr_dis.first; }
   inline const NV_extracted &cex() const { return curr_dis.second; }
   inline NV_extracted &ex() { return curr_dis.second; }
   // incompleted tabs
   std::unordered_set<const NV_tab_fields *> m_inc_tabs;
   // renderer
   const NV_rlist *m_rend = nullptr;
   // named symbols
   typedef std::unordered_map<std::string_view, const asymbol *> Ced_named;
   Ced_named m_named;
   int setup_f(Ced_named::const_iterator &, const char *fname);
   int setup_s(int s_idx);
   // allowed sections with code, key is .text section index and value is index of section with attributes
   std::unordered_map<int, int> m_code_sects;
   // key - section name, value - index
   std::unordered_map<std::string, int> m_named_cs;
   static std::regex rs_digits;
};