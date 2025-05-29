#include <functional>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <vector>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <climits>
#include <stdio.h>

#define NV_MASK(name, size) static const std::pair<short, short> name[size]
#define NV_ENUM(name)  static const std::unordered_map<int, const char *> name
#define NV_TAB(name)   static const std::unordered_map<int, const unsigned short *> name
#define NV_PRED(name)  static const NV_Preds name

enum NV_Format {
  NV_BITSET,
  NV_UImm,
  NV_SImm,
  NV_SSImm,
  NV_RSImm,
  NV_F64Imm,
  NV_F16Imm,
  NV_F32Imm,
};

enum NV_Brt {
 BRT_CALL = 1,
 BRT_RETURN = 2,
 BRT_BRANCH = 3,
 BRT_BRANCHOUT = 4,
};

enum NV_Scbd {
 SOURCE_RD = 1,
 SOURCE_WR = 2,
 SINK = 3,
 SOURCE_SINK_RD = 4,
 SOURCE_SINK_WR = 5,
 NON_BARRIER_INT_INST = 6,
};

enum NV_Scbd_Type {
 BARRIER_INST = 1,
 MEM_INST = 2,
 BB_ENDING_INST = 3,
};

struct nv_vattr {
 std::string_view name;
 enum NV_Format kind;
 bool has_ast; // final *
};

struct nv_eattr {
 bool ignore; // has initial /
 bool print;  // /PRINT
 bool has_def_value;
 int def_value;
 const char *ename;
 const std::unordered_map<int, const char *> *em;
};

struct nv_named_eattr {
 std::string_view name;
 const nv_eattr *ea;
};

struct nv_float_conv {
 std::string_view name;
 std::string_view fmt_var;
 short format; // NV_Format
 short v1, v2; // -1 means no value for fmt_var
};

typedef int (*nv_filter)(std::function<uint64_t(const std::pair<short, short> *, size_t)> &);
typedef std::unordered_map<std::string_view, uint64_t> NV_extracted;
struct nv_width {
  std::string_view name;
  short w;
};
typedef std::initializer_list<nv_width> NV_width;
typedef std::initializer_list<nv_float_conv> NV_conv;
typedef void (*nv_extract)(std::function<uint64_t(const std::pair<short, short> *, size_t)> &, NV_extracted &);
typedef int (*nv_pred)(const NV_extracted &);
typedef std::unordered_map<std::string_view, nv_pred> NV_Preds;
struct nv_instr;
typedef int (*nv_tabref)(const nv_instr *i, const NV_extracted &kv);
typedef std::initializer_list< std::pair<std::string_view, nv_tabref> > NV_cond_list;
typedef std::pair<const char *, const NV_cond_list *> NV_one_cond;
typedef std::initializer_list<NV_one_cond> NV_gnames; // column or row names

// fields to patch
struct NV_field {
  std::string_view name;
  const std::pair<short, short> *mask;
  int mask_size;
  int scale = 0;
};

// const bank - can have 2 or 3 masks and always 2 fields
struct NV_cbank {
  const std::pair<short, short> *mask1, *mask2, *mask3;
  int mask1_size, mask2_size, mask3_size;
  unsigned short scale;
  std::string_view f1, f2;
};

struct NV_tab_fields {
  const std::pair<short, short> *mask;
  int mask_size;
  // use nv_instr->check_tab for reverse lookup in tab
  const std::unordered_map<int, const unsigned short *> *tab;
  std::initializer_list<std::string_view> fields;
};

// tables
struct NV_tab {
  const char *name,
   *connection;
  const NV_gnames cols, rows; // names of columns & rows
  const std::initializer_list< const std::initializer_list<short> > values;
  std::optional<short> get(int col, int row) const {
    if ( row < 0 || (size_t)row >= values.size() ) return std::nullopt;
    auto &r = *( values.begin() + row );
    if ( r.size() == 1 ) return std::optional<short>(*r.begin());
    if ( col < 0 || (size_t)col >= r.size() ) return std::nullopt;
    // initializer_list miss [], so dirty hack from
    // https://stackoverflow.com/questions/17787394/why-doesnt-stdinitializer-list-provide-a-subscript-operator
    return std::optional<short>( *(r.begin() + col) );
  }
};

// refs from instruction to tabs
struct NV_tabref {
  const NV_tab *tab;
  nv_tabref filter;
  short idx;
};

typedef std::initializer_list<NV_tabref> NV_tabrefs;

struct nv_instr {
 const char *mask;
 const char *cname; // class
 const char *name;
 int line;
 short n; // number for formatting
 char alt;
 short meaning_bits;
 char brt,   // NV_Brt or 0
  scbd,      // NV_Scbd or 0
  scbd_type; // NV_Scbd_Type or 0
 const char *target_index;
 const char *cc_index;
 const char *sidl_name;
 const NV_Preds *predicated;
 const NV_conv *vf_conv;
 const NV_width *vwidth;
 const std::initializer_list<const nv_vattr> *vas;
 const std::initializer_list<const nv_named_eattr> eas;
 nv_filter filter;
 nv_extract extract;
 // table refs
 const NV_tabrefs *rows;
 const NV_tabrefs *cols;
 // patch info
 // field directly translating into some mask (value / scale if presents) sorted by name
 const std::initializer_list<const NV_field> fields;
 const std::initializer_list<const NV_tab_fields *> tab_fields;
 const NV_cbank *cb_field = nullptr; // const bank can be 0 or 1
 /* methods */
 // check values against some tab, put resulting value in res_val
 static bool check_tab(const std::unordered_map<int, const unsigned short *> *tab,
   const std::vector<unsigned short> &curr, int &res_val)
 {
   if ( !tab || !curr.size() || curr.size() > USHRT_MAX ) return false;
   auto curr_size = (unsigned short)curr.size();
   for ( auto &ti: *tab ) {
     auto &ct = ti.second;
     if ( ct[0] != curr_size ) continue;
     int idx = 0;
     for ( auto c: curr ) {
       if ( c != ct[1+idx] ) continue;
       ++idx;
     }
     res_val = ti.first;
     return true;
   }
   return false;
 }
 int count_Pr() const
 {
  static const std::unordered_set<std::string_view> prs { "Pq", "Pp", "Pa", "Pb" ,"Pc" ," Ps" ,"Plg" };
  return std::count_if(eas.begin(), eas.end(), [](const nv_named_eattr &ec) {
    auto eic = prs.find(ec.name);
    return ( eic != prs.end() );
   });
 }
} __attribute__ ((aligned (8)));

// binary tree
struct NV_bt_node {
  int m_bit;
  inline bool is_leaf() const { return m_bit & 0x10000; }
  inline int bit() const { return m_bit & 0xffff; }
  std::initializer_list<const nv_instr *> ins;
};

struct NV_non_leaf: public NV_bt_node
{
  const NV_bt_node *left, *right;
};

const uint64_t s_masks[64] = {
/*  0 */ 0x1,
/*  1 */ 0x3,
/*  2 */ 0x7,
/*  3 */ 0xf,
/*  4 */ 0x1f,
/*  5 */ 0x3f,
/*  6 */ 0x7f,
/*  7 */ 0xff,
/*  8 */ 0x1ff,
/*  9 */ 0x3ff,
/* 10 */ 0x7ff,
/* 11 */ 0xfff,
/* 12 */ 0x1fff,
/* 13 */ 0x3fff,
/* 14 */ 0x7fff,
/* 15 */ 0xffff,
/* 16 */ 0x1ffff,
/* 17 */ 0x3ffff,
/* 18 */ 0x7ffff,
/* 19 */ 0xfffff,
/* 20 */ 0x1fffff,
/* 21 */ 0x3fffff,
/* 22 */ 0x7fffff,
/* 23 */ 0xffffff,
/* 24 */ 0x1ffffff,
/* 25 */ 0x3ffffff,
/* 26 */ 0x7ffffff,
/* 27 */ 0xfffffff,
/* 28 */ 0x1fffffff,
/* 29 */ 0x3fffffff,
/* 30 */ 0x7fffffff,
/* 31 */ 0xffffffff,
/* 32 */ 0x1ffffffffL,
/* 33 */ 0x3ffffffffL,
/* 34 */ 0x7ffffffffL,
/* 35 */ 0xfffffffffL,
/* 36 */ 0x1fffffffffL,
/* 37 */ 0x3fffffffffL,
/* 38 */ 0x7fffffffffL,
/* 39 */ 0xffffffffffL,
/* 40 */ 0x1ffffffffffL,
/* 41 */ 0x3ffffffffffL,
/* 42 */ 0x7ffffffffffL,
/* 43 */ 0xfffffffffffL,
/* 44 */ 0x1fffffffffffL,
/* 45 */ 0x3fffffffffffL,
/* 46 */ 0x7fffffffffffL,
/* 47 */ 0xffffffffffffL,
/* 48 */ 0x1ffffffffffffL,
/* 49 */ 0x3ffffffffffffL,
/* 50 */ 0x7ffffffffffffL,
/* 51 */ 0xfffffffffffffL,
/* 52 */ 0x1fffffffffffffL,
/* 53 */ 0x3fffffffffffffL,
/* 54 */ 0x7fffffffffffffL,
/* 55 */ 0xffffffffffffffL,
/* 56 */ 0x1ffffffffffffffL,
/* 57 */ 0x3ffffffffffffffL,
/* 58 */ 0x7ffffffffffffffL,
/* 59 */ 0xfffffffffffffffL,
/* 60 */ 0x1fffffffffffffffL,
/* 61 */ 0x3fffffffffffffffL,
/* 62 */ 0x7fffffffffffffffL,
/* 63 */ 0xffffffffffffffffL,
};

struct NV_base_decoder {
 uint8_t opcode = 0; // highest 6 bits and the lowest 2 bits are the opcode
 uint8_t ctrl = 0; // the middle 56 bits are used to control the execution of the following 7 instructions, each
   // of which is assigned to an 8-bit control code
   // Bit 4, 5, and 7 represent shared memory, global memory, and
   // the texture cache dependency barrier, respectively.
  inline uint64_t _extract(uint64_t v, short pos, short len) const {
    return (v >> pos) & s_masks[len - 1];
  }
  inline int _check_bit(uint64_t v, int idx) const {
    // visual studio has _bitset64
    return (v >> idx) & 1;
  }
  inline uint64_t get_cword() const {
    return 0;
  }
  template <typename T>
  inline T bit_set(T number, int n, int x) {
    return (number & ~(1L << n)) | ((T)x << n);
  }
  inline uint64_t _put(uint64_t v, uint64_t what, short pos, short len) {
    what = what & (~(s_masks[len - 1] << pos)); // zero all bits in mask
    v &= s_masks[len - 1]; // make new value
    return what | (v << pos);
  }
 protected:
   inline size_t curr_off() const {
      return curr - start - 8;
   }
   inline size_t offset_next() const {
     return curr - start;
   }
   static void to_str(uint64_t v, std::string &res, int idx = 63) {
     for ( int i = idx; i >= 0; i-- )
     {
       if ( v & (1L << i) ) res.push_back('1');
       else res.push_back('0');
     }
   }
   const unsigned char *start = nullptr, *curr = nullptr, *end;
   int m_idx = 0;
   inline int is_inited() const
   { return (curr != nullptr) && (curr < end); }
   void _init(const unsigned char *buf, size_t size) {
     start = curr = buf;
     end = buf + size;
     m_idx = 0;
     opcode = ctrl = 0;
   }
};

struct nv64: public NV_base_decoder {
 protected:
  const int _width = 64;
  uint64_t *value = nullptr, *cqword = nullptr;
  void _init(const unsigned char *buf, size_t size) {
    NV_base_decoder::_init(buf, size);
    cqword = (uint64_t *)curr;
  }
  inline int check_bit(int idx) const {
    return _check_bit(*value, idx);
  }
  int check_mask(const char *mask) const
  {
    uint64_t m = 1L;
    for ( int i = 63; i >= 0; i-- ) {
      if ( '1' == mask[i] && !(*value & m) ) return 0;
      if ( '0' == mask[i] && (*value & m) ) return 0;
      m <<= 1;
    }
    return 1;
  }
  int gen_mask(std::string &res) const
  {
    if ( !is_inited() ) return 0;
    to_str(*value, res);
    return 1;
  }
  int set_mask(const char *mask) {
    if ( !is_inited() ) return 0;
    if ( !value ) value = (uint64_t *)(curr + 8);
    *value = 0L;
    uint64_t m = 1L;
    for ( int i = 63; i >= 0; i-- ) {
      if ( '1' == mask[i] ) *value |= m;
      m <<= 1;
    }
    return 1;
  }
  int flush() {
    // make cqword
    *cqword |= (ctrl & 0xff) << (8 * m_idx + 2);
    curr += 8;
    value++;
    m_idx++;
    if ( 7 == m_idx ) {
      value = nullptr;
      m_idx = 0;
      curr += 8 * 8;
      cqword = (uint64_t *)curr;
      return (curr >= end);
    }
    return 0;
  }
  // if idx == 0 - read control word, then first opcode
  int next() {
    if ( !is_inited() ) return 0;
    if ( !m_idx ) {
      // check that we have space for least 8 64 qwords
      if ( end - curr < 8 * 8 ) return 0;
      cqword = (uint64_t *)curr;
      curr += 8;
      // 6 bit - 3f, << 2 = 0xfc
      opcode = ((*cqword) & 0x3) | (((*cqword) >> (64 - 8)) & 0xfc);
    }
    // fill ctrl
    ctrl = ((*cqword) >> (8 * m_idx + 2)) & 0xff;
    // fill value
    value = (uint64_t *)curr;
    curr += 8;
    m_idx++;
    if ( 7 == m_idx ) m_idx = 0;
    return 1;
  }
  uint64_t extract(const std::pair<short, short> *mask, size_t mask_size) const
  {
    uint64_t res = 0L;
    for ( size_t m = 0; m < mask_size; m++ )
     res = (res << mask[m].second) | _extract(*value, mask[m].first, mask[m].second);
    return res;
  }
  int put(const std::pair<short, short> *mask, size_t mask_size, uint64_t v)
  {
    if ( !is_inited() ) return 0;
    for ( int m = (int)mask_size - 1; m >= 0; --m )
    {
     *value = _put(v, *value, mask[m].first, mask[m].second);
     v >>= mask[m].second;
    }
    return 1;
  }
};

struct nv88: public NV_base_decoder {
 protected:
  const int _width = 88;
  uint64_t *value = nullptr, *cqword = nullptr, cword;
  void _init(const unsigned char *buf, size_t size) {
    NV_base_decoder::_init(buf, size);
    cqword = (uint64_t *)curr;
  }
  inline uint64_t get_cword() const {
    return cword;
  }
  inline int check_bit(int idx) const
  {
    if ( idx < 64 )
     return _check_bit(*value, idx);
    return _check_bit(cword, idx - 64);
  }
  int gen_mask(std::string &res) const
  {
    if ( !is_inited() ) return 0;
    to_str(cword, res, 23);
    to_str(*value, res);
    return 1;
  }
  int set_mask(const char *mask)
  {
    if ( !is_inited() ) return 0;
    if ( !value ) value = (uint64_t *)(curr + 8);
    int j, i = 87;
    uint64_t m = 1L;
    *value = 0; cword = 0;
    for ( j = 0; j < 64; i--, j++ ) {
      if ( '1' == mask[i] ) *value |= m;
      m <<= 1;
    }
    m = 1L;
    for ( j = 0; j < 24; i--, j++ ) {
      if ( '1' == mask[i] ) cword |= m;
      m <<= 1;
    }
    return 1;
  }
  int check_mask(const char *mask) const
  {
    uint64_t m = 1L;
    int j, i = 87;
    for ( j = 0; j < 64; i--, j++ ) {
      if ( '1' == mask[i] && !(*value & m) ) return 0;
      if ( '0' == mask[i] && (*value & m) ) return 0;
      m <<= 1;
    }
    m = 1L;
    for ( j = 0; j < 24; i--, j++ ) {
      if ( '1' == mask[i] && !(cword & m) ) return 0;
      if ( '0' == mask[i] && (cword & m) ) return 0;
      m <<= 1;
    }
    return 1;
  }
  int flush() {
    *cqword |= (cword & 0x1fffff) << (21 * m_idx);
    value++;
    m_idx++;
    if ( 3 == m_idx ) {
      m_idx = 0;
      value = nullptr;
      curr += 4 * 8;
      cqword = (uint64_t *)curr;
      return (curr >= end);
    }
    return 0;
  }
  // if idx == 0 - read control word, then first opcode
  int next() {
    if ( !is_inited() ) return 0;
    if ( !m_idx ) {
      // check that we have space for least 8 64 qwords
      if ( end - curr < 4 * 8 ) return 0;
      cqword = (uint64_t *)curr;
      curr += 8;
    }
    // fill cword - 21 bit = 1fffff
    cword = ((*cqword) >> (21 * m_idx)) & 0x1fffff;
    // fill ctrl - see https://github.com/NervanaSystems/maxas/wiki/Control-Codes
    switch(m_idx) {
      case 0: ctrl = ((*cqword) & 0x00000000001e0000) >> 17;
       break;
      case 1: ctrl = ((*cqword) & 0x000003c000000000) >> 38;
       break;
      case 2: ctrl = ((*cqword) & 0x7800000000000000) >> 59;
    }
    // fill value
    value = (uint64_t *)curr;
    curr += 8;
    m_idx++;
    if ( 3 == m_idx ) m_idx = 0;
    return 1;
  }
  int put(const std::pair<short, short> *mask, size_t mask_size, uint64_t v)
  {
    if ( !is_inited() ) return 0;
    for ( int m = (int)mask_size - 1; m >= 0; --m ) {
     if ( mask[m].first + mask[m].second <= 64 ) {
       *value = _put(v, *value, mask[m].first, mask[m].second);
       v >>= mask[m].second;
       continue;
     }
     if ( mask[m].first > 63 ) {
       cword = _put(v, cword, mask[m].first - 64, mask[m].second);
       v >>= mask[m].second;
       continue;
     }
     // some value splitted on both value & cword
     for ( int i = 0; i < mask[m].second; ++i )
     {
       int idx = i + mask[m].first;
       if ( idx < 64 )
        *value = bit_set(*value, idx, v & 1);
       else
        cword = bit_set(cword, idx - 64, v & 1);
       v >>= 1;
     }
    }
    return 1;
  }
  uint64_t extract(const std::pair<short, short> *mask, size_t mask_size) const
  {
    uint64_t res = 0L;
    for ( size_t m = 0; m < mask_size; m++ ) {
     if ( mask[m].first + mask[m].second <= 64 ) {
       res = (res << mask[m].second) | _extract(*value, mask[m].first, mask[m].second);
       continue;
     }
     if ( mask[m].first > 63 ) {
       res = (res << mask[m].second) | _extract(cword, mask[m].first - 64, mask[m].second);
// printf("%X> %d %d: %X\n", cword, mask[m].first - 64, mask[m].second, res);
       continue;
     }
     uint64_t tmp = 0;
     for ( int i = 0; i < mask[m].second; i++ ) {
       if ( check_bit(mask[m].first + i) ) tmp |= 1L << i;
     }
     res = (res << mask[m].second) | tmp;
    }
// printf("%X\n", res);
    return res;
  }
};

struct nv128: public NV_base_decoder {
 protected:
  const int _width = 128;
  // from https://stackoverflow.com/questions/16088282/is-there-a-128-bit-integer-in-gcc
#ifdef __SIZEOF_INT128__
  __uint128_t q;
  inline uint64_t extract128(__uint128_t v, short pos, short len) const {
    return (v >> pos) & s_masks[len - 1];
  }
  inline __uint128_t _put128(uint64_t v, __uint128_t what, short pos, short len)
  {
    what &= ~((__uint128_t)s_masks[len - 1] << pos); // zero all bits in mask
    v &= s_masks[len - 1]; // make new value
    return what | ((__uint128_t)v << pos);
  }
#else
  uint64_t q1, q2;
#endif
  // shadow from base
  inline size_t curr_off() const {
    return curr - start - 16;
  }
  inline int check_bit(int idx) const
  {
#ifdef __SIZEOF_INT128__
    int res = (q >> idx) & 1L;
#ifdef DEBUG
 printf("check_bit %d: %d\n", idx, res);
#endif
    return res;
#else
    if ( idx < 64 )
      return _check_bit(q1, idx);
    return _check_bit(q2, idx - 64);
#endif
  }
  int gen_mask(std::string &res) const
  {
    if ( !is_inited() ) return 0;
#ifdef __SIZEOF_INT128__
    __uint128_t m = 1L;
    int i = 127;
    for ( int j = 0; j < 128; i--, j++ ) {
      if ( q & m ) res.push_back('1');
      else res.push_back('0');
      m <<= 1L;
    }
    std::reverse(res.begin(), res.end());
#else
    to_str(q2, res);
    to_str(q1, res);
#endif
    return 1;
  }
  void print_int128(const char *mask) {
    uint64_t high = q >> 64;
    uint64_t low = q & 0xFFFFFFFFFFFFFFFF;
    printf("q 0x%016lx%016lx %d\n", high, low, check_mask(mask));
  }
  int set_mask(const char *mask)
  {
    if ( !is_inited() ) return 0;
#ifdef __SIZEOF_INT128__
    __uint128_t m = 1L;
    q = 0L;
    int i = 127;
    for ( int j = 0; j < 128; i--, j++ ) {
      if ( '1' == mask[i] ) q |= m;
      m <<= 1L;
    }
#ifdef DEBUG
 print_int128(mask);
#endif
#else
    q1 = q2 = 0L;
    uint64_t m = 1L;
    int j, i = 127;
    for ( j = 0; j < 64; i--, j++ ) {
      if ( '1' == mask[i] ) q1 |= m;
      m <<= 1;
    }
    m = 1L;
    for ( j = 0; j < 64; j++, i-- ) {
      if ( '1' == mask[i] ) q2 |= m;
      m <<= 1;
    }
#endif
    return 1;
  }
  int check_mask(const char *mask) const
  {
#ifdef DEBUG
printf("check_mask %s\n", mask);
printf("%2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X %2.2X\n",
 curr[-1], curr[-2], curr[-3], curr[-4], curr[-5], curr[-6], curr[-7], curr[-8],
 curr[-9], curr[-10], curr[-11], curr[-12], curr[-13], curr[-14], curr[-15], curr[-16]
);
#endif
#ifdef __SIZEOF_INT128__
    __uint128_t m = 1L;
    int i = 127;
    for ( int j = 0; j < 128; i--, j++ ) {
      if ( '1' == mask[i] && !(q & m) ) {
#ifdef DEBUG
printf("stop1 %d\n", i);
#endif
       return 0; }
      if ( '0' == mask[i] && (q & m) ) {
#ifdef DEBUG
printf("stop0 %d\n", i);
#endif
       return 0; }
      m <<= 1L;
    }
#else
    uint64_t m = 1L;
    int j, i = 127;
    for ( j = 0; j < 64; i--, j++ ) {
      if ( '1' == mask[i] && !(q1 & m) ) return 0;
      if ( '0' == mask[i] && (q1 & m) ) return 0;
      m <<= 1;
    }
    m = 1L;
    for ( j = 0; j < 64; j++, i-- ) {
      if ( '1' == mask[i] && !(q2 & m) ) return 0;
      if ( '0' == mask[i] && (q2 & m) ) return 0;
      m <<= 1;
    }
#endif
    return 1;
  }
  int flush() {
    if ( curr >= end ) return -2;
#ifdef __SIZEOF_INT128__
    *(__uint128_t *)curr = q;
    curr += 16;
#else
    *(uint64_t *)curr = q1;
    curr += 8;
    *(uint64_t *)curr = q2;
    curr += 8;
#endif
    return (curr >= end);
  }
  int next() {
    if ( !is_inited() ) return 0;
    if ( end - curr < 2 * 8 ) return 0;
#ifdef __SIZEOF_INT128__
    q = *(__uint128_t *)curr;
    curr += 16;
#else
    q1 = *(uint64_t *)curr;
    curr += 8;
    q2 = *(uint64_t *)curr;
    curr += 8;
#endif
    return 1;
  }
  int put(const std::pair<short, short> *mask, size_t mask_size, uint64_t v)
  {
    if ( !is_inited() ) return 0;
    for ( int m = (int)mask_size - 1; m >= 0; --m ) {
#ifdef __SIZEOF_INT128__
     q = _put128(v, q, mask[m].first, mask[m].second);
     v >>= mask[m].second;
#else
     if ( mask[m].first > 63 ) {
       q2 = _put(v, q2, mask[m].first - 64, mask[m].second);
       v >>= mask[m].second;
       if ( !v ) break;
       continue;
     }
     if ( mask[m].first + mask[m].second <= 64 ) {
       q1 = _put(v, q1, mask[m].first, mask[m].second);
       v >>= mask[m].second;
       if ( !v ) break;
       continue;
     }
     // some value splitted on both q2 & q1
     for ( int i = 0; i < mask[m].second; ++i )
     {
       int idx = i + mask[m].first;
       if ( idx < 64 )
        q1 = bit_set(q1, idx, v & 1);
       else
        q2 = bit_set(q2, idx - 64, v & 1);
       v >>= 1;
     }
#endif
    }
    return 1;
  }
  uint64_t extract(const std::pair<short, short> *mask, size_t mask_size) const
  {
    uint64_t res = 0L;
    for ( size_t m = 0; m < mask_size; m++ ) {
#ifdef __SIZEOF_INT128__
     res = (res << mask[m].second) | extract128(q, mask[m].first, mask[m].second);
#else
     if ( mask[m].first > 63 ) {
       res = (res << mask[m].second) | _extract(q2, mask[m].first - 64, mask[m].second);
       continue;
     }
     if ( mask[m].first + mask[m].second <= 64 ) {
       res = (res << mask[m].second) | _extract(q1, mask[m].first, mask[m].second);
       continue;
     }
     // 1st part from q1, then from q2
     uint64_t tmp = 0;
     for ( int i = 0; i < mask[m].second; i++ ) {
       if ( check_bit(mask[m].first + i) ) tmp |= 1L << i;
     }
     res = (res << mask[m].second) | tmp;
#endif
    }
    return res;
  }
};

// renderer guts
enum NV_rend {
 R_value = 1,
 R_enum,
 R_predicate,
 R_opcode,
 R_C, R_CX,
 R_TTU,
 R_M1, // like TMA
 R_desc,
 R_mem
};

struct ve_base {
  NV_rend type;
  char pfx;
  const char *arg;
};

struct render_base
{
  NV_rend type;
  char pfx;
  char sfx;
  char mod; // !~- etc
  render_base(NV_rend _t, char _p, char _s, char _m):
   type(_t), pfx(_p), sfx(_s), mod(_m) {}
  render_base() = default;
  virtual ~render_base() = default;
};

#define NR_BORING type = t; pfx = _p; sfx = _s; mod = _m;

struct render_named: public render_base
{
  render_named() = default;
  render_named(NV_rend t, char _p, char _s, char _m, const char *_n):
   name(_n)
  { NR_BORING }
  const char *name;
};

// C: name [value] [list]
// CX: name [enum] [value]
// A: name [list]
// DESC: [enum][list]
// T: [value] - TTU
// M1: [enum]
// remaining: prefix name [list]
// for R_mem name is null
struct render_TTU: public render_base
{
  render_TTU(NV_rend t, char _p, char _s, char _m, ve_base _v)
  {
    NR_BORING
    left = _v;
  }
  ve_base left;
};

struct render_M1: public render_named
{
  render_M1(NV_rend t, char _p, char _s, char _m, const char *_n, ve_base _b)
  {
    NR_BORING
    name = _n;
    left = _b;
  }
  ve_base left;
};

struct render_C: public render_named
{
  render_C(NV_rend t, char _p, char _s, char _m, const char *_n, ve_base _b)
  {
    NR_BORING
    name = _n;
    left = _b;
  }
  ve_base left;
  std::list<ve_base> right;
};

struct render_desc: public render_base
{
  render_desc(NV_rend t, char _p, char _s, char _m, ve_base _b)
  {
    NR_BORING
    left = _b;
  }
  ve_base left;
  std::list<ve_base> right;
};

struct render_mem: public render_named
{
  render_mem(NV_rend t, char _p, char _s, char _m, const char *_n)
  {
    NR_BORING
    name = _n;
  }
  std::list<ve_base> right;
};

#undef NR_BORING

// helper to fill list
template <typename T>
struct from
{
   from(T& _f) {
     _l = &_f.right;
   }
   template <typename O>
   from<T> &push(O &&item) {
    _l->push_back( std::move(item) );
    return *this;
   }
  protected:
   std::list<ve_base> *_l;
};

#define NVREND_PUSH(c)  res->push_back( c );

// per-instruction object
struct NV_rlist: public std::list<render_base *> {
 ~NV_rlist() {
    for ( auto ptr: *this ) delete ptr;
  }
};

typedef void (*NV_fill_render)(NV_rlist *);

struct NV_one_render
{
  NV_rlist rlist;
  NV_fill_render fill;
  std::once_flag once;
  NV_one_render(NV_fill_render _f): fill(_f) {}
};

// you can't forward declare 'static' array in terrible C++
// https://stackoverflow.com/questions/936446/is-it-possible-to-forward-declare-a-static-array
extern NV_one_render ins_render[];

typedef std::vector<std::pair<const std::string_view, const std::vector<const nv_instr *> > > NV_sorted;

// disasm interface
struct INV_disasm {
  virtual void init(const unsigned char *buf, size_t size) = 0;
  virtual int get(std::vector< std::pair<const struct nv_instr *, NV_extracted> > &, int do_next = 1) = 0;
  // reverse method of check_mask - generate mask from currently instruction, for -N option
  virtual int gen_mask(std::string &) = 0;
  virtual size_t offset() const = 0;
  virtual size_t off_next() const = 0;
  virtual int width() const = 0;
  virtual void get_ctrl(uint8_t &_op, uint8_t &_ctrl) const = 0;
  virtual uint64_t get_cword() const = 0;
  virtual const NV_rlist *get_rend(int idx) const = 0;
  virtual const NV_sorted *get_instrs() const = 0;
  // patch methods
  virtual int set_mask(const char *) = 0;
  virtual int put(const std::pair<short, short> *, size_t, uint64_t v) = 0;
  virtual int put_ctrl(uint8_t op) { return 0; } // has sense only for 64bit opcodes
  virtual int put_opcode(uint8_t op) { return 0; } // has sense only for 64bit opcodes
  virtual int flush() = 0; // warning - it's like next, so use eiter get(.., 1) or chain of put & finally flush
    // it return 1 if you need to really store you buffer and negative number on error
  virtual ~INV_disasm() = default;
  int rz;
};

template <typename T>
struct dirty_trait {
 static const bool value = false;
};

template <>
struct dirty_trait<nv64> {
 static const bool value = true;
};

template <typename T>
struct NV_disasm: public INV_disasm, T
{
  NV_disasm(const NV_non_leaf *root, int _rz, int cnt, const NV_sorted *ins)
  {
    rz = _rz;
    m_root = root;
    m_cnt = cnt;
    m_instr = ins;
  }
  virtual int width() const { return T::_width; }
  virtual size_t offset() const { return T::curr_off(); }
  virtual size_t off_next() const { return T::offset_next(); }
  virtual void init(const unsigned char *buf, size_t size) {
    T::_init(buf, size);
  }
  virtual void get_ctrl(uint8_t &_op, uint8_t &_ctrl) const {
   _op = T::opcode;
   _ctrl = T::ctrl;
  }
  virtual uint64_t get_cword() const {
    return T::get_cword();
  }
  virtual int gen_mask(std::string &res) {
    return T::gen_mask(res);
  }
  const NV_rlist *get_rend(int idx) const
  {
    if ( idx < 0 || idx >= m_cnt ) return nullptr;
    NV_rlist *res = &ins_render[idx].rlist;
    std::call_once(ins_render[idx].once, ins_render[idx].fill, res);
    return res;
  }
  virtual int get(std::vector< std::pair<const struct nv_instr *, NV_extracted> > &res, int do_next = 1)
  {
    if ( do_next ) {
      if ( !T::next() ) return -1;
    } else {
      if ( !T::is_inited() ) return -1;
    }
    // traverse decode tree
    std::list<const struct nv_instr *> tmp;
    rec_find(m_root, tmp);
    for ( auto i: tmp ) {
      if ( i->filter && !i->filter( extr_func ) ) continue;
      NV_extracted ex_data;
      i->extract( extr_func, ex_data );
      res.push_back( { i, std::move( ex_data ) } );
    }
    return !res.empty();
  }
  virtual const NV_sorted *get_instrs() const {
    return m_instr;
  }
  virtual int set_mask(const char *mask) {
    return T::set_mask(mask);
  }
  virtual int put(const std::pair<short, short> *mask, size_t mask_size, uint64_t v) {
    return T::put(mask, mask_size, v);
  }
  virtual int flush()
  {
    if ( !T::is_inited() ) return -1;
    return T::flush();
  }
  // these two has sense for 64bit opcodes only
  virtual int put_ctrl(uint8_t op) override {
    if constexpr ( dirty_trait<T>::value ) {
      T::ctrl = op;
      return 1;
    }
    return 0;
  }
  virtual int put_opcode(uint8_t op) override {
    if constexpr ( dirty_trait<T>::value ) {
      T::opcode = op;
      return 1;
    }
    return 0;
  }
 protected:
  int m_cnt;
  // boring repeating code but you can't just pass lambda to argument std::function&
  std::function<uint64_t(const std::pair<short, short> *, size_t)> extr_func =
      [&](const std::pair<short, short> *m, size_t ms) { return T::extract(m, ms); };
  void rec_find(const NV_bt_node *curr, std::list<const nv_instr *> &res) {
    if ( curr == nullptr ) return;
    std::copy_if( curr->ins.begin(), curr->ins.end(), std::back_inserter(res),
     [&](const nv_instr *ins){ return T::check_mask(ins->mask); } );
    if ( curr->is_leaf() )
      return;
    const NV_non_leaf *b2 = (const NV_non_leaf *)curr;
    if ( T::check_bit(b2->bit()) ) rec_find(b2->right, res);
    else rec_find(b2->left, res);
  }
  const NV_non_leaf *m_root = nullptr;
  const NV_sorted *m_instr = nullptr;
};


extern "C" INV_disasm *get_sm();
extern "C" const char *get_vq_name(int);
