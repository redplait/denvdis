#include <functional>
#include <algorithm>
#include <unordered_map>
#include <list>
#include <string_view>

#define NV_MASK(name, size) static const std::pair<short, short> name[size]
#define NV_ENUM(name)  static const std::unordered_map<int, const char *> name
#define NV_TAB(name)   static const std::unordered_map<int, const int *> name

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

struct nv_vattr {
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

typedef int (*nv_filter)(std::function<uint64_t(const std::pair<short, short> *, size_t)> &);
typedef std::unordered_map<std::string_view, uint64_t> NV_extracted;
typedef void (*nv_extract)(std::function<uint64_t(const std::pair<short, short> *, size_t)> &, NV_extracted &);

struct nv_instr {
 const char *name;
 const char *mask;
 int n; // number for formatting
 int line;
 int alt;
 int meaning_bits;
 const std::unordered_map<std::string_view, const nv_vattr> vas;
 const std::unordered_map<std::string_view, const nv_eattr *> eas;
 nv_filter filter;
 nv_extract extract;
};

// binary tree
struct NV_bt_node {
  bool is_leaf;
  std::list<const nv_instr *> ins;
};

struct NV_non_leaf: public NV_bt_node
{
  int bit;
  const NV_bt_node *left, *right;
};

struct NV_base_decoder {
 uint8_t opcode = 0; // highest 6 bits and the lowest 2 bits are the opcode
 uint8_t ctrl = 0; // the middle 56 bits are used to control the execution of the following 7 instructions, each
   // of which is assigned to an 8-bit control code
   // Bit 4, 5, and 7 represent shared memory, global memory, and
   // the texture cache dependency barrier, respectively.
 protected:
   inline size_t curr_off() const {
     return curr - start - 8;
   }
   const unsigned char *start = nullptr, *curr = nullptr, *end;
   int m_idx = 0;
   inline int is_inited() const
   { return (curr != nullptr) && (curr < end); }
   void _init(const unsigned char *buf, size_t size) {
     curr = buf;
     start = curr;
     end = buf + size;
     m_idx = 0;
     opcode = ctrl = 0;
   }
};

struct nv64: public NV_base_decoder {
 protected:
  const int _width = 64;
  uint64_t *value, cqword;
  inline int check_bit(int idx) const
  {
    return *value & (1L << idx);
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
  // if idx == 0 - read control word, then first opcode
  int next() {
    if ( !is_inited() ) return 0;
    if ( !m_idx ) {
      // check that we have space for least 8 64 qwords
      if ( end - curr < 8 * 8 ) return 0;
      cqword = *(uint64_t *)curr;
      curr += 8;
      // 6 bit - 3f, << 2 = 0xfc
      opcode = cqword & 0x3 | ((cqword >> (64 - 8)) & 0xfc);
    }
    // fill ctrl
    ctrl = (cqword >> (8 * m_idx + 2)) & 0xff;
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
    for ( int m = 0; m < mask_size; m++ ) {
     for ( int i = 0; i < mask[m].second; i++ ) {
       res <<= 1;
       if ( check_bit(mask[m].first + i) ) res |= 1L;
     }
    }
    return res;
  }
};

struct nv88: public NV_base_decoder {
 protected:
  const int _width = 88;
  uint64_t *value, cqword, cword;
  inline int check_bit(int idx) const
  {
    if ( idx < 64 )
      return *value & (1L << idx);
    idx -= 64;
    return cword & (1L << idx);
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
    for ( j = 0; j < 24; j++, i-- ) {
      if ( '1' == mask[i] && !(cword & m) ) return 0;
      if ( '0' == mask[i] && (cword & m) ) return 0;
      m <<= 1;
    }
    return 1;
  }
  // if idx == 0 - read control word, then first opcode
  int next() {
    if ( !is_inited() ) return 0;
    if ( !m_idx ) {
      // check that we have space for least 8 64 qwords
      if ( end - curr < 4 * 8 ) return 0;
      cqword = *(uint64_t *)curr;
      curr += 8;
    }
    // fill cword - 21 bit = 1fffff
    cword = (cqword >> (21 * m_idx)) & 0x1fffff;
    // fill value
    value = (uint64_t *)curr;
    curr += 8;
    m_idx++;
    if ( 3 == m_idx ) m_idx = 0;
    return 1;
  }
  inline uint64_t e1(short start, short len, uint64_t &res) const
  {
    for ( int i = 0; i < len; i++ ) {
      res <<= 1;
      if ( *value & (1L << (i + start)) ) res |= 1L;
    }
    return res;
  }
  uint64_t extract(const std::pair<short, short> *mask, size_t mask_size) const
  {
    uint64_t res = 0L;
    for ( int m = 0; m < mask_size; m++ ) {
     if ( mask[m].first + mask[m].second < 64 ) {
       e1(mask[m].first, mask[m].second, res);
       continue;
     }
     for ( int i = 0; i < mask[m].second; i++ ) {
       res <<= 1;
       if ( check_bit(mask[m].first + i) ) res |= 1L;
     }
    }
    return res;
  }
};

struct nv128: public NV_base_decoder {
 protected:
  const int _width = 128;
  uint64_t q1, q2;
  inline int check_bit(int idx) const
  {
    if ( idx < 64 )
      return q1 & (1L << idx);
    idx -= 64;
    return q2 & (1L << idx);
  }
  int check_mask(const char *mask) const
  {
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
    return 1;
  }
  // if idx == 0 - read control word, then first opcode
  int next() {
    if ( !is_inited() ) return 0;
    if ( end - curr < 2 * 8 ) return 0;
    q1 = *(uint64_t *)curr;
    curr += 8;
    q2 = *(uint64_t *)curr;
    curr += 8;
    return 1;
  }
  inline uint64_t e1(short start, short len, uint64_t &res) const
  {
    for ( int i = 0; i < len; i++ ) {
      res <<= 1;
      if ( q1 & (1 << (i + start)) ) res |= 1L;
    }
    return res;
  }
  inline uint64_t e2(short start, short len, uint64_t &res) const
  {
    for ( int i = 0; i < len; i++ ) {
      res <<= 1;
      if ( q2 & (1 << (i + start)) ) res |= 1L;
    }
    return res;
  }
  uint64_t extract(const std::pair<short, short> *mask, size_t mask_size) const
  {
    uint64_t res = 0L;
    for ( int m = 0; m < mask_size; m++ ) {
     if ( mask[m].first > 63 ) {
       e2(mask[m].first - 64, mask[m].second, res);
       continue;
     }
     if ( mask[m].first + mask[m].second < 64 ) {
       e1(mask[m].first, mask[m].second, res);
       continue;
     }
     // 1st part from q1, then from q2
     for ( int i = 0; i < mask[m].second; i++ ) {
       res <<= 1;
       if ( check_bit(mask[m].first + i) ) res |= 1L;
     }
    }
    return res;
  }
};

// disasm interface
struct INV_disasm {
  virtual void init(const unsigned char *buf, size_t size) = 0;
  virtual int get(std::list< std::pair<const struct nv_instr *, NV_extracted> > &) = 0;
  virtual size_t offset() const = 0;
  virtual int width() const = 0;
  virtual void get_ctrl(uint8_t &_op, uint8_t &_ctrl) const = 0;
  virtual ~INV_disasm() = default;
};

template <typename T>
struct NV_disasm: public INV_disasm, T
{
  NV_disasm(const NV_non_leaf *root)
  {
    m_root = root;
  }
  virtual int width() const { return T::_width; }
  virtual size_t offset() const { return T::curr_off(); }
  virtual void init(const unsigned char *buf, size_t size) {
    T::_init(buf, size);
  }
  virtual void get_ctrl(uint8_t &_op, uint8_t &_ctrl) const {
   _op = T::opcode;
   _ctrl = T::ctrl;
  }
  virtual int get(std::list< std::pair<const struct nv_instr *, NV_extracted> > &res)
  {
    if ( !T::next() ) return 0;
    // traverse decode tree
    std::list<const struct nv_instr *> tmp;
    rec_find(m_root, tmp);
    // boring repeating code but you can't just pass lambda to std::function ref
    std::function<uint64_t(const std::pair<short, short> *, size_t)> extr_func =
      [&](const std::pair<short, short> *m, size_t ms) { return T::extract(m, ms); };
    for ( auto i: tmp ) {
      if ( !i->filter( extr_func ) ) continue;
      NV_extracted ex_data;
      i->extract( extr_func, ex_data );
      res.push_back( { i, std::move( ex_data ) } );
    }
    return !res.empty();
  }
 protected:
  void rec_find(const NV_bt_node *curr, std::list<const nv_instr *> &res) {
    if ( curr == nullptr ) return;
    std::copy_if( curr->ins.begin(), curr->ins.end(), res.begin(), [&](const nv_instr *ins){ return T::check_mask(ins->mask); } );
    if ( curr->is_leaf )
      return;
    const NV_non_leaf *b2 = (const NV_non_leaf *)curr;
    if ( T::check_bit(b2->bit) ) rec_find(b2->right, res);
    else rec_find(b2->left, res);
  }
  const NV_non_leaf *m_root = nullptr;
};

extern "C" INV_disasm *get_sm();
