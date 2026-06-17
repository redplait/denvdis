#pragma once

struct reg_reuse {
  unsigned short mask = 0, // actual values of reuse_src_X
    mask2 = 0; // if reuse_src_X presents in this instruction
  unsigned char keep = 0, // actual values of keep_X
    keep2 = 0; // if keep_x presents in this instruction
  inline void clear() {
    mask = mask2 = 0;
  }
  int apply(const struct nv_instr *, const NV_extracted &kv);
  // 1 << (idx - ISRC_A)
  inline int ra() const { return mask & 1; }
  inline int rb() const { return mask & 2; }
  inline int rc() const { return mask & 4; }
  inline int re() const { return mask & 8; }
  inline int rh() const { return mask & 16; }
  // there are only keep_a & keep_b - both were introduced in sm100
  inline int ka() const { return keep & 1; }
  inline int kb() const { return keep & 2; }
};

struct found_tab_cross {
  const NV_tab *tab = nullptr;
  int row = 0, col = 0;
  short value = 0;
  // couple comfortable methods to avoid code duplication
  const char *row_name() const {
    if ( !tab || size_t(row) >= tab->rows.size() ) return nullptr;
    auto &sv = *(tab->rows.begin() + row);
    return sv.first;
  }
  const char *col_name() const {
    if ( !tab || size_t(col) >= tab->cols.size() ) return nullptr;
    auto &sv = *(tab->cols.begin() + col);
    return sv.first;
  }
};

typedef std::pair<const NV_tab *, int> RegTabChain;
typedef std::list<RegTabChain> RegTabChains;

std::optional<found_tab_cross> find_tab_cross(const RegTabChains &rows, const RegTabChains &cols);

struct reg_history {
  unsigned long off;
  // 0x8000 - write, else read
  // 0x4000 - Uniform predicate, else just predicate
  // next 3 bits are predicate reg index + 1 (bcs T == 7 and 0 is perfectly valid predicate)
  // next 1 bit - if was load from Special Reg (1 << 10)
  // next 1 bit - reuse flag (1 << 9)
  // next 1 bit - part of compound (1 << 8)
  // next 1 bit - list of compound (1 << 7)
  // next 3 bit is index for wide operation - it can be up to 256 bit (like SRC_I) / 32 = 8
  // finally low 4 bit is NVP_ops
  typedef unsigned short RH;
  static constexpr RH reuse = 1 << 9;
  static constexpr RH comp  = 1 << 8;
  static constexpr RH in_list = 1 << 7;
  RH kind;
  inline bool is_upred() const {
    return kind & 0x4000;
  }
  inline bool is_reuse() const {
    return kind & reuse;
  }
  inline bool has_pred(int &p) const {
    p = (kind >> 11) & 0x7;
    if ( p ) {
      p--;
      return true;
    }
    return false;
  }
  inline bool has_ops(int &op) const {
    op = kind & 0x7;
    if ( op ) {
      op--;
      return true;
    }
    return false;
  }
  static inline RH windex(int w) {
    return (w & 7) << 4;
  }
  inline int windex() const {
    return (kind >> 4) & 7;
  }
  RegTabChains tab_chain;
};

struct typed_reg_history: public reg_history {
  NVP_type type = GENERIC;
};

struct cbank_history {
  unsigned long off, cb_off;
  // kind - low 4 bits is size in bytes
  unsigned short cb_num, kind;
};

// snapshot of registers acessed/patched for current single instruction
struct track_snap {
  // key: GPR has prefix 0, UGPR 0x8000
  // value: 0x80 - write
  //        0x40 - reuse
  //        0x20 - read even if we already have write
  //        0x0x - ISRC_XX
  std::unordered_map<unsigned short, unsigned char> gpr;
  // 2 set of predictes: 1 - read, 2 - write
  static constexpr int pr_size = 7;
  char pr[pr_size] = { 0, 0, 0, 0, 0, 0, 0 },
      upr[pr_size] = { 0, 0, 0, 0, 0, 0, 0 };
  // cc
  std::optional<unsigned char> cc;
  void reset() {
    gpr.clear(); cc.reset();
    memset(pr, 0, pr_size); memset(upr, 0, pr_size);
  }
  bool empty_pr() const {
    return std::all_of(pr, pr + pr_size, [](char c) -> bool { return !c; });
  }
  bool empty_upr() const {
    return std::all_of(upr, upr + pr_size, [](char c) -> bool { return !c; });
  }
  bool empty() const {
    if ( !gpr.empty() ) return false;
    return empty_pr() && empty_upr();
  }
};

// register tracks
// there can be 4 groups of register
// - general purpose registers
// - predicate registers
// and since sm75 also
// - uniform gpr
// - uniform predicates
// keys are register index
struct reg_pad {
  typedef std::unordered_map<int, std::vector<reg_history> > RSet;
  typedef std::unordered_map<int, std::vector<typed_reg_history> > TRSet;
  std::vector<reg_history> cc;
  TRSet gpr, ugpr;
  RSet pred, upred;
  std::vector<cbank_history> cbs;
  track_snap *snap = nullptr;
  reg_reuse m_reuse;
  reg_history::RH pred_mask = 0;
  // if you want some inheritance - make destructor virtual
  ~reg_pad() {
    if ( snap ) delete snap;
  }
  // boring stuff
  reg_history::RH check_reuse(int op) const {
    if ( op < ISRC_A) return 0;
    if ( m_reuse.mask & (1 << (op - ISRC_A)) ) return reg_history::reuse;
    return 0;
  }
  void add_cb(unsigned long off, unsigned long cb_off, unsigned short cb_num, unsigned short k) {
    cbs.push_back( { off, cb_off, cb_num, k });
  }
  RegTabChains* _add(RSet &rs, int idx, unsigned long off, reg_history::RH k) {
    if ( snap ) {
      if ( &rs == &pred ) {
        if ( k & 0x8000 )
         snap->pr[idx] |= 2;
        else
         snap->pr[idx] |= 1;
      } else {
        if ( k & 0x8000 )
         snap->upr[idx] |= 2;
        else
         snap->upr[idx] |= 1;
      }
    }
    k |= pred_mask;
    auto ri = rs.find(idx);
    if ( ri != rs.end() ) {
      if ( !ri->second.empty() ) { // check if prev item is the same
        auto &last = ri->second.back();
        if ( last.off == off && last.kind == k ) return nullptr;
      }
      ri->second.push_back( { off, k } );
      return &ri->second.back().tab_chain;
    } else {
     std::vector<reg_history> tmp;
     tmp.push_back( { off, k } );
     auto et = rs.emplace(idx, std::move(tmp) );
     return &et.first->second.back().tab_chain;
    }
  }
  RegTabChains* rcc(unsigned long off) {
    if ( snap ) snap->cc.emplace(1);
    cc.push_back( { off, pred_mask } );
    return &cc.back().tab_chain;
  }
  RegTabChains* wcc(unsigned long off) {
    if ( snap ) snap->cc.emplace(2);
    reg_history::RH kind = 0x8000 | pred_mask;
    cc.push_back( { off, kind } );
    return &cc.back().tab_chain;
  }
  RegTabChains* _add(TRSet &rs, int idx, unsigned long off, reg_history::RH k, NVP_type t = GENERIC) {
    k |= pred_mask;
    auto ri = rs.find(idx);
    if ( ri != rs.end() ) {
      if ( !ri->second.empty() ) { // check if prev item is the same
        auto &last = ri->second.back();
        if ( last.off == off && last.kind == k ) return nullptr;
      }
      ri->second.push_back( { off, k, {}, t } );
      return &ri->second.back().tab_chain;
    } else {
     std::vector<typed_reg_history> tmp;
     tmp.push_back( { off, k, {}, t } );
     auto et = rs.emplace(idx, std::move(tmp) );
     return &et.first->second.back().tab_chain;
    }
  }
  RegTabChains* rgpr(int r, unsigned long off, reg_history::RH k, int op, NVP_type t = GENERIC) {
     auto reuse = check_reuse(op);
     if ( snap ) {
       std::unordered_map<unsigned short, unsigned char>::iterator si = snap->gpr.find(r);
       if ( si != snap->gpr.end() ) {
         si->second |= 0x20;
         if ( reuse ) si->second |= 0x40;
       } else
       snap->gpr[r] = op | (reuse ? 0x40 : 0) | (k & 0x8000 ? 0x80: 0);
     }
     return _add(gpr, r, off, k | reuse, t);
  }
  RegTabChains* wgpr(int r, unsigned long off, reg_history::RH k, NVP_type t = GENERIC) {
     if ( snap ) snap->gpr[r] = 0x80;
     return _add(gpr, r, off, k | 0x8000, t);
  }
  RegTabChains* rugpr(int r, unsigned long off, reg_history::RH k, int op, NVP_type t = GENERIC) {
     auto reuse = check_reuse(op);
     if ( snap ) {
       std::unordered_map<unsigned short, unsigned char>::iterator si = snap->gpr.find(r | 0x8000);
       if ( si != snap->gpr.end() ) {
         si->second |= 0x20;
         if ( reuse ) si->second |= 0x40;
       } else
       snap->gpr[r | 0x8000] = op | (reuse ? 0x40 : 0) | (k & 0x8000 ? 0x80: 0);
     }
     return _add(ugpr, r, off, k | reuse, t);
  }
  RegTabChains* wugpr(int r, unsigned long off, reg_history::RH k, NVP_type t = GENERIC) {
     if ( snap ) snap->gpr[r | 0x8000] = 0x80;
     return _add(ugpr, r, off, k | 0x8000, t);
  }
  inline RegTabChains* rpred(int r, unsigned long off, reg_history::RH k) {
    return _add(pred, r, off, k);
  }
  inline RegTabChains* wpred(int r, unsigned long off, reg_history::RH k) {
    return _add(pred, r, off, k | 0x8000);
  }
  inline RegTabChains* rupred(int r, unsigned long off, reg_history::RH k) {
    return _add(upred, r, off, k);
  }
  inline RegTabChains* wupred(int r, unsigned long off, reg_history::RH k) {
    return _add(upred, r, off, k | 0x8000);
  }
  bool empty() const {
    return gpr.empty() && pred.empty() && ugpr.empty() && upred.empty() && cbs.empty();
  }
  void clear() {
     pred_mask = 0;
     gpr.clear();
     pred.clear();
     ugpr.clear();
     upred.clear();
     cbs.clear();
     cc.clear();
  }
};

// interface for latency tracking
// type is
// 0 for gpr
// 1 for predicates
// 2 for cc
// | 0x80 for uni
typedef std::function<void(unsigned char type, unsigned char what, unsigned long dst, unsigned long src, const found_tab_cross &)> TLTrackCB;

std::string lt_what(unsigned char type, unsigned char what);
