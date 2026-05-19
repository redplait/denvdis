#pragma once
#include "decuda_base.h"

// dirty hack to extract encrypted tables from ptxas
class de_ptx: public decuda_base {
 public:
  de_ptx(ELFIO::elfio *rdr):
     decuda_base(rdr)
   {
   }
   void dump_res() const {}
   struct lat_res {
    int what; // 0 - num, 1 - '-', 2 - decrypted string
    int num;
    std::string dec;
   };
   typedef std::map<uint64_t, lat_res> res_map;
   typedef std::map<std::string_view, int> cicc_names;
   // for extracting PTX ops
   struct ptx_op {
     int idx; // r8d
     const char *cx, *dx, *si;
     unsigned char st[16];
     inline void re_st() {
       memset(st, 0, sizeof(st));
     }
     ptx_op() {
       reset();
     }
     ptx_op(const ptx_op &outer): idx(outer.idx), cx(outer.cx), dx(outer.dx), si(outer.si) {
       memcpy(st, outer.st, sizeof(st));
     }
     ptx_op &operator=(const ptx_op&) = delete;
     void reset() {
       idx = 0;
       cx = dx = si = nullptr;
       re_st();
     }
     bool has_st() const {
       return std::all_of(st, st + sizeof(st), [](unsigned char c) { return c != 0; });
     };
   };
   int hack_ptx_ops(uint64_t start, uint64_t end, uint64_t reg_call);
 protected:
   virtual int _read();
   void dump_ptx_ops(std::list<ptx_op> &) const;
   void hack_ctor(uint64_t, const char *fname);
   void hack_cicc_intr(uint64_t, const char *fname);
   void hack_sp(uint64_t, const char *fname);
   int hack_sp(diter &, res_map &);
   int hack(diter &, res_map &);
   template <typename T>
   int hack_cicc(diter &, T &);
   int check(lat_res &, uint64_t off);
   int dump_deres(const char *fname, const res_map &);
   int dump_cicc(const char *fname, const cicc_names &);
};