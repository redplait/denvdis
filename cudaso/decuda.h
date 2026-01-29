#pragma once

#include "decuda_base.h"
#include <functional>

class rtmem_storage;

struct one_intf {
  unsigned char uuid[16];
  uint64_t addr;
  int size = 0;
};

class decuda: public decuda_base {
 public:
   decuda(ELFIO::elfio *rdr):
     decuda_base(rdr)
   {
   }
   void verify(FILE *fp) const {
     _verify(fp, nullptr);
   }
   void dump_res() const;
   void verify_patch(FILE *fp, const struct dbg_patch *, int) const;
   int patch_logger(FILE *fp, const unsigned char *, size_t) const;
 protected:
   virtual int _read() override;
   uint32_t read_size(uint64_t);
   int patch_tracepoints(uint64_t, const unsigned char *, size_t) const;
   int patch_dbg(FILE *fp, uint64_t, const struct dbg_patch *, int) const;
   void _verify(FILE *, std::function<void(uint64_t, rtmem_storage &)> *) const;
   int find_intf_tab();
   int resolve_flag_sztab();
   int resolve_indirects();
   int try_sizetab(uint64_t);
   int try_dbgtab(uint64_t);
   void fill_sztab();
   void fill_dbgtab();
   int resolve_api_gate(ptrdiff_t);
   // verifier methods
   template <typename T>
   bool dump_xxx(FILE *fp, const char *, int64_t delta) const;
   bool dump_str_with_len(FILE *fp, const char *addr, const char *len, int64_t delta) const;
   void dump_bss_publics(FILE *fp, int64_t delta) const;
   void check_addr(FILE *fp, uint64_t, int64_t delta, const char *pfx, rtmem_storage &) const;
   void check_dword(FILE *fp, uint64_t, int64_t delta, const char *pfx, rtmem_storage &) const;
   // output data
   uint64_t m_api_gate = 0;
   uint64_t m_api_data = 0;
   uint64_t m_intf_tab = 0;
   std::vector<one_intf> m_intfs;
   const one_intf *find(const unsigned char *key) const {
     for ( auto &i: m_intfs ) {
       if ( !memcmp(i.uuid, key, 16) ) return &i;
     }
     return nullptr;
   }
   // dbg trace data
   uint64_t m_trace_fn = 0;
   uint64_t m_trace_flag = 0;
   uint64_t m_trace_key = 0;
   // dbg flags
   uint64_t m_flag_sztab_addr = 0,
     m_dbgtab_addr = 0;
   int m_flag_sztab_size = 0;
   std::vector<uint32_t> m_flag_sztab;
   std::vector<uint64_t> m_dbgtab; // hopefully size is the same as of m_flag_sztab
   inline bool has_flag_sztab() const {
     return (m_flag_sztab_addr != 0) && (m_flag_sztab_size != 0);
   }
   // key is name from m_syms so we can use string_view
   // value is pair<ptr, original function>
   struct one_forward {
     uint64_t off;
     uint64_t cb;
     uint64_t flag_addr = 0;
   };
   std::map<std::string_view, one_forward > m_forwards;
};

decuda *get_decuda(const char *);
