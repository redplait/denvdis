#pragma once
#include "decuda_base.h"

struct cupti_item {
  uint64_t addr, value, ind = 0;
};

class de_cupti: public decuda_base {
 public:
  de_cupti(ELFIO::elfio *rdr):
     decuda_base(rdr)
   {
   }
   void dump_res() const;
 protected:
  virtual int _read() override;
  int try_ext(uint64_t);
  int try_subscribe();
  int fsm_log(diter &, uint64_t off, uint64_t &res);
  // output data
  uint64_t m_cupti_root = 0;
  uint64_t m_dbg_root = 0;
  std::vector<cupti_item> m_items;
  // from old cuptiSubscribe
  uint64_t curr_func = 0;
  uint64_t curr_data = 0;
  // tlg
  Tlg m_tlg;
};