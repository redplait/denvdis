#pragma once
#include "decuda_base.h"
#ifdef WITH_CEREAL
#include <cereal/types/vector.hpp>
#endif

struct one_intf {
  unsigned char uuid[16];
  uint64_t addr = 0;
  int size = 0;
#ifdef WITH_CEREAL
  template <class Archive>
  void serialize( Archive & ar ) {
    ar( uuid, addr, size );
  }
#endif
};

struct decuda_data {
  uint64_t m_api_gate = 0;
  uint64_t m_api_data = 0;
  uint64_t m_intf_tab = 0;
  std::vector<one_intf> m_intfs;
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
#ifdef WITH_CEREAL
  template <class Archive>
  void serialize( Archive & ar ) {
    // when we will have reflection support in every popular c++ compiler?
    ar( m_api_gate, m_api_data, m_intf_tab, CEREAL_NVP(m_intfs), m_trace_fn, m_trace_flag, m_trace_key,
      m_flag_sztab_addr, m_dbgtab_addr, m_flag_sztab_size, CEREAL_NVP(m_flag_sztab), CEREAL_NVP(m_dbgtab)
    );
  }
#endif
};
