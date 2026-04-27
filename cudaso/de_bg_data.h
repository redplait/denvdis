#pragma once
#include "decuda_base.h"
#ifdef WITH_CEREAL
#include <cereal/types/vector.hpp>
#endif

const char *find_de_tlg(size_t);
int patch_tlg(uint64_t delta, char, const Tlg &);
constexpr const char *de_api = "GetCUDADebuggerAPI";

struct bg_api {
  uint64_t addr = 0;
  uint64_t sub = 0;
  std::string name;
#ifdef WITH_CEREAL
  template <class Archive>
  void serialize( Archive & ar ) {
    ar( CEREAL_NVP(addr), CEREAL_NVP(sub), CEREAL_NVP(name) );
  }
#endif
};

struct de_bg_data {
  uint64_t m_api = 0;
  uint64_t m_state = 0;
  uint64_t m_bg_log = 0;
  uint64_t m_log_root = 0;
  std::vector<bg_api> m_apis;
  Tlg m_tlg;
  ELFIO::Elf64_Addr m_pivot = 0;
#ifdef WITH_CEREAL
  template <class Archive>
  void save( Archive & ar ) const
  {
    ar( CEREAL_NVP(m_api), CEREAL_NVP(m_state), CEREAL_NVP(m_bg_log), CEREAL_NVP(m_log_root),
     CEREAL_NVP(m_pivot), CEREAL_NVP(m_apis), CEREAL_NVP(m_tlg) );
  }
  template <class Archive>
  void load( Archive & ar ) {
    ar( CEREAL_NVP(m_api), CEREAL_NVP(m_state), CEREAL_NVP(m_bg_log), CEREAL_NVP(m_log_root),
     CEREAL_NVP( m_pivot), CEREAL_NVP(m_apis), CEREAL_NVP(m_tlg) );
    // do post-load processing for tlg
    for ( size_t i = 0; i < m_tlg.size(); ++i ) m_tlg[i].name = find_de_tlg(i);
  }
#endif
};
