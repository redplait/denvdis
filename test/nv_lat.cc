#include "nv_rend.h"
#include <string_view>
// for sv literals
using namespace std::string_view_literals;

// generated with script/pal.pl -C
#include "lat.inc"

int NV_renderer::check_lat_set(const NV_sorted *ns) const {
  int res = 0;
  for ( auto &nitem: *ns ) {
    auto fi = s_lats.find(nitem.first);
    if ( fi == s_lats.cend() ) {
      fprintf(m_out, "LMissed: %.*s\n", (int)nitem.first.size(), nitem.first.data());
      res++;
    }
  }
  return res;
}