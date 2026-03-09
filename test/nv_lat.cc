#include "nv_rend.h"
#include <string_view>
// for sv literals
using namespace std::string_view_literals;

// generated with script/pal.pl -C
#include "lat.inc"

int NV_renderer::check_lat_set(const NV_sorted *ns) const {
  int res = 0;
  std::unordered_set<std::string_view> visited;
  for ( auto &nitem: *ns ) {
    auto fi = s_lats.find(nitem.first);
    if ( fi == s_lats.cend() ) {
      fprintf(m_out, "LMissed: %.*s\n", (int)nitem.first.size(), nitem.first.data());
      res++;
    } else {
      visited.insert(nitem.first);
    }
  }
  // dump unvisited
  for ( auto &li: s_lats ) {
    auto fi = visited.find(li.first);
    if ( fi != visited.cend() ) continue;
    fprintf(m_out, "LUnk: %.*s\n", (int)li.first.size(), li.first.data());
  }
  return res;
}