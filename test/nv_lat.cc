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

static bool check_zero_kv(const NV_extracted &kv, const char *what) {
  auto ki = kv.find(what);
  if ( ki == kv.cend() ) return false;
  return !ki->second;
}

static bool check_nonzero_kv(const NV_extracted &kv, const char *what) {
  auto ki = kv.find(what);
  if ( ki == kv.cend() ) return false;
  return ki->second;
}

inline static bool has_key(const NV_extracted &kv, const char *what) {
  auto ki = kv.find(what);
  return ki != kv.end();
}

static const std::unordered_map<std::string_view, int> s_spec9 = {
 { "UTCCP"sv, 10 },
 { "UTCHMMA"sv, 12 },
 { "UTCIMMA"sv, 12 },
 { "UTCMXQMMA"sv, 14 },
 { "UTCOMMA"sv, 12 },
 { "UTCQMMA"sv, 12 },
 { "UTCSHIFT"sv, 13 },
};

std::optional<int> NV_renderer::calc_latency(const struct nv_instr *ins, const NV_extracted &kv) const {
  std::string_view iname = ins->name;
  std::string_view iclas = ins->cname;
  std::optional<int> res;
  // pre-classify - rename some opcodes
  switch( ins->name[0] ) {
    case 'A': if ( iname == "AL2P"sv ) {
 /* AL2P has only 2 form:
     - NonZeroRegister:Ra ',' SImm(11)*:Ra_offset - class al2p__RaNonRZ
     - ZeroRegister("RZ"):Ra ',' UImm(10)*:Ra_offset - class al2p__RaRZ
    so I guessed that _INDEXED is first one
  */
        if ( iclas != "al2p__RaRZ"sv ) iname = "AL2P_INDEXED"sv;
      }
     break;
    case 'V': if ( iname == "VOTE_VTG"sv ) iname = "VOTE"sv; // bcs VOTE_VTG is just alt alias
     break;
  }
  // check in s_lats
  auto li = s_lats.find(iname);
  if ( li == s_lats.end() ) return res;
  // check that we have value (can be missed for bad states
  if ( li->second.first ) {
    res.emplace(li->second.first);
    if ( !li->second.second ) // if no state for post-processing - we are done
      return res;
  }
  // post-processing - keep them in sorted order for better navigation
  switch(li->second.second) {
    case LatSpecial::Spec3: // with Rb (and I guess with URb too)
      if ( has_key(kv, "Rb") || has_key(kv, "URb") ) {
 // NANOSLEEP: 19
 // NANOTRAP: 18
 // WARPSYNC: 18
        if ( iname == "NANOSLEEP"sv ) res.emplace(19);
        else res.emplace(18);
      }
     break;
    case LatSpecial::Spec9: {
       auto i9 = s_spec9.find(iname);
       if ( i9 != s_spec9.cend() ) {
         res.emplace(i9->second);
       }
     }
     break;
    case LatSpecial::Spec25:
    case LatSpecial::Spec11: res.emplace(7);
     break;
    case LatSpecial::Spec23: // .FINAL
      if ( iclas == "out__FINAL"sv ) res.emplace(9);
     break;
    case LatSpecial::Spec24: // .FLUSH
      if ( check_nonzero_kv(kv, "flush") ) res.emplace(15);
     break;
    case LatSpecial::Spec26: // .GSB - sm90 only?
      if ( check_zero_kv(kv, "gsb") ) res.emplace(9);
      break;
    case LatSpecial::Spec28: // .IMM
      if ( has_key(kv, "sImm") ) res.emplace(7);
      break;
    case LatSpecial::Spec29: // .RELEASE - I guess stands for dealloc
      if ( iclas.ends_with("dealloc") ) res.emplace(18);
      break;
    case LatSpecial::Spec31: // .SP
      if ( iclas.starts_with("qmma_sp"sv) ) res.emplace(7);
      break;
    case LatSpecial::Spec33: // .SYNC.DEFER_BLOCKING
      if ( check_nonzero_kv(kv, "defer_blocking") ) res.emplace(23);
     break;
    case LatSpecial::Spec34: // .WIDE
      if ( check_nonzero_kv(kv, "wide") ) {
 // IMUL: 9
 // IMUL32I: 12
        if ( iname == "IMUL"sv ) res.emplace(9);
        else if ( iname == "IMUL32I"sv ) res.emplace(12);
      }
     break;
  }
  return res;
}