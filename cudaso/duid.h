#pragma once

using Duid = struct _duid { unsigned char res[16];
 bool operator==(const _duid& other) const = default;
};
template <>
struct std::hash<Duid> {
 std::size_t operator()(const _duid& u) const noexcept {
   std::size_t h1 = std::hash<uint64_t>{}(*(uint64_t *)u.res);
   std::size_t l0 = std::hash<uint64_t>{}(*(uint64_t *)(u.res + 8));
#ifdef DEBUG
      printf("%8.8X-%4.4hX-%4.4hX-%2.2X%2.2X-%2.2X%2.2X%2.2X%2.2X%2.2X%2.2X hash %X\n",
       *(uint32_t *)(u.res), *(unsigned short *)(u.res + 4), *(unsigned short *)(u.res + 6),
       u.res[8], u.res[9], u.res[10], u.res[11], u.res[12], u.res[13], u.res[14], u.res[15], h1 ^ l0);
#endif
   return h1 ^ l0;
 }
};
typedef std::unordered_map<Duid, const char *> DuidMap;

Duid constexpr conv2uid(uint32_t a, uint16_t b, uint16_t c, unsigned char d0, unsigned char d1,
  unsigned char d2, unsigned char d3, unsigned char d4, unsigned char d5, unsigned char d6, unsigned char d7) {
  Duid r;
  r.res[0] = (unsigned char)(a & 0xff);
  r.res[1] = (unsigned char)((a >> 8) & 0xff);
  r.res[2] = (unsigned char)((a >> 16) & 0xff);
  r.res[3] = (unsigned char)((a >> 24) & 0xff);
  r.res[4] = (unsigned char)(b & 0xff);
  r.res[5] = (unsigned char)((b >> 8) & 0xff);
  r.res[6] = (unsigned char)(c & 0xff);
  r.res[7] = (unsigned char)((c >> 8) & 0xff);
  r.res[8] = d0;  r.res[9] = d1;  r.res[10] = d2; r.res[11] = d3;
  r.res[12] = d4; r.res[13] = d5; r.res[14] = d6; r.res[15] = d7;
  return r;
}

static const DuidMap s_duids = {
#include "dmap.inc"
};

const char *get_duid(const unsigned char *res) {
  auto di = s_duids.find( *(Duid *)res );
  if ( di == s_duids.end() ) return nullptr;
  return di->second;
}