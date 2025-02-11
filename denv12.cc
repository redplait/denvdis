#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <elfio/elfio.hpp>
#include "lz4.h"

typedef  uint32_t _DWORD;
typedef  unsigned char _BYTE;
// sub-dir to store results
const char *subdir = "data12/";

const _DWORD seeds[64] = {
 0x1E0D8064, 0x16EFF6E2, 0x3A29FF98, 0x0AD28CF5E, 0x0CEFC4584, 0x0ACE3AB65,
 0x924997EB, 0x0F7A813C3, 0x0DED07CDD, 0x0EC7278F8, 0x2B9412A5, 0x909A5339,
 0x99AE3F04, 0x0C1BF7532, 0x0FEDB9102, 0x0F0B5D67D, 0x0B6B33C3D, 0x0A276BC00,
 0x8550E19C, 0x2D664A09, 0x79D56AB7, 0x0D11B3EEA, 0x95717BA3, 0x8B87B259,
 0x18819F5C, 0x934BCB27, 0x0BD6B2A11, 0x19F5C8A0, 0x40C21FF2, 0x0DA03DF6C,
 0x82B15521, 0x0E6778963, 0x2473C9A9, 0x37E4D9C6, 0x68F44F17, 0x0AF8607C7,
 0x0DCB8A6FA, 0x8E470A88, 0x831A14E7, 0x7F359E74, 0x0A1E052FD, 0x330B4C41,
 0x0BBF38F44, 0x0C7A5808, 0x9BE53BF1, 0x0C05DD426, 0x60A77EF9, 0x342551E8,
 0x5A4D0FD7, 0x0C41D36D3, 0x62202F8A, 0x8D61AAB0, 0x4801698C, 0x6ED8ED38,
 0x2E05CC42, 0x65B1022, 0x0E46234E, 0x0EE5FE9BA, 0x0A430B4B9, 0x571CBE54,
 0x9D96FBCD, 0x70C56731, 0x566D4315, 0x0CA6FD22C,
};

const _BYTE *salt = (const _BYTE *)seeds;

// 0x4816 0 1 0xe9
struct decr_ctx {
 _DWORD wtf;
 _DWORD seed; // 4
 _DWORD res;  // 8
 _BYTE l;     // 12
};

_DWORD decrypt(decr_ctx *ctx, unsigned char *a2, int a3) {
  unsigned int v3; // ecx
  _DWORD result; // rax
  char v5; // r8
  int64_t v6; // r11
  char v7; // dl
  char v8; // bl

  v3 = ctx->seed;
  result = ctx->res;
  v5 = ctx->l;
  if ( a3 )
  {
    v6 = (int64_t)&a2[a3 - 1 + 1];
    do
    {
      result = result - 1;
      if ( result )
      {
        v3 >>= 8;
      }
      else
      {
        result = 4;
        // very similar to srand, see https://github.com/insidegui/flux/blob/master/libc/stdlib/rand.c
        v3 = 1103515245 * ctx->wtf + 0x3039;
        ctx->wtf = v3;
      }
      v7 = *a2;
      v8 = *((_BYTE *)salt + (_BYTE)(*a2 ^ v5)) ^ v3;
      v5 = *a2;
      *a2++ = v8;
    }
    while ( a2 != (unsigned char *)v6 );
    ctx->seed = v3;
    ctx->res = result;
    ctx->l = v7;
  }
  else
  {
    ctx->seed = v3;
    ctx->res = result;
    ctx->l = v5;
  }
  return result;
}

char decompress_buf[0x1000000];

struct one_md {
 size_t off, size;
 int cont, init;
 const char *name;
};

// for release 12.8, V12.8.55, md5 199cc993077066c9e184bfeef9543093
const one_md mds[] = {
 // 72, 75 - B4A - volta
 { 0x82B680, 0x45CDB, 0, 0xb4a, "sm75_1" },
 { 0x8263C0, 0x52A6, 1, 0xb4a, "sm75_2" },
 { 0x826380, 0x20, 2, 0xb4a, "sm75_3" },
 // 80, 86 - 8416 - ampere
 { 0xAED5A0, 0x4F2CE, 0, 0x8416, "sm80_1" },
 { 0xAE6DC0, 0x67C5, 1, 0x8416, "sm80_2" },
 { 0xAE6D80, 0x20, 2, 0x8416, "sm80_3" },
 { 0x878100, 0x52D36, 0, 0x8416, "sm86_1" },
 { 0x8713A0, 0x6D51, 1, 0x8416, "sm86_2" },
 { 0x871360, 0x20, 2, 0x8416, "sm86_3" },
 // 89 - 1684 - ada
 { 0x8D1D00, 0x575A0, 0, 0x1684, "sm89_1" },
 { 0x8CAE80, 0x6E7E, 1, 0x1684, "sm89_2" },
 { 0x8CAE40, 0x20, 2, 0x1684, "sm89_3" },
 // 90 - 3927 - h100
 { 0xA3A5E0, 0x6D365, 0, 0x3927, "sm90_1" },
 { 0xA31DE0, 0x87EE, 1, 0x3927, "sm90_2" },
 { 0xA31DA0, 0x20, 2, 0x3927, "sm90_3" },
 // 100 - 120 - 9327 - blackwell
 { 0x932440, 0x5EE14, 0, 0x9327, "sm100_1" },
 { 0x929300, 0x9123, 1, 0x9327, "sm100_2" },
 { 0x9292C0, 0x20, 2, 0x9327, "sm100_3" },
 { 0x99A1C0, 0x59F35, 0, 0x9327, "sm101_1" },
 { 0x9912A0, 0x8F1D, 1, 0x9327, "sm101_2" },
 { 0x991260, 0x20, 2, 0x9327, "sm101_3" },
 { 0x779E60, 0x7D2EB, 0, 0x9327, "sm120_1" },
 { 0x770B00, 0x934A, 1, 0x9327, "sm120_2" },
 { 0x770AC0, 0x20, 2, 0x9327, "sm120_3" },
 { 0, 0, 0, 0, NULL }, // last
};

using namespace ELFIO;

bool decrypt_part(section *d, int idx) {
  auto ptr = d->get_data() + mds[idx].off - d->get_address();
  std::string fname = subdir;
  fname += mds[idx].name;
  fname += ".txt";
  // alloc mem
  unsigned char *out = (unsigned char *)malloc(mds[idx].size);
  if ( !out ) {
    fprintf(stderr, "cannot alloc %X bytes for index %d\n", mds[idx].size, idx);
    return false;
  }
  FILE *fp = fopen(fname.c_str(), "w");
  if ( !fp ) {
    fprintf(stderr, "cannot create %s, error %d (%s)\n", fname.c_str(), errno, strerror(errno));
    free(out);
    return false;
  }
  decr_ctx ctx;
  ctx.wtf = mds[idx].init;
  ctx.seed = 0;
  ctx.res = 1;
  // via assign to mdObfuscation_ptr
  _BYTE l = mds[idx].init & 0xff;
  ctx.l = ~l;
  memcpy(out, ptr, mds[idx].size);
  auto dres = decrypt(&ctx, out, mds[idx].size);
 // printf("idx %d dres %X\n", idx, dres);
  if ( (out[0] & 0xf0) == 0xf0 ) {
    // lz4 decompress
    auto res = LZ4_decompress_safe((const char *)out, decompress_buf, mds[idx].size, sizeof(decompress_buf));
  printf("idx %d - len %X lz4 %X\n", idx, mds[idx].size, res);
    if ( res )
      fwrite(decompress_buf, 1, res, fp);
    else {
      fprintf(stderr, "cannot unpack idx %d\n", idx);
      fwrite(out, 1, mds[idx].size, fp);
    }
  } else
    fwrite(out, 1, mds[idx].size, fp);
  fclose(fp);
  free(out);
  return true;
}

int main(int argc, char **argv) {
  const char *def = "./nvdisasm";
  if ( argc > 1 ) def = argv[1];
  elfio elf;
  auto res = elf.load(def);
  if ( !res ) {
    fprintf(stderr, "cannot load %s\n", def);
    return 1;
  }
  // all md located in .data section
  section *data = nullptr;
  Elf_Half n = elf.sections.size();
  for ( Elf_Half i = 0; i < n; i++) {
    section *s = elf.sections[i];
    const char* name = s->get_name().c_str();
    if ( !strcmp(".data", name) ) {
      data = s;
      break;
    }
  }
  if ( !data ) {
    fprintf(stderr, "cannot find section .data in %s\n", def);
    return 2;
  }
  // make subdir
  if ( mkdir(subdir, 0744) ) {
    if ( errno != EEXIST ) {
      fprintf(stderr, "cannot create %s, error %d (%s)\n", subdir, errno, strerror(errno));
      return 3;
    }
  }
  // process machine descriptions in table mds
  for ( int idx = 0; mds[idx].off; ++idx )
    decrypt_part(data, idx);
}