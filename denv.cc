#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <elfio/elfio.hpp>

typedef  uint32_t _DWORD;
typedef  unsigned char _BYTE;
// sub-dir to store results
const char *subdir = "data/";

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

const int buf_size = 0x3DC;
unsigned char copy_buf[buf_size];

struct one_md {
 size_t off, size;
 int cont, init;
 const char *name;
};

// for release 10.1, V10.1.243, md5 390a9f711384f09fbf0e77c6e8903efb
const one_md mds[] = {
 { 0x6FFC40, 0x92405, 0, 0x4816, "sm3_1" },
 { 0x792060, 0x11ACF, 1, 0x4816, "sm3_2" },
 { 0x7A3B40, 0x10F41, 2, 0x4816, "sm3_3" },
 { 0x7B4AA0, 0xB4D23, 0, 0x1486, "sm4_1" },
 { 0x8697E0, 0x11E22, 1, 0x1486, "sm4_2" },
 { 0x87B620, 0x137A3, 2, 0x1486, "sm4_3" },
 { 0x88EDE0, 0x20C872, 0, 0x1648, "sm5_1" },
 { 0xA9B660, 0x2F8E, 1, 0x1648,   "sm5_2" },
 { 0xA9E600, 0x8AB4, 2, 0x1648,   "sm5_3" },
 { 0xAA70C0, 0x267DAF, 0, 0x6C9E, "sm52_1" },
 { 0xD0EE80, 0x2F91, 1, 0x6C9E,   "sm52_2" },
 { 0xD11E20, 0xF6A0, 2, 0x6C9E,  "sm52_3" },
 { 0xD214E0, 0x26ADF8, 0, 0xC401, "sm55_1" },
 { 0xF8C2E0, 0x2FC6, 1, 0xC401,   "sm55_2" },
 { 0xF8F2C0, 0xF6A0, 2, 0xC401,   "sm55_3" },
 { 0xF9E980, 0x26F06B, 0, 0xE44, "sm57_1" },
 { 0x120DA00, 0x2FE3, 1, 0xE44, "sm57_2" },
 { 0x1210A00, 0x10AC6, 2, 0xE44, "sm57_3" },
 { 0x12214E0, 0x280C8D, 0, 0xC401, "sm70_1" },
 { 0x14A2180, 0x32E7, 1, 0xC401,   "sm70_2" },
 { 0x14A5480, 0x20, 2, 0xC401,     "sm70_3" },
 { 0x14A54C0, 0x29311D, 0, 0xC401, "sm72_1" },
 { 0x17385E0, 0x35E6, 1, 0xC401,   "sm72_2" },
 { 0x173BBE0, 0x20, 2,  0xC401,    "sm72_3" },
 { 0x173BC20, 0x47E5AC, 0, 0xC401, "sm75_1" },
 { 0x1BBA1E0, 0x539A, 1, 0xC401,   "sm75_2" },
 { 0x1BBF580, 0x20, 2, 0xC401,     "sm75_2" },
 { 0, 0, 0, 0, NULL }, // last
};

using namespace ELFIO;

bool decrypt_part(section *d, int idx) {
  auto ptr = d->get_data() + mds[idx].off - d->get_address();
  std::string fname = subdir;
  fname += mds[idx].name;
  fname += ".txt";
  FILE *fp = fopen(fname.c_str(), "w");
  if ( !fp ) {
    fprintf(stderr, "cannot create %s, error %d (%s)\n", fname.c_str(), errno, strerror(errno));
    return false;
  }
  decr_ctx ctx;
  ctx.wtf = mds[idx].init;
  ctx.seed = 0;
  ctx.res = 1;
  // via assign to mdObfuscation_ptr
  _BYTE l = mds[idx].init & 0xff;
  ctx.l = ~l;
  for ( size_t curr = 0; curr < mds[idx].size; curr += buf_size )
  {
    auto csize = std::min(buf_size, int(mds[idx].size - curr));
    memcpy(copy_buf, ptr + curr, csize);
    decrypt(&ctx, copy_buf, csize);
    fwrite(copy_buf, 1, csize, fp);
  }
  fclose(fp);
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
  section *data;
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