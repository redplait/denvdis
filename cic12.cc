#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <elfio/elfio.hpp>

typedef  uint32_t _DWORD;
typedef  unsigned char _BYTE;
// sub-dir to store results
const char *subdir = "cicc12/";

const _DWORD seeds[64] = {
 0x7638D244, 0x8DDFDA30, 0xA6954CAC, 0xC1E302AF, 0x1D2968DD,
 0x8807F899, 0x569A6F63, 0x7103C6EE, 0xE1DC78CA, 0x64B4BE83,
 0x2B690A0E, 0xC9DB4FFC, 0xA734F4EA, 0x87C59EBF, 0xB10B2CD4,
 0x31554041, 0xF9D8A473, 0x96E211A8, 0x664D1AD3, 0x89E0C2A5,
 0x2DA1BD4A, 0xEFFB79EC, 0xDEC35CAD, 0xE60CB660, 0x7CCBCEBB,
 0xF54E1400, 0x6A51D18B, 0xFED7FA74, 0x82265AF7, 0x7E46359C,
 0x59AE5325, 0x9FB93C21, 0x9B7B6201, 0x5E8E4B10, 0x5FC87D94,
 0xA997CFD0, 0x671B392F, 0x19F25B2A, 0xB32E3308, 0x619DF348,
 0x5847A36C, 0xBA9128EB, 0x15CD801E, 0x8F320F17, 0x425D7ACC,
 0x50433EE9, 0xABE4E892, 0x36ED6B45, 0x1C7237B7, 0x8C84F6C7,
 0x65FF816D, 0x0D13F0D9, 0xC4FD5722, 0xC03D52B5, 0x3A7785D6,
 0x75232093, 0x160449A2, 0x987FB286, 0x1854E5BC, 0x6E7D527,
 0xAA70B03F, 0x1F056E8A, 0xF190B824, 0x93BA012
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
        v3 = 0x41C64E6D * ctx->wtf + 0x3039;
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
 const char *name;
 int split = 0;
};

// md5 fc2c33750c78ec555a746a60f9af6535
const one_md mds[] = {
 { 0x4CA7760, 0x860, "c4CA7760" },
 { 0x4CA7380, 0x3D8, "c4CA7380" },
 { 0x4CD5960, 0xA0F0E, "llvm", 1 },
 { 0x4D776A0, 0x14387, "ptx_out", 1 },
 { 0x4D77420, 0x277, "regs", 1 },
 { 0, 0, NULL }, // last
};

const one_md bcs[] = {
 { 0x3DCF740, 0x106A7C, "c1" },
 { 0x3DC6400, 0x4988, "c2" },
 { 0x3D652C0, 0x205B0, "c3" },
 { 0x3D85880, 0x205E8, "c4" },
 { 0x3DA5E80, 0x20570, "c0" },
 { 0x3DCADA0, 0x4988, "c21" },
 { 0x3D63D20, 0x1588, "c5" },
 { 0x3ED61C0, 0x1066D4, "c6" },
 { 0x3FDCFC0, 0x7464C, "c7" },
 { 0x41071E0, 0x9B7A4, "c8" },
 { 0x41A29A0, 0xC2EC, "c9" },
 { 0x42E0200, 0x15a8, "c10" },
 { 0x42E17C0, 0x15A8, "c11" },
 { 0x43048E0, 0x205E8, "c12" },
 { 0x42E2D80, 0x1588, "c13" },
 { 0x42E4320, 0x205B0, "c14" },
 { 0x4324EE0, 0x20570, "c15" },
 { 0x4345460, 0x106A7C, "c16" },
 { 0x444BEE0, 0x1066D4, "c17" },
 { 0x45525C0, 0x7464C, "c18" },
 { 0x45FFE60, 0xC2EC, "c19" },
 { 0x4CA7FC0, 0xFD34, "c20" },
 { 0x4D8C680, 0xFD34, "c22" },
 { 0, 0, NULL }, // last
};

using namespace ELFIO;

bool copy_bc(section *d, int idx) {
  auto ptr = d->get_data() + bcs[idx].off - d->get_address();
  std::string fname = subdir;
  fname += bcs[idx].name;
  fname += ".bc";
  FILE *fp = fopen(fname.c_str(), "w");
  if ( !fp ) {
    fprintf(stderr, "cannot create %s, error %d (%s)\n", fname.c_str(), errno, strerror(errno));
    return false;
  }
  fwrite(ptr, 1, bcs[idx].size, fp);
  fclose(fp);
  return true;
}

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
  // via assign to mdObfuscation_ptr
  _BYTE l = 0;
  for ( size_t curr = 0; curr < mds[idx].size; curr += buf_size )
  {
    auto csize = std::min(buf_size, int(mds[idx].size - curr));
    memcpy(copy_buf, ptr + curr, csize);
    size_t i = 0, old_i = 0;
    while ( i < csize )
    {
      copy_buf[i] ^= l;
      l += 3;
      if ( mds[idx].split && !copy_buf[i] ) {
        if (old_i < i )
          fwrite(copy_buf + old_i, 1, i - old_i, fp);
        fprintf(fp, "\n");
        old_i = i + 1;
      }
      i++;
    }
    if ( mds[idx].split ) {
     if ( old_i < i ) fwrite(copy_buf + old_i, 1, i - old_i, fp);
    } else
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
  section *data = nullptr;
  section *rodata = nullptr;
  Elf_Half n = elf.sections.size();
  for ( Elf_Half i = 0; i < n; i++) {
    section *s = elf.sections[i];
    const char* name = s->get_name().c_str();
    if ( !strcmp(".data", name) ) {
      data = s;
      if ( data && rodata ) break;
    }
    if ( !strcmp(".rodata", name) ) {
      rodata = s;
      if ( data && rodata ) break;
    }
  }
  if ( !data ) {
    fprintf(stderr, "cannot find section .data in %s\n", def);
    return 2;
  }
  if ( !rodata ) {
    fprintf(stderr, "cannot find section .rodata in %s\n", def);
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
  for ( int idx = 0; bcs[idx].off; ++idx )
    copy_bc(rodata, idx);
}