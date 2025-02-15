#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <elfio/elfio.hpp>

typedef  uint32_t _DWORD;
typedef  unsigned char _BYTE;
// sub-dir to store results
const char *subdir = "macros/";

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
 const char *name;
};

// for release 10.1, V10.1.243, md5 54eb83211a7f313d764dce52765a9c1e
const one_md mds[] = {
 // from utilFuncsFermi
 { 0x987340, 0x9877E8 - mds[0].off, "1" },
 { 0x986E80, 0x987328 - mds[1].off, "2" },
 { 0x9869C0, 0x986E68 - mds[2].off, "3" },
 { 0x986500, 0x9869A8 - mds[3].off, "4" },
 { 0x986320, 0x9864F8 - mds[4].off, "5" },
 { 0x985E00, 0x986320 - mds[5].off, "6" },
 { 0x985AC0, 0x985E00 - mds[6].off, "7" },
 { 0x9857E0, 0x985AB0 - mds[7].off, "8" },
 { 0x9855E0, 0x9857D8 - mds[8].off, "9" },
 { 0x985340, 0x9855CF - mds[9].off, "10" },
 { 0x984700, 0x985338 - mds[10].off, "11" },
 { 0x983580, 0x9846F0 - mds[11].off, "12" },
 { 0x982F00, 0x983578 - mds[12].off, "13" },
 { 0x981600, 0x982EE8 - mds[13].off, "14" },
 { 0x97FD00, 0x9815F0 - mds[14].off, "15" },
 { 0x97EE00, 0x97FCF8 - mds[15].off, "16" },
 { 0x97E480, 0x97EE00 - mds[16].off, "17" },
 { 0x97DBC0, 0x97E468 - mds[17].off, "18" },
 { 0x97CC80, 0x97DBB8 - mds[18].off, "19" },
 { 0x97B2E0, 0x97CC68 - mds[19].off, "20" },
 { 0x9799E0, 0x97B2C8 - mds[20].off, "21" },
 { 0x9780C0, 0x9799E0 - mds[21].off, "22" },
 { 0x9771C0, 0x9780B8 - mds[22].off, "23" },
 { 0x975CA0, 0x9771B8 - mds[23].off, "24" },
 { 0x974660, 0x975C88 - mds[24].off, "25" },
 { 0x9737A0, 0x974648 - mds[25].off, "26" },
 { 0x9732A0, 0x973790 - mds[26].off, "27" },
 { 0x971D40, 0x973290 - mds[27].off, "28" },
 { 0x971960, 0x971D40 - mds[28].off, "29" },
 { 0x9706C0, 0x971958 - mds[29].off, "30" },
 { 0x9700E0, 0x9706B0 - mds[30].off, "31" },
 { 0x96FD60, 0x9700D8 - mds[31].off, "32" },
 { 0x96F160, 0x96FD58 - mds[32].off, "33" },
 { 0x96E840, 0x96F160 - mds[33].off, "34" },
 { 0x96E300, 0x96E830 - mds[34].off, "35" },
 { 0x96D700, 0x96E2F8 - mds[35].off, "36" },
 { 0x96CB00, 0x96D6F8 - mds[36].off, "37" },
 { 0x96C660, 0x96CAF8 - mds[37].off, "38" },
 { 0x96C220, 0x96C648 - mds[38].off, "39" },
 { 0x96B4A0, 0x96C220 - mds[39].off, "40" },
 { 0x96A2E0, 0x96B490 - mds[40].off, "41" },
 { 0x969EA0, 0x96A2E0 - mds[41].off, "42" },
 { 0x969780, 0x969E90 - mds[42].off, "43" },
 { 0x969340, 0x969768 - mds[43].off, "44" },
 { 0x968720, 0x969340 - mds[44].off, "45" },
 { 0x9682E0, 0x968720 - mds[45].off, "46" },
 { 0x967BC0, 0x9682D0 - mds[46].off, "47" },
 { 0x967780, 0x967BA8 - mds[47].off, "48" },
 { 0x9669E0, 0x967780 - mds[48].off, "49" },
 { 0x965820, 0x9669D0 - mds[49].off, "50" },
 { 0x9653E0, 0x965820 - mds[50].off, "51" },
 { 0x964CC0, 0x9653D0 - mds[51].off, "52" },
 { 0x964880, 0x964CA8 - mds[52].off, "53" },
 { 0x963E60, 0x964880 - mds[53].off, "54" },
 { 0x963000, 0x963E50 - mds[54].off, "55" },
 { 0x962BC0, 0x963000 - mds[55].off, "56" },
 { 0x9624A0, 0x962BB0 - mds[56].off, "57" },
 { 0x961F80, 0x9624A0 - mds[57].off, "58" },
 { 0x960AA0, 0x961F78 - mds[58].off, "59" },
 { 0x960660, 0x960A98 - mds[59].off, "60" },
 { 0x95F3C0, 0x960658 - mds[60].off, "61" },
 { 0x95EEA0, 0x95F3B0 - mds[61].off, "62" },
 { 0x95E680, 0x95EE90 - mds[62].off, "63" },
 { 0x95E140, 0x95E668 - mds[63].off, "64" },
 { 0x95D940, 0x95E130 - mds[64].off, "65" },
 { 0x95D4E0, 0x95D938 - mds[65].off, "66" },
 { 0x95CD60, 0x95D4D8 - mds[66].off, "67" },
 { 0x95C8E0, 0x95CD50 - mds[67].off, "68" },
 { 0x95C180, 0x95C8D8 - mds[68].off, "69" },
 { 0x95BD20, 0x95C178 - mds[69].off, "70" },
 { 0x95B5A0, 0x95BD18 - mds[70].off, "71" },
 { 0x95B120, 0x95B590 - mds[71].off, "72" },
 { 0x95A9C0, 0x95B118 - mds[72].off, "73" },
 { 0x95A4A0, 0x95A9B0 - mds[73].off, "74" },
 { 0x959C80, 0x95A488 - mds[74].off, "75" },
 { 0x959740, 0x959C68 - mds[75].off, "76" },
 { 0x958F40, 0x959730 - mds[76].off, "77" },
 { 0x958B60, 0x958F38 - mds[77].off, "78" },
 { 0x957F60, 0x958B50 - mds[78].off, "79" },
 { 0x957B80, 0x957F60 - mds[79].off, "80" },
 { 0x956940, 0x957B70 - mds[80].off, "81" },
 { 0x956620, 0x956938 - mds[81].off, "82" },
 { 0x9545C0, 0x9546D8 - mds[82].off, "83" },
 { 0x956440, 0x956520 - mds[83].off, "84" },
 { 0x956520, 0x956618 - mds[84].off, "85" },
 { 0x955720, 0x955800 - mds[85].off, "86" },
 { 0x956260, 0x956340 - mds[86].off, "87" },
 { 0x956340, 0x956438 - mds[87].off, "88" },
 { 0x956080, 0x956160 - mds[88].off, "89" },
 { 0x956160, 0x956258 - mds[89].off, "90" },
 { 0x955EA0, 0x955F80 - mds[90].off, "91" },
 { 0x955F80, 0x956078 - mds[91].off, "92" },
 { 0x955CC0, 0x955DA0 - mds[92].off, "93" },
 { 0x955DA0, 0x955E98 - mds[93].off, "94" },
 { 0x955AE0, 0x955BC0 - mds[94].off, "95" },
 { 0x955BC0, 0x955CB8 - mds[95].off, "96" },
 { 0x955900, 0x9559E0 - mds[96].off, "97" },
 { 0x9559E0, 0x955AD8 - mds[97].off, "98" },
 { 0x955800, 0x9558F8 - mds[98].off, "99" },
 { 0x8302E0, 0xE65A0, "FmtFermi" },
 { 0x816920, 0x8302E0 - 0x816920, "Fermi" },
 { 0, 0, NULL }, // last
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
  ctx.wtf = 0x5389A4F8;
  ctx.seed = 0;
  ctx.res = 1;
  // via assign to mdObfuscation_ptr
  _BYTE l = ctx.wtf & 0xff;
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
  // all md located in .rodata section
  section *data = nullptr;
  Elf_Half n = elf.sections.size();
  for ( Elf_Half i = 0; i < n; i++) {
    section *s = elf.sections[i];
    const char* name = s->get_name().c_str();
    if ( !strcmp(".rodata", name) ) {
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