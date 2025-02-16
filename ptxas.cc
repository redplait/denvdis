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
 { 0x955540, 0x955620 - mds[99].off, "100" },
 { 0x955620, 0x955718 - mds[100].off, "101" },
 { 0x955360, 0x955440 - mds[101].off, "102" },
 { 0x955440, 0x955538 - mds[102].off, "103" },
 { 0x955180, 0x955260 - mds[103].off, "104" },
 { 0x955260, 0x955358 - mds[104].off, "105" },
 { 0x954FA0, 0x955080 - mds[105].off, "106" },
 { 0x955080, 0x955178 - mds[106].off, "107" },
 { 0x954DC0, 0x954EA0 - mds[107].off, "108" },
 { 0x954EA0, 0x954F98 - mds[108].off, "109" },
 { 0x954BE0, 0x954CC0 - mds[109].off, "110" },
 { 0x954CC0, 0x954DB8 - mds[110].off, "111" },
 { 0x954A00, 0x954AE0 - mds[111].off, "112" },
 { 0x954AE0, 0x954BD8 - mds[112].off, "113" },
 { 0x954820, 0x954900 - mds[113].off, "114" },
 { 0x954900, 0x9549F8 - mds[114].off, "115" },
 { 0x9546E0, 0x954810 - mds[115].off, "116" },
 { 0x9514E0, 0x951640 - mds[116].off, "117" },
 { 0x9542E0, 0x954430 - mds[117].off, "118" },
 { 0x954440, 0x9545A8 - mds[118].off, "119" },
 { 0x952EC0, 0x953010 - mds[119].off, "120" },
 { 0x954000, 0x954150 - mds[120].off, "121" },
 { 0x954160, 0x9542D0 - mds[121].off, "122" },
 { 0x953D20, 0x953E70 - mds[122].off, "123" },
 { 0x953E80, 0x953FF0 - mds[123].off, "124" },
 { 0x953A40, 0x953B90 - mds[124].off, "125" },
 { 0x953BA0, 0x953D10 - mds[125].off, "126" },
 { 0x953760, 0x9538B0 - mds[126].off, "127" },
 { 0x9538C0, 0x953A30 - mds[127].off, "128" },
 { 0x953480, 0x9535D0 - mds[128].off, "129" },
 { 0x9535E0, 0x953750 - mds[129].off, "130" },
 { 0x9531A0, 0x9532F0 - mds[130].off, "131" },
 { 0x953300, 0x953470 - mds[131].off, "132" },
 { 0x953020, 0x953188 - mds[132].off, "133" },
 { 0x952BE0, 0x952D30 - mds[133].off, "134" },
 { 0x952D40, 0x952EA8 - mds[134].off, "135" },
 { 0x952900, 0x952A50 - mds[135].off, "136" },
 { 0x952A60, 0x952BC8 - mds[136].off, "137" },
 { 0x952620, 0x952770 - mds[137].off, "138" },
 { 0x952780, 0x9528E8 - mds[138].off, "139" },
 { 0x952340, 0x952490 - mds[139].off, "140" },
 { 0x9524A0, 0x952608 - mds[140].off, "141" },
 { 0x952060, 0x9521B0 - mds[141].off, "142" },
 { 0x9521C0, 0x952328 - mds[142].off, "143" },
 { 0x951D80, 0x951ED0 - mds[143].off, "144" },
 { 0x951EE0, 0x952048 - mds[144].off, "145" },
 { 0x951AA0, 0x951BF0 - mds[145].off, "146" },
 { 0x951C00, 0x951D68 - mds[146].off, "147" },
 { 0x9517C0, 0x951910 - mds[147].off, "148" },
 { 0x951920, 0x951A88 - mds[148].off, "149" },
 { 0x951640, 0x9517B8 - mds[149].off, "150" },
 { 0x94E400, 0x94E560 - mds[150].off, "151" },
 { 0x951200, 0x951350 - mds[151].off, "152" },
 { 0x951360, 0x9514C8 - mds[152].off, "153" },
 { 0x94FDE0, 0x94FF30 - mds[153].off, "154" },
 { 0x950F20, 0x951070 - mds[154].off, "155" },
 { 0x951080, 0x9511E8 - mds[155].off, "156" },
 { 0x950C40, 0x950D90 - mds[156].off, "157" },
 { 0x950DA0, 0x950F08 - mds[157].off, "158" },
 { 0x950960, 0x950AB0 - mds[158].off, "159" },
 { 0x950AC0, 0x950C28 - mds[159].off, "160" },
 { 0x950680, 0x9507D0 - mds[160].off, "161" },
 { 0x9507E0, 0x950948 - mds[161].off, "162" },
 { 0x9503A0, 0x9504F0 - mds[162].off, "163" },
 { 0x950500, 0x950668 - mds[163].off, "164" },
 { 0x9500C0, 0x950210 - mds[164].off, "165" },
 { 0x950220, 0x950388 - mds[165].off, "166" },
 { 0x94FF40, 0x9500A8 - mds[166].off, "167" },
 { 0x94FB00, 0x94FC50 - mds[167].off, "168" },
 { 0x94FC60, 0x94FDC8 - mds[168].off, "169" },
 { 0x94F820, 0x94F970 - mds[169].off, "170" },
 { 0x94F980, 0x94FAE8 - mds[170].off, "171" },
 { 0x94F540, 0x94F690 - mds[171].off, "172" },
 { 0x94F6A0, 0x94F808 - mds[172].off, "173" },
 { 0x94F260, 0x94F3B0 - mds[173].off, "174" },
 { 0x94F3C0, 0x94F528 - mds[174].off, "175" },
 { 0x94EF80, 0x94F0D0 - mds[175].off, "176" },
 { 0x94F0E0, 0x94F248 - mds[176].off, "177" },
 { 0x94ECA0, 0x94EDF0 - mds[177].off, "178" },
 { 0x94EE00, 0x94EF68 - mds[178].off, "179" },
 { 0x94E9C0, 0x94EB10 - mds[179].off, "180" },
 { 0x94EB20, 0x94EC88 - mds[180].off, "181" },
 { 0x94E6E0, 0x94E830 - mds[181].off, "182" },
 { 0x94E840, 0x94E9A8 - mds[182].off, "183" },
 { 0x94E560, 0x94E6D8 - mds[183].off, "184" },
 { 0x94BB20, 0x94BC78 - mds[184].off, "185" },
 { 0x94E1A0, 0x94E2B8 - mds[185].off, "186" },
 { 0x94E2C0, 0x94E3F0 - mds[186].off, "187" },
 { 0x94D100, 0x94D218 - mds[187].off, "188" },
 { 0x94DF40, 0x94E058 - mds[188].off, "189" },
 { 0x94E060, 0x94E190 - mds[189].off, "190" },
 { 0x94DCE0, 0x94DDF8 - mds[190].off, "191" },
 { 0x94DE00, 0x94DF30 - mds[191].off, "192" },
 { 0x94DA80, 0x94DB98 - mds[192].off, "193" },
 { 0x94DBA0, 0x94DCD0 - mds[193].off, "194" },
 { 0x94D820, 0x94D938 - mds[194].off, "195" },
 { 0x94D940, 0x94DA70 - mds[195].off, "196" },
 { 0x94D5C0, 0x94D6D8 - mds[196].off, "197" },
 { 0x94D6E0, 0x94D810 - mds[197].off, "198" },
 { 0x94D360, 0x94D478 - mds[198].off, "199" },
 { 0x94D480, 0x94D5B0 - mds[199].off, "200" },
 { 0x94D220, 0x94D350 - mds[200].off, "201" },
{ 0x94CEA0, 0x94CFB8 - mds[201].off, "202" },
{ 0x94CFC0, 0x94D0F0 - mds[202].off, "203" },
{ 0x94CC40, 0x94CD58 - mds[203].off, "204" },
{ 0x94CD60, 0x94CE90 - mds[204].off, "205" },
{ 0x94C9E0, 0x94CAF8 - mds[205].off, "206" },
{ 0x94CB00, 0x94CC30 - mds[206].off, "207" },
{ 0x94C780, 0x94C898 - mds[207].off, "208" },
{ 0x94C8A0, 0x94C9D0 - mds[208].off, "209" },
{ 0x94C520, 0x94C638 - mds[209].off, "210" },
{ 0x94C640, 0x94C770 - mds[210].off, "211" },
{ 0x94C2C0, 0x94C3D8 - mds[211].off, "212" },
{ 0x94C3E0, 0x94C510 - mds[212].off, "213" },
{ 0x94C060, 0x94C178 - mds[213].off, "214" },
{ 0x94C180, 0x94C2B0 - mds[214].off, "215" },
{ 0x94BE00, 0x94BF18 - mds[215].off, "216" },
{ 0x94BF20, 0x94C050 - mds[216].off, "217" },
{ 0x94BC80, 0x94BDF0 - mds[217].off, "218" },
{ 0x949AC0, 0x949BD0 - mds[218].off, "219" },
{ 0x94B940, 0x94BA18 - mds[219].off, "220" },
{ 0x94BA20, 0x94BB10 - mds[220].off, "221" },
{ 0x94AC20, 0x94ACF8 - mds[221].off, "222" },
{ 0x94B760, 0x94B837 - mds[222].off, "223" },
{ 0x94B840, 0x94B930 - mds[223].off, "224" },
{ 0x94B580, 0x94B658 - mds[224].off, "225" },
{ 0x94B660, 0x94B750 - mds[225].off, "226" },
{ 0x94B3A0, 0x94B478 - mds[226].off, "227" },
{ 0x94B480, 0x94B570 - mds[227].off, "228" },
{ 0x94B1C0, 0x94B298 - mds[228].off, "229" },
{ 0x94B2A0, 0x94B390 - mds[229].off, "230" },
{ 0x94AFE0, 0x94B0B8 - mds[230].off, "231" },
{ 0x94B0C0, 0x94B1B0 - mds[231].off, "232" },
{ 0x94AE00, 0x94AED8 - mds[232].off, "233" },
{ 0x94AEE0, 0x94AFD0 - mds[233].off, "234" },
{ 0x94AD00, 0x94ADF0 - mds[234].off, "235" },
{ 0x94AA40, 0x94AB18 - mds[235].off, "236" },
{ 0x94AB20, 0x94AC10 - mds[236].off, "237" },
{ 0x94A860, 0x94A938 - mds[237].off, "238" },
{ 0x94A940, 0x94AA30 - mds[238].off, "239" },
{ 0x94A680, 0x94A758 - mds[239].off, "240" },
{ 0x94A760, 0x94A850 - mds[240].off, "241" },
{ 0x94A4A0, 0x94A578 - mds[241].off, "242" },
{ 0x94A580, 0x94A670 - mds[242].off, "243" },
{ 0x94A2C0, 0x94A398 - mds[243].off, "244" },
{ 0x94A3A0, 0x94A490 - mds[244].off, "245" },
{ 0x94A0E0, 0x94A1B8 - mds[245].off, "246" },
{ 0x94A1C0, 0x94A2B0 - mds[246].off, "247" },
{ 0x949F00, 0x949FD8 - mds[247].off, "248" },
{ 0x949FE0, 0x94A0D0 - mds[248].off, "249" },
{ 0x949D20, 0x949DF8 - mds[249].off, "250" },
{ 0x949E00, 0x949EF0 - mds[250].off, "251" },
{ 0x949BE0, 0x949D08 - mds[251].off, "252" },
{ 0x9499A0, 0x949AB0 - mds[252].off, "253" },
{ 0x949800, 0x949988 - mds[253].off, "254" },
{ 0x9496E0, 0x9497F0 - mds[254].off, "255" },
{ 0x949540, 0x9496C8 - mds[255].off, "256" },
{ 0x949420, 0x949530 - mds[256].off, "257" },
{ 0x949300, 0x949410 - mds[257].off, "258" },
{ 0x9491E0, 0x949300 - mds[258].off, "259" },
{ 0x949060, 0x9491E0 - mds[259].off, "260" },
{ 0x948F40, 0x949060 - mds[260].off, "261" },
{ 0x948DC0, 0x948F40 - mds[261].off, "262" },
{ 0x948CA0, 0x948DC0 - mds[262].off, "263" },
{ 0x948B20, 0x948CA0 - mds[263].off, "264" },
{ 0x948A00, 0x948B20 - mds[264].off, "265" },
{ 0x948880, 0x948A00 - mds[265].off, "266" },
{ 0x948720, 0x948880 - mds[266].off, "267" },
{ 0x9485C0, 0x948720 - mds[267].off, "268" },
{ 0x948460, 0x9485B8 - mds[268].off, "269" },
{ 0x948300, 0x948460 - mds[269].off, "270" },
{ 0x948200, 0x9482F0 - mds[270].off, "271" },
{ 0x947F60, 0x9481F0 - mds[271].off, "272" },
{ 0x947CA0, 0x947F58 - mds[272].off, "273" },
{ 0x9479E0, 0x947C98 - mds[273].off, "274" },
{ 0x947800, 0x9479D0 - mds[274].off, "275" },
{ 0x947620, 0x947800 - mds[275].off, "276" },
{ 0x947440, 0x947620 - mds[276].off, "277" },
{ 0x947260, 0x947430 - mds[277].off, "278" },
{ 0x947080, 0x947260 - mds[278].off, "279" },
{ 0x946EA0, 0x947080 - mds[279].off, "280" },
{ 0x946C00, 0x946E90 - mds[280].off, "281" },
{ 0x946940, 0x946BF8 - mds[281].off, "282" },
{ 0x946680, 0x946938 - mds[282].off, "283" },
{ 0x9462A0, 0x946668 - mds[283].off, "284" },
{ 0x945E80, 0x946288 - mds[284].off, "285" },
{ 0x945A60, 0x945E68 - mds[285].off, "286" },
{ 0x945700, 0x945A50 - mds[286].off, "287" },
{ 0x945360, 0x9456F0 - mds[287].off, "288" },
{ 0x944FC0, 0x945350 - mds[288].off, "289" },
{ 0x944E00, 0x944FA8 - mds[289].off, "290" },
{ 0x944C40, 0x944DF8 - mds[290].off, "291" },
{ 0x944A80, 0x944C38 - mds[291].off, "292" },
{ 0x9447C0, 0x944A78 - mds[292].off, "293" },
{ 0x9444E0, 0x9447C0 - mds[293].off, "294" },
{ 0x944200, 0x9444E0 - mds[294].off, "295" },
{ 0x943E20, 0x9441F0 - mds[295].off, "296" },
{ 0x943A20, 0x943E08 - mds[296].off, "297" },
{ 0x943300, 0x943A10 - mds[297].off, "298" },
{ 0x942BC0, 0x9432F0 - mds[298].off, "299" },
{ 0x942720, 0x942BB0 - mds[299].off, "300" },
{ 0x942280, 0x942720 - mds[300].off, "301" },
{ 0x941DA0, 0x942270 - mds[301].off, "302" },
{ 0x9418A0, 0x941D88 - mds[302].off, "303" },
{ 0x9414C0, 0x941890 - mds[303].off, "304" },
{ 0x9410C0, 0x9414A8 - mds[304].off, "305" },
{ 0x9409A0, 0x9410B0 - mds[305].off, "306" },
{ 0x940260, 0x940990 - mds[306].off, "307" },
{ 0x93FDC0, 0x940250 - mds[307].off, "308" },
{ 0x93F920, 0x93FDC0 - mds[308].off, "309" },
{ 0x93F440, 0x93F910 - mds[309].off, "310" },
{ 0x93EF40, 0x93F428 - mds[310].off, "311" },
{ 0x93EB60, 0x93EF30 - mds[311].off, "312" },
{ 0x93E760, 0x93EB48 - mds[312].off, "313" },
{ 0x93E040, 0x93E750 - mds[313].off, "314" },
{ 0x93D900, 0x93E030 - mds[314].off, "315" },
{ 0x93D460, 0x93D8F0 - mds[315].off, "316" },
{ 0x93CFC0, 0x93D460 - mds[316].off, "317" },
{ 0x93CAE0, 0x93CFB0 - mds[317].off, "318" },
{ 0x93C5E0, 0x93CAC8 - mds[318].off, "319" },
{ 0x93C200, 0x93C5D0 - mds[319].off, "320" },
{ 0x93BE00, 0x93C1E8 - mds[320].off, "321" },
{ 0x93B6E0, 0x93BDF0 - mds[321].off, "322" },
{ 0x93AFA0, 0x93B6D0 - mds[322].off, "323" },
{ 0x93AB00, 0x93AF90 - mds[323].off, "324" },
{ 0x93A660, 0x93AB00 - mds[324].off, "325" },
{ 0x93A180, 0x93A650 - mds[325].off, "326" },
{ 0x939C80, 0x93A168 - mds[326].off, "327" },
{ 0x9398E0, 0x939C70 - mds[327].off, "328" },
{ 0x939500, 0x9398D0 - mds[328].off, "329" },
{ 0x939120, 0x9394F0 - mds[329].off, "330" },
{ 0x938DE0, 0x939120 - mds[330].off, "331" },
{ 0x938A60, 0x938DE0 - mds[331].off, "332" },
{ 0x9386E0, 0x938A60 - mds[332].off, "333" },
{ 0x938540, 0x9386C8 - mds[333].off, "334" },
{ 0x9383A0, 0x938538 - mds[334].off, "335" },
{ 0x938200, 0x938398 - mds[335].off, "336" },
{ 0x937F40, 0x9381F8 - mds[336].off, "337" },
{ 0x937C60, 0x937F38 - mds[337].off, "338" },
{ 0x937980, 0x937C58 - mds[338].off, "339" },
{ 0x9376E0, 0x937970 - mds[339].off, "340" },
{ 0x937420, 0x9376D0 - mds[340].off, "341" },
{ 0x937160, 0x937410 - mds[341].off, "342" },
{ 0x936F80, 0x937148 - mds[342].off, "343" },
{ 0x936DA0, 0x936F80 - mds[343].off, "344" },
{ 0x936BC0, 0x936DA0 - mds[344].off, "345" },
{ 0x9369E0, 0x936BA8 - mds[345].off, "346" },
{ 0x936800, 0x9369E0 - mds[346].off, "347" },
{ 0x936620, 0x936800 - mds[347].off, "348" },
{ 0x936380, 0x936610 - mds[348].off, "349" },
{ 0x9360C0, 0x936370 - mds[349].off, "350" },
{ 0x935E00, 0x9360B0 - mds[350].off, "351" },
{ 0x935A20, 0x935DE8 - mds[351].off, "352" },
{ 0x935600, 0x935A08 - mds[352].off, "353" },
{ 0x9351E0, 0x9355E8 - mds[353].off, "354" },
{ 0x934E80, 0x9351D0 - mds[354].off, "355" },
{ 0x934AE0, 0x934E70 - mds[355].off, "356" },
{ 0x934740, 0x934AD0 - mds[356].off, "357" },
{ 0x934580, 0x934728 - mds[357].off, "358" },
{ 0x9343C0, 0x934578 - mds[358].off, "359" },
{ 0x934200, 0x9343B8 - mds[359].off, "360" },
{ 0x933F40, 0x9341F8 - mds[360].off, "361" },
{ 0x933C60, 0x933F38 - mds[361].off, "362" },
{ 0x933980, 0x933C57 - mds[362].off, "363" },
{ 0x9335A0, 0x933970 - mds[363].off, "364" },
{ 0x9331A0, 0x933588 - mds[364].off, "365" },
{ 0x932A80, 0x932B5C - mds[365].off, "366" },
{ 0x932340, 0x932A70 - mds[366].off, "367" },
{ 0x931EA0, 0x932330 - mds[367].off, "368" },
{ 0x931A00, 0x931EA0 - mds[368].off, "369" },
{ 0x931520, 0x9319F0 - mds[369].off, "370" },
{ 0x931020, 0x931508 - mds[370].off, "371" },
{ 0x930C40, 0x931010 - mds[371].off, "372" },
{ 0x930840, 0x930C28 - mds[372].off, "373" },
{ 0x930120, 0x930830 - mds[373].off, "374" },
{ 0x92F9E0, 0x930110 - mds[374].off, "375" },
{ 0x92F540, 0x92F9D0 - mds[375].off, "376" },
{ 0x92F0A0, 0x92F540 - mds[376].off, "377" },
{ 0x92EBC0, 0x92F090 - mds[377].off, "378" },
{ 0x92E6C0, 0x92EBA8 - mds[378].off, "379" },
{ 0x92E2E0, 0x92E3BC - mds[379].off, "380" },
{ 0x92DEE0, 0x92E2C8 - mds[380].off, "381" },
{ 0x92D7C0, 0x92DED0 - mds[381].off, "382" },
{ 0x92D080, 0x92D7B0 - mds[382].off, "383" },
{ 0x92CBE0, 0x92D070 - mds[383].off, "384" },
{ 0x92C740, 0x92CBE0 - mds[384].off, "385" },
{ 0x92C260, 0x92C730 - mds[385].off, "386" },
{ 0x92BD60, 0x92C248 - mds[386].off, "387" },
{ 0x92B980, 0x92BD50 - mds[387].off, "388" },
{ 0x92B580, 0x92B968 - mds[388].off, "389" },
{ 0x92AE60, 0x92B570 - mds[389].off, "390" },
{ 0x92A720, 0x92AE50 - mds[390].off, "391" },
{ 0x92A280, 0x92A710 - mds[391].off, "392" },
{ 0x929DE0, 0x92A280 - mds[392].off, "393" },
{ 0x929900, 0x929DD0 - mds[393].off, "394" },
{ 0x929400, 0x9298E8 - mds[394].off, "395" },
{ 0x929060, 0x9293F0 - mds[395].off, "396" },
{ 0x928C80, 0x929050 - mds[396].off, "397" },
{ 0x9288A0, 0x928C70 - mds[397].off, "398" },
{ 0x928560, 0x9288A0 - mds[398].off, "399" },
{ 0x9281E0, 0x928558 - mds[399].off, "400" },
{ 0x927E60, 0x9281D8 - mds[400].off, "401" },
{ 0x927CC0, 0x927E48 - mds[401].off, "402" },
{ 0x927B20, 0x927CB8 - mds[402].off, "403" },
{ 0x927980, 0x927B18 - mds[403].off, "404" },
{ 0x9276C0, 0x927978 - mds[404].off, "405" },
{ 0x9273E0, 0x9275F4 - mds[405].off, "406" },
{ 0x927100, 0x9273D8 - mds[406].off, "407" },
{ 0x926E60, 0x9270F0 - mds[407].off, "408" },
{ 0x926BA0, 0x926E50 - mds[408].off, "409" },
{ 0x9268E0, 0x926B90 - mds[409].off, "410" },
{ 0x926700, 0x9268C8 - mds[410].off, "411" },
{ 0x926520, 0x926700 - mds[411].off, "412" },
{ 0x926340, 0x926520 - mds[412].off, "413" },
{ 0x926160, 0x926328 - mds[413].off, "414" },
{ 0x925F80, 0x926160 - mds[414].off, "415" },
{ 0x925DA0, 0x925F80 - mds[415].off, "416" },
{ 0x925B00, 0x925D90 - mds[416].off, "417" },
{ 0x925840, 0x925AF0 - mds[417].off, "418" },
{ 0x925580, 0x925830 - mds[418].off, "419" },
{ 0x9251A0, 0x925568 - mds[419].off, "420" },
{ 0x924D80, 0x925188 - mds[420].off, "421" },
{ 0x924960, 0x924D68 - mds[421].off, "422" },
{ 0x924600, 0x924950 - mds[422].off, "423" },
{ 0x924260, 0x9245F0 - mds[423].off, "424" },
{ 0x923EC0, 0x924250 - mds[424].off, "425" },
{ 0x923D00, 0x923EA8 - mds[425].off, "426" },
{ 0x923B40, 0x923CF8 - mds[426].off, "427" },
{ 0x923980, 0x923B38 - mds[427].off, "428" },
{ 0x9236C0, 0x923978 - mds[428].off, "429" },
{ 0x9233E0, 0x9236B8 - mds[429].off, "430" },
{ 0x923100, 0x9233D8 - mds[430].off, "431" },
{ 0x922D20, 0x9230F0 - mds[431].off, "432" },
{ 0x922920, 0x922D08 - mds[432].off, "433" },
{ 0x922200, 0x922910 - mds[433].off, "434" },
{ 0x921AC0, 0x9221F0 - mds[434].off, "435" },
{ 0x921620, 0x921AB0 - mds[435].off, "436" },
{ 0x921180, 0x921620 - mds[436].off, "437" },
{ 0x920CA0, 0x921170 - mds[437].off, "438" },
{ 0x9207A0, 0x920C88 - mds[438].off, "439" },
{ 0x9203C0, 0x920790 - mds[439].off, "440" },
{ 0x91FFC0, 0x9203A8 - mds[440].off, "441" },
{ 0x91F8A0, 0x91FFB0 - mds[441].off, "442" },
{ 0x91F160, 0x91F890 - mds[442].off, "443" },
{ 0x91ECC0, 0x91F150 - mds[443].off, "444" },
{ 0x91E820, 0x91EAA1 - mds[444].off, "445" },
{ 0x91E340, 0x91E810 - mds[445].off, "446" },
{ 0x91DE40, 0x91E328 - mds[446].off, "447" },
{ 0x91DA60, 0x91DE30 - mds[447].off, "448" },
{ 0x91D660, 0x91DA48 - mds[448].off, "449" },
{ 0x91CF40, 0x91D650 - mds[449].off, "450" },
{ 0x91C800, 0x91CF30 - mds[450].off, "451" },
{ 0x91C360, 0x91C7F0 - mds[451].off, "452" },
{ 0x91BEC0, 0x91C360 - mds[452].off, "453" },
{ 0x91B9E0, 0x91BEB0 - mds[453].off, "454" },
{ 0x91B4E0, 0x91B9C8 - mds[454].off, "455" },
{ 0x91B100, 0x91B4D0 - mds[455].off, "456" },
{ 0x91AD00, 0x91B0E8 - mds[456].off, "457" },
{ 0x91A5E0, 0x91ACF0 - mds[457].off, "458" },
{ 0x919EA0, 0x91A5D0 - mds[458].off, "459" },
{ 0x919A00, 0x919E90 - mds[459].off, "460" },
{ 0x919560, 0x919A00 - mds[460].off, "461" },
{ 0x919080, 0x919550 - mds[461].off, "462" },
{ 0x918B80, 0x919068 - mds[462].off, "463" },
{ 0x9187E0, 0x918B70 - mds[463].off, "464" },
{ 0x918400, 0x9187D0 - mds[464].off, "465" },
{ 0x918020, 0x9183F0 - mds[465].off, "466" },
{ 0x917CE0, 0x918020 - mds[466].off, "467" },
{ 0x917960, 0x917CD8 - mds[467].off, "468" },
{ 0x9175E0, 0x917958 - mds[468].off, "469" },
{ 0x917440, 0x9175C8 - mds[469].off, "470" },
{ 0x9172A0, 0x917438 - mds[470].off, "471" },
{ 0x917100, 0x917298 - mds[471].off, "472" },
{ 0x916E40, 0x9170F8 - mds[472].off, "473" },
{ 0x916B60, 0x916E38 - mds[473].off, "474" },
{ 0x916880, 0x916B58 - mds[474].off, "475" },
 // ptxFmtFermi
 { 0x8302E0, 0xE65A0, "FmtFermi" },
 // ptxInstructionMacrosFermi
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
    // chop trailing zeros
    while( csize && !copy_buf[csize - 1]) csize--;
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