#include <idc.idc>

static main() {
  auto fp = fopen("yy_state_list", "w");
  auto start = 0x55968F8894A8;
  auto end = 0x55968FA01428;
  auto i, w1, w2;
  for ( i = start; i < end; i = i + 8 ) {
    w1 = dword(i);
    w2 = dword(i+4);
    fprintf(fp, "{ %d, %d },\n", w1, w2);
  }
  fclose(fp);
}