#include <idc.idc>

static decr_str(addr)
{
  auto curr = addr;
  auto si = 0;
  auto di = 0xCED6C4C1;
  auto c = Byte(addr);
  while( c ) {
    c = 0xff & (c ^ (di >> (si & 0x18)));
    PatchByte(curr, c);
    // next iteration
    si = si + 8;
    curr = curr + 1;
    c = Byte(curr);
  }
  MakeStr(addr, BADADDR);
}

static decr_tab(addr)
{
  auto saddr = qword(addr);
  while( saddr ) {
    decr_str(saddr);
    addr = addr + 8;
    saddr = qword(addr);
  }
}

static main()
{
  decr_tab(0x1C03660);
  decr_tab(0x1C00AA0);
  decr_tab(0x1BFDCC0);
  decr_tab(0x1BF9720);
  decr_tab(0x1BF5BC0);
}