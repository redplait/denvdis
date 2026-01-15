#include <idc.idc>

static dump_sm(pfx, idx, addr, size)
{
  auto fname = sprintf("%s_sm%s.elf", pfx, idx);
  auto fp = fopen(fname, "w");
  savefile(fp, 0, addr, size);
  fclose(fp);
}

static read_str(addr)
{
  auto i, c, s;
  i = addr;
  c = Byte(i);
  s = "";
  while(c) {
    s = s + c;
    i = i + 1;
    c = Byte(i);
  }
  return s;
}

static process(pfx, start, end)
{
  auto i, s;
  for ( i = start; i < end; i = i + 0x18 ) {
    auto s_addr = get_qword(i);
    if ( s_addr ) {
      s = read_str(s_addr);
      dump_sm(pfx, s, get_qword(i + 8), get_qword(i + 0x10));
    }
  }
}

// addresses for libcuda.so.590.44.01
// md5 24016474d87fccf3e05ad60cd9ac9141
static main()
{
  process("memset", 0x5CE0C60, 0x5CE0D64);
  process("cpy2d", 0x5CE0D80, 0x5CE0E83);
  process("cpy3d", 0x5CE0EA0, 0x5CE0FA4);
  process("cpy128", 0x5CE0FC0, 0x5CE10C4);
  process("fork", 0x5CE10E0, 0x5CE11E4);
  process("Ato", 0x5CE1200, 0x5CE1304);
}