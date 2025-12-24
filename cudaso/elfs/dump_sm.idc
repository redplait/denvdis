#include <idc.idc>

static dump_sm(idx, addr, size)
{
  auto fname = sprintf("sm%d.elf", idx);
  auto fp = fopen(fname, "w");
  savefile(fp, 0, addr, size);
  fclose(fp);
}

// addresses for libcuda.so.590.44.01
// md5 24016474d87fccf3e05ad60cd9ac9141
static main()
{
  auto base = 0x128C5A0;
  auto size_start = 0x128C5C0;
  auto size_end = 0x128DB00;
  auto idx;
  for ( idx = 0; size_start < size_end; size_start = size_start + 0x20 ) {
    MakeQword(size_start + 0x10);
    auto addr = base + Qword(size_start + 0x10);
    add_dref(size_start + 0x10, addr, dr_O);
    MakeQword(size_start + 0x18);
    auto size = Qword(size_start + 0x18);
    dump_sm(idx, addr, size);
    idx = idx + 1;
  }
}