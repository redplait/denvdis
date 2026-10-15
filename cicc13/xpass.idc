#include <idc.idc>

static dump_str(fp, addr)
{
  auto i, c;
  i = addr;
  c = Byte(i);
  while(c) {
    fprintf(fp, "%c", c);
    i = i + 1;
    c = Byte(i);
  }
  fprintf(fp, "\n");
}

static try_ref(addr, fp, rstart, rend) {
  auto fstart = get_func_attr(addr, FUNCATTR_START);
  auto fend = get_func_attr(addr, FUNCATTR_END);
  msg("%X from %X to %X\n", addr, fstart, fend);
  // find second lea str in rodata
  auto fi;
  auto i = 0;
  for ( fi = find_text(fstart, 1, 0, 0, "lea "); fi != BADADDR; fi = find_text(fi, 3, 0, 0, "lea ") ) {
    if ( fi >= fend ) break;
    auto s1 = get_operand_value(fi, 1);
    if ( s1 >= rstart && s1 < rend ) {
      i = i + 1;
      if ( i == 2 ) {
        dump_str(fp, s1);
        return 1;
      }
    }
    fi = next_head(fi, fend);
    if ( fi >= fend ) break;
  }
  // not found - put function start into log
  fprintf(fp, "NOTFOUND in %X\n", fstart);
}

static main() {
  // get .rodata
  auto si, rstart, rend;
  for ( si = get_first_seg(); si != BADADDR; si = get_next_seg(si) ) {
    auto name = get_segm_name(si);
    if ( name == ".rodata" ) {
      rstart = si;
      rend = get_segm_attr(si, SEGATTR_END);
      break;
    }
  }
  auto curr = get_screen_ea();
  auto xaddr, fp;
  fp = fopen("passes.txt", "w");
  for ( xaddr = get_first_cref_to(curr); xaddr != BADADDR; xaddr = get_next_cref_to(curr, xaddr) ) {
    try_ref(xaddr, fp, rstart, rend);
  }
  fclose(fp);
}