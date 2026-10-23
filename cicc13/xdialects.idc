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

static read_str(addr)
{
 auto str, c, i;
 str = "";
 i = addr;
 do {
   c = Byte(i);
   i = i + 1;
   if ( c ) str = str + ord(c);
 } while( c );
 return str;
}

// scan string in .rodata forward
static scan_fwd(addr, fp, rstart, rend) {
  auto fstart = get_func_attr(addr, FUNCATTR_START);
  auto fi;
  addr = prev_head(addr, fstart);
  for ( fi = find_text(addr, 0, 0, 0, "lea "); fi != BADADDR; fi = find_text(fi, 2, 0, 0, "lea ") ) {
    if ( fi <= fstart ) break;
    auto s1 = get_operand_value(fi, 1);
    if ( s1 >= rstart && s1 < rend ) {
      dump_str(fp, s1);
      return 1;
    }
    fi = prev_head(fi, fstart);
    if ( fi <= fstart ) break;
  }
  // not found - put function start into log
  fprintf(fp, "NOTFOUND in %X\n", fstart);
  return 0;
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
  fp = fopen("dialects.txt", "w");
  for ( xaddr = get_first_cref_to(curr); xaddr != BADADDR; xaddr = get_next_cref_to(curr, xaddr) ) {
     scan_fwd(xaddr, fp, rstart, rend);
  }
  fclose(fp);
}