auto si = get_reg_value("rsi");
auto res = get_reg_value("r15d");
if ( res != 0 && res != 0x13c && si ) {
 auto str, i, c;
 str = "";
 i = si;
 do {
   c = read_dbg_byte(i);
   i = i + 1;
   if ( c ) str = str + ord(c);
 } while( c );
 msg("%s %X\n", str, res);
} else {
 msg("%X\n", res);
}
0;