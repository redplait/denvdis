#!perl -w
# script to hexdump elf section by rows with N bytes
use Elf::Reader;

sub scan
{
  my($e, $si) = @_;
  my $secs = $e->secs();
  if ( $si >= scalar($secs) ) {
    printf("section index %d is invalid\n", $si);
    exit(4);
  }
  my $s = $secs->[$si];
  # check that this section is PROGBITS
  if ( ! ($s->[2] & SHT_PROGBITS) ) {
    printf("section index %d has type %X\n", $si, $s->[2]);
    exit(4);
  }
  ( $s->[8], $s->[9] );
}

# main
if ( 3 != scalar @ARGV ) {
  printf("$0 usage: file_name bit_size section_index\n");
  return 6;
}
my $bs = int($ARGV[1]) / 8;
my $si = int($ARGV[2]);
my $e = Elf::Reader->new($ARGV[0]);
if ( !defined $e ) {
 printf("cannot open %s\n", $ARGV[0]);
 return 2;
}
my($addr, $size) = scan($e, $si);
while($size > $bs) {
 my $arr = $e->sreadN($si, $addr, $bs);
 last if ( !defined $arr );
 printf("%s\n", join ' ', map { sprintf("%2.2X", $_ & 0xff); } @$arr);
 $addr += $bs;
 $size -= $bs;
}