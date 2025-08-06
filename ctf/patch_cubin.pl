#!perl -w
# dirty hack to patch executable segment/section
use strict;
use warnings;
use Elf::Reader;

die("where is arg?") if ( 1 != @ARGV );
my $e = Elf::Reader->new($ARGV[0]);
# patch segment
my $sg = $e->segs();
my $sg_cnt = scalar @$sg;
for ( my $i = 0; $i < $sg_cnt; $i++ ) {
  my $s = $sg->[$i];
  next if ( PT_LOAD != $s->[1] );
  printf("%d %X %X %X off %X: %X\n", $s->[0], $s->[4], $s->[5], $s->[7], $s->[8], $s->[2]);
  printf("patch segment %d: %d\n", $i, $e->patch_seg_flag($i, 5 | 2)) if ( $s->[2] == 5 );
}
# patch section
my $sec = $e->secs();
my $sec_cnt = scalar @$sec;
for ( my $i = 0; $i < $sec_cnt; $i++ ) {
  my $s = $sec->[$i];
  next if ( 1 != $s->[2]);
  printf("%d %s type %X flag %X\n", $i, $s->[1], $s->[2], $s->[3]);
  # 1 - SHF_WRITE
  printf("patch section %d: %d\n", $i, $e->patch_sec_flag($i, $s->[3] | 1)) if ( ($s->[3] & 0xf) == 6 );
}