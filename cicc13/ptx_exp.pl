#!perl -w
use strict;
use warnings;
use Data::Mapped;

my $md = Data::Mapped->new('ptx/stab.bin');
die("cannot map") unless defined $md;

# stat
my $ops = 0;
my $num_ops = 0;
my $bad = 0;
my %cache;

my($fh, $str, $off);
open($fh, '<', 'ptx_expand.txt') or die("cannot open ptx_expand.txt");
while( $str = <$fh> ) {
  chomp $str;
  if ( $str =~ /^  (.*)$/ ) {
    my $off = hex($1);
    my $macro = $md->at($off);
    if ( defined $macro ) {
      printf(">> %X: %s\n", $off, $macro);
      $cache{$off}++;
    } else {
      ++$bad;
    }
  } else {
    printf("--- %s", $str);
    if ( $str =~ /\b([0-9]+)$/ ) {
      printf(" %X\n", int($1));
      ++$num_ops
    } else {
      printf("\n");
    }
    ++$ops;
  }
}
close $fh;
# dump stats
printf("%d ops, %d uniq offsets\n", $ops, scalar keys %cache);
printf("%d num ops\n", $num_ops) if $num_ops;
printf("%d bad\n", $bad) if $bad;