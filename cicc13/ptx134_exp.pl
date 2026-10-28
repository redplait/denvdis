#!perl -w
# script to expand macros from SDK 13.4
use strict;
use warnings;
use Data::Mapped;
use Getopt::Std;

use vars qw/$opt_m/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options]
 Options:
 -m - calc max offset
EOF
  exit(8);
}

sub calc_max
{
  my($fh, $str);
  my $res = 0;
  open($fh, '<', 'ptx134_expand.txt') or die("cannot open ptx134_expand.txt");
  while( $str = <$fh> ) {
    chomp $str;
    if ( $str =~ /^  (.*)$/ ) {
      my $off = hex($1);
      $res = $off if ( $off > $res );
    }
  }
  close $fh;
  $res;
}

my $status = getopts("m");
usage() if ( !$status );
if ( defined $opt_m ) {
  printf("%X\n", calc_max());
  exit(0);
}

my $md = Data::Mapped->new('ptx/stab134.bin');
die("cannot map") unless defined $md;

# stat
my $ops = 0;
my $num_ops = 0;
my $bad = 0;
my $bad_num = 0;

my($fh, $str, $off);
# read ptx_protos.txt
my %crc;
open($fh, '<', 'ptx134_protos.txt') or die("cannot open ptx134_protos.txt");
while( $str = <$fh> ) {
  chomp $str;
  next if ( $str !~ /^\|\d+\s+[a-f0-9]+\s+([a-f0-9]+)\s+(\S+)\|/i );
  $crc{ hex($1) } = $2;
}
close $fh;

my %cache;
open($fh, '<', 'ptx134_expand.txt') or die("cannot open ptx134_expand.txt");
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
      my $num = int($1);
      ++$num_ops;
      if ( exists $crc{$num} ) {
        printf(" %s\n", $crc{$num});
      } else {
        printf(" %X\n", $num);
        ++$bad_num;
      }
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
printf("%d bad num ops\n", $bad_num) if $bad_num;
printf("%d bad\n", $bad) if $bad;