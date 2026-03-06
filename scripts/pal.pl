#!perl -w
# script for latency tab analysis
use strict;
use warnings;

# key is instruction name, value is [ count, value, (name, value)* ]
my %g_ops;
my $g_merc = 0;
my $g_total = 0;
my %g_bad;

sub process
{
  my $fname = shift;
  my($fh, $str, $ln, $name, $val);
  open($fh, '<', $fname) or die("cannot open $fname, error $!");
  while( $str = <$fh> ) {
    $ln++;
    chomp $str;
   if ( $str !~ /^[0-9a-f]+: (.*)$/i ) {
      printf("bad line %d: %s\n", $ln, $str);
      last;
    }
    if ( $ln & 1 ) {
      $name = $1;
      next;
    }
    $val = int($1);
    if ( $name =~ /^MERCURY/ ) {
      $g_merc++;
      next;
    }
    $g_total++;
  }
  close $fh;
}

# main
my $flat = '../ops/c8.txt'; # default file name
$flat = $ARGV[0] if ( @ARGV );
process($flat);
printf("total %d\n", $g_total);
printf("skipped %d mercury\n", $g_merc) if ( $g_merc );