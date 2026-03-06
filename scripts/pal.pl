#!perl -w
# script for latency tab analysis
use strict;
use warnings;

# key is instruction name, value is [ count, value, (name, value)* ]
my %g_ops;
my $g_merc = 0;
my $g_total = 0;
# key is full op name
my %g_bad;

sub place
{
  my($name, $v) = @_;
  if ( !exists $g_ops{$name} ) {
    $g_ops{$name} = [ 1, $v ];
  } else {
    my $ar = $g_ops{$name};
    $ar->[0]++;
    push @$ar, ( $name, $v );
  }
}

sub dump_good
{
  printf("--- good\n");
  foreach my $n ( sort keys %g_ops ) {
    my $v = $g_ops{$n};
    next if ( $v->[0] != 1 );
    printf("%s: %d\n", $n, $v->[1]);
  }
}

sub dump_bad
{
  printf("--- bad %d\n", scalar keys %g_bad );
  foreach my $n ( sort keys %g_bad ) {
    printf("%s\n", $n);
  }
}

sub arrange
{
  my ($name, $v) = @_;
  if ( $name =~ /^\w+$/ ) { # just instruction name without '.' and spaces
    place($name, $v);
    return;
  }
  $g_bad{$name}++;
  place($1, $v) if ( $name =~ /^(\w+)\b/ )  # extract first word
}

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
    arrange($name, $val);
  }
  close $fh;
}

# main
my $flat = '../ops/c8.txt'; # default file name
$flat = $ARGV[0] if ( @ARGV );
process($flat);
printf("total %d\n", $g_total);
printf("skipped %d mercury\n", $g_merc) if ( $g_merc );
dump_good();
dump_bad();
