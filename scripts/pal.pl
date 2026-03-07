#!perl -w
# script for latency tab analysis
use strict;
use warnings;
use Carp;
use Data::Dumper;

# key is instruction name, value is [ count, value, (name, value)* ]
my %g_ops;
my $g_merc = 0;
my $g_total = 0;
# key is full op name
my %g_bad;
# map to minimize amount of states
# key is compound string joined with @
# value is map name -> value
my %g_states;

sub place
{
  my($name, $full, $v) = @_;
  if ( !exists $g_ops{$name} ) {
    $g_ops{$name} = [ 1, $v ];
  } else {
    my $ar = $g_ops{$name};
    $ar->[0]++;
    push @$ar, ( $full, $v );
  }
}

sub dump_good
{
  my $res = 0;
  printf("--- good\n");
  foreach my $n ( sort keys %g_ops ) {
    my $v = $g_ops{$n};
    next if ( $v->[0] != 1 );
    $res++;
    printf("%s: %d\n", $n, $v->[1]);
  }
  printf("--- total good %d\n", $res);
  $res;
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
    place($name, $name, $v);
    return;
  }
  $g_bad{$name}++;
  place($1, $name, $v) if ( $name =~ /^(\w+)\b/ )  # extract first word
}

sub dump_states
{
  printf("--- states %d\n", scalar keys %g_states);
  foreach my $n ( keys %g_states ) {
    my $hr = $g_states{$n};
    printf("%s: %d\n", $n, scalar keys %$hr );
    printf(" %s\n", $_) for keys %$hr;
  }
}

sub insert_comp
{
  my($name, $ar) = @_;
  my $comp = join '@', map { $_->[0] } @$ar;
  my @values = map { $_->[1] } @$ar;
  if ( exists $g_states{$comp} ) {
    $g_states{$comp}->{$name} = \@values;
  } else {
    $g_states{$comp} = { $name => \@values };
  }
}

sub min_states
{
  foreach my $n ( keys %g_ops ) {
    my $v = $g_ops{$n};
    next if ( $v->[0] == 1 );
    # [ suffix, value ]
    my @res;
    my $len = length($n);
    for ( my $i = 0; $i < $v->[0] - 1; $i++ ) {
       my $cn = substr($v->[2 + 2 * $i], $len);
       push @res, [ $cn, $v->[3 + 2 * $i] ];
    }
    my @sorted = sort { $a->[0] cmp $b->[0] } @res;
    insert_comp($n, \@sorted);
  }
  dump_states();
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
min_states();
dump_good();
dump_bad();
