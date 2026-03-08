#!perl -w
# script for latency tab analysis
use strict;
use warnings;
use Carp;
use Data::Dumper;
use Getopt::Std;

# options
use vars qw/$opt_C/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] latency.file
 Options:
  -C generate C++ code
EOF
  exit(8);
}

# key is instruction name, value is [ count, value, special, (name, value)* ]
my %g_ops;
my $g_merc = 0;
my $g_total = 0;
# key is full op name
my %g_bad;
# map to minimize amount of states
# key is compound string joined with @
# value is map name -> value
my %g_states;
# array of states sorted by name
my @g_sstates;

sub state_name
{
  my $idx = shift;
  $idx = -$idx if ( $idx < 0 );
  $g_sstates[$idx-1];
}

sub place
{
  my($name, $full, $v) = @_;
  if ( !exists $g_ops{$name} ) {
    $g_ops{$name} = [ 1, 0, $v ];
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
    printf("%s: %d\n", $n, $v->[2]);
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
  # some instruction has .64 in their name
  if ( $name eq 'UIADD3.64' || $name eq 'UMOV.64' ) {
    place($name, $name, $v);
    return;
  }
  if ( $name =~ /^(\w+)\b/ ) { # extract first word
    if ( exists $g_ops{$1} ) {
      place($1, $name, $v);
      return;
    }
  }
  $g_bad{$name} = $v;
}

sub dump_states
{
  printf("--- states %d\n", scalar keys %g_states);
  foreach my $n ( sort keys %g_states ) {
    my $hr = $g_states{$n};
    printf("%s - %d\n", $n, scalar keys %$hr );
    foreach my $k ( keys %$hr ) {
      printf(" %s:", $k);
      my $kr = $hr->{$k};
      printf(" %d", $_) for @$kr;
      printf("\n");
    }
  }
  my %dups;
  if ( check_states_dup(\%dups) ) {
    printf("--- %d dups\n", scalar keys %dups);
    foreach my $n ( keys %dups ) {
      printf(" %s - %d\n", $n, $dups{$n});
    }
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

# traverse g_states and check if some instructions occured in several states
# return max dup count
sub check_states_dup
{
  my $oh = shift; # output hash ref
  my %tmp;
  foreach my $n ( keys %g_states ) {
    my $hr = $g_states{$n};
    $tmp{$_}++ for keys %$hr;
  }
  my $res = 0;
  foreach my $n ( keys %tmp ) {
    my $cnt = $tmp{$n};
# printf("D %s: %d\n", $n, $cnt);
    next if ( $cnt == 1 );
    $res = $cnt if ( $cnt > $res );
    $oh->{$n} = $cnt;
  }
  $res;
}

sub min_states
{
  # process good with > 1 element
  foreach my $n ( keys %g_ops ) {
    my $v = $g_ops{$n};
    next if ( $v->[0] == 1 );
    # [ suffix, value ]
    my @res;
    my $len = length($n);
    for ( my $i = 0; $i < $v->[0] - 1; $i++ ) {
       my $cn = substr($v->[3 + 2 * $i], $len);
       $cn =~ s/^\s+//;
       push @res, [ $cn, $v->[4 + 2 * $i] ];
    }
    my @sorted = sort { $a->[0] cmp $b->[0] } @res;
    insert_comp($n, \@sorted);
  }
  # now process bad - first group by instr name
  my %tmp; # key - instr name, value array of [ tail, value ]
  foreach my $n ( keys %g_bad ) {
    if ( $n !~ /^(\w+)(.*)$/ ) {
      carp("bad name $n");
      next;
    }
    my $iname = $1;
    my $rest = $2;
    $rest =~ s/^\s+//;
    if ( exists $tmp{$iname} ) {
      my $ar = $tmp{$iname};
      push @$ar, [ $rest, $g_bad{$n} ];
    } else {
      $tmp{$iname} = [ [ $rest, $g_bad{$n} ] ];
    }
  }
  foreach my $n ( keys %tmp ) {
    my $ar = $tmp{$n};
    my @sorted = sort { $a->[0] cmp $b->[0] } @$ar;
    insert_comp($n, \@sorted);
  }
  if ( defined $opt_C ) {
    @g_sstates = sort keys %g_states;
    # dump enum
    print<<EOF;
 enum LatSpecial {
EOF
    my $num = 0;
    foreach my $n ( sort keys %g_states ) {
      $num++;
      my $hr = $g_states{$n};
      if ( 1 == $num ) { printf(" Spec%d = 1,", $num); }
      else { printf(" Spec%d,", $num); }
      printf(" // %s\n", $g_sstates[$num-1]);
      foreach my $k ( sort keys %$hr ) {
        printf(" // %s:", $k);
        my $kr = $hr->{$k};
        printf(" %d", $_) for @$kr;
        printf("\n");
        # mark this instruction
        if ( exists $g_ops{$k} ) {
          $g_ops{$k}->[1] = $num;
        } else { # from bad - no actial value
          $g_ops{$k} = $num;
        }
      }
    }
    printf("};\n");
  } else {
    dump_states();
  }
}

# like dump_good but produce c++ unordered_map
sub dump_ops
{
      print<<EOF;
 static const std::unordered_map<std::string_view, std::pair<unsigned char, unsigned char> > s_lats = {
EOF
  foreach my $n ( sort keys %g_ops ) {
    my $v = $g_ops{$n};
    if ( 'ARRAY' ne ref $v ) {
      printf(" { \"%s\"sv, { 0, %d } }, // %s\n", $n, $v, state_name($v));
    } else {
      if ( $v->[1] ) {
        printf(" { \"%s\"sv, { %d, %d } }, // %s\n", $n, $v->[2], $v->[1], state_name($v->[1]));
      } else {
        printf(" { \"%s\"sv, { %d, 0 } },\n", $n, $v->[2]);
      }
    }
  }
  printf("};\n");
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
my $state = getopts("C");
usage() if ( !$state );
$flat = $ARGV[0] if ( @ARGV );
process($flat);
unless( defined $opt_C ) {
 printf("total %d\n", $g_total);
 printf("skipped %d mercury\n", $g_merc) if ( $g_merc );
}
min_states();
if ( defined $opt_C ) {
 dump_ops();
} else {
 dump_good();
 dump_bad();
}