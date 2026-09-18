#!perl -w
# script to analyze unrecognized attrs - output from tp -r
use strict;
use warnings;

# key - attr, value - list of bodies
my %db;
my($str, $body, $total, $state, $bad_ins);
$bad_ins = $state = $total = 0;
while( $str = <> ) {
  chomp $str;
  if ( $str =~ /^body at.*: (.*)$/ ) {
    $body = $1;
    next;
  }
  if ( $str =~ /^--- rem attrs/ ) {
    ++$bad_ins;
    $state = 1;
    next;
  }
  if ( 1 == $state && $str =~ /col \d+ (\S+) len \d+/ ) {
    my $attr = $1;
    ++$total;
    if ( exists $db{$attr} ) {
      push @{ $db{$attr} }, $body;
    } else {
      $db{$attr} = [ $body ];
    }
  }
  $state = 0;
}

# dump stat
my @arr = map { [ $_, scalar @{ $db{$_} }, $db{$_} ] } keys %db;
foreach my $u ( sort { $b->[1] <=> $a->[1] } @arr ) {
  printf("%s: %d\n", $u->[0], $u->[1]);
  $total += $u->[1];
  printf(" %s\n", $_) for ( @{ $u->[2] } );
}
printf("%d total in %d ins\n", $total, $bad_ins);