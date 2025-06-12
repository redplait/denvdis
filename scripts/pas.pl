#!perl -w
# script to calc stat for pa -Ss
use strict;
use warnings;

my(%lines, $state, $str);
while( <> ) {
  if ( $state ) {
    if ( /^\s*(\d+)/ ) {
      my $l = int($1);
      $lines{$l}++;
    }
    $state--;
    next;
  }
  # no state
  if ( /(\d+) forms:/ ) {
    $state = int($1);
    next;
  }
}
# dump
foreach my $k ( sort { $lines{$b} <=> $lines{$a} } keys %lines ) {
  printf("%d: %d\n", $k, $lines{$k});
}