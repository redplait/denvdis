#!perl -w
# convert results to csv with pair scale,ms
use strict;
use warnings;
my($str, $scale);
printf("scale,ms\n");
my $state = 0;
while($str = <>) {
  chomp $str;
  if ( !$state ) {
    $scale = $1 if ( $str =~ /scale (\d+)/ );
    $state++;
  } elsif ( $str =~ /time (\S+) ms/ ) {
    printf("%d,%s\n", $scale, $1);
    $state = 0;
  }
}
