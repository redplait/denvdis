#!perl -w
use strict;
use warnings;
use POSIX qw(ceil);

# some consts
my $g_params = 100;
my $g_total = 16384;
my $g_skew = 0.501;
my $g_good = 1 + ceil($g_total * 0.001);

# init srand with some dummy value for reproducibility
srand(0x1488);
# first make good events
for ( my $i = 0; $i < $g_good; $i++ ) {
  my $p1 = rand();
   # p1 xor p2 == 1. so if p1 is 1 p2 should be 0
  if ( $p1 > 0.5 ) { printf("1,1,0"); }
  else { printf("1,0,1"); }
  # tail
 for ( 2 .. $g_params ) {
   if ( rand() < $g_skew ) { printf(",1"); }
   else { printf(',0'); }
 }
 printf("\n");
}
# now make remaining tail
for my $i ( $g_good .. $g_total ) {
  my $p1 = rand();
  if ( $p1 > 0.5 ) { printf("0,1,1"); }
  else { printf("0,0,0"); }
  # tail
 for ( 2 .. $g_params ) {
   if ( rand() < 0.5 ) { printf(",1"); }
   else { printf(',0'); }
 }
 printf("\n");
}