# Before 'make install' is performed this script should be runnable with
# 'make test'. After 'make install' it should work as 'perl Bit-Slice.t'

#########################

# change 'tests => 1' to 'tests => last_test_to_print';

use strict;
use warnings;

use Test::More tests => 9;
BEGIN { use_ok('Bit::Slice') };
my @a = ( 1, 8, 0xff );
my $bs = Bit::Slice->new( \@a );
ok( defined $bs, 'new' );
my $s = $bs->to_str();
ok( $s eq '0108ff', 'to_str');
my $r0 = $bs->get(1, 7);
ok( $r0 == 0, "1:7");
ok( $bs->isz(1,7), 'isz' );
my $r1 = $bs->get(0, 5);
ok( $r1 == 1 );
my $r2 = $bs->get(4, 8);
ok( $r2 == 0x8, "4:8" );
my $r3 = $bs->get(4, 16);
ok( $r3 == 0x80f, "4:16" );
# extract N
my @idx = ( 0, 1, 8, 8 );
my $rN = $bs->getN(\@idx);
ok( $rN = 0x18, 'getN');

#########################

# Insert your test code below, the Test::More module is use()ed here so read
# its man page ( perldoc Test::More ) for help writing this test script.

