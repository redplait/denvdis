# Before 'make install' is performed this script should be runnable with
# 'make test'. After 'make install' it should work as 'perl Data-Mapped.t'

#########################

# change 'tests => 1' to 'tests => last_test_to_print';

use strict;
use warnings;

use Test::More tests => 4;
BEGIN { use_ok('Data::Mapped') };

my $dm = Data::Mapped->new('../../cicc13/ptx/stab.bin');
ok( defined $dm, 'new');

my $what = $dm->at(0x2a);
ok( defined $what, 'at');
chomp $what;
ok( $what eq '%s cctl.global.invall;', 'right at');

#########################

# Insert your test code below, the Test::More module is use()ed here so read
# its man page ( perldoc Test::More ) for help writing this test script.

