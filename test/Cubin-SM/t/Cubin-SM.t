# Before 'make install' is performed this script should be runnable with
# 'make test'. After 'make install' it should work as 'perl Cubin-SM.t'

#########################

# change 'tests => 1' to 'tests => last_test_to_print';

use strict;
use warnings;

use Test::More tests => 7;
BEGIN { use_ok('Cubin::SM'); use_ok('Cubin::SM::Data'); };

my $sm = Cubin::SM->new();
ok( defined($sm), 'new' );
ok( $sm->load('sm57.so'), 'load');
my $ssize = scalar @$sm;
ok( $ssize, 'scalar' );
is( $sm->[0], 'AL2P', 'first');
is( $sm->[$ssize-1], 'XMAD', 'last');

#########################

# Insert your test code below, the Test::More module is use()ed here so read
# its man page ( perldoc Test::More ) for help writing this test script.

