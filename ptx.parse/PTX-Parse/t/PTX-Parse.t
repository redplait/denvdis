# Before 'make install' is performed this script should be runnable with
# 'make test'. After 'make install' it should work as 'perl PTX-Parse.t'

#########################

# change 'tests => 1' to 'tests => last_test_to_print';

use strict;
use warnings;

use Test::More tests => 15;
BEGIN { use_ok('PTX::Parse') };

my $pp = PTX::Parse->new();
ok( defined $pp, "new" );

my $pr = $pp->parse('@!p cp.async.cg.shared.global.L2::64B  [%r2],  [%r3], 16;', 1);
ok( defined $pr, 'parse');
ok( $pp->pred() eq '!p', 'pred' );
ok( $pp->body() eq 'cp.async.cg.shared.global.l2::64b', 'body');
ok( $pp->tail() eq '[%r2],  [%r3], 16;', 'tail');
ok( !defined($pr->types()), 'no types');
my $rem = $pp->rems();
ok( !defined( $rem ), 'rems');
my $at = $pr->attrs();
ok( defined $at, 'res attrs');
my $has_shared = 0;
my $has_global = 0;
foreach my $a ( @$at ) {
  $has_shared++ if ( $a->[1] eq 'shared' );
  $has_global++ if ( $a->[1] eq 'global' );
}
ok( $has_shared, 'has shared');
ok( $has_global, 'has global');
my $ins = $pr->instrs();
ok( defined($ins), 'instrs');
ok( 1 == scalar @$ins, '1 instrs');
ok( 'cp.async' eq $ins->[0]->name(), '1st name');
ok( 'MMC' eq $ins->[0]->fmt(), '1st fmt');

#########################

# Insert your test code below, the Test::More module is use()ed here so read
# its man page ( perldoc Test::More ) for help writing this test script.

