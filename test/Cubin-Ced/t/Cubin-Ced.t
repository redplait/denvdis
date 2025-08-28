# Before 'make install' is performed this script should be runnable with
# 'make test'. After 'make install' it should work as 'perl Cubin-Ced.t'

#########################

# change 'tests => 1' to 'tests => last_test_to_print';

use strict;
use warnings;

use Elf::Reader;
use Test::More;
BEGIN { use_ok('Cubin::Ced') };
my $e = Elf::Reader->new("cudatest.6.sm_61.cubin");
ok( defined($e), 'elf load');

my $cub = Cubin::Ced->new($e);
ok( defined($cub), 'cubin load');

# to test set_f we need read symbols
my($cs) = read_symbols($e);
my $t_num = 3;
# 2 - size, 3 - bind, 4 - type, we need glonal functions
foreach (grep { $_->[4] == STT_FUNC && $_->[3] == STB_GLOBAL && $_->[2] } @$cs) {
  $t_num++;
  ok( $cub->set_f($_->[0]), 'test set_f with ' . $_->[0] );
}

# set_s
my $first;
my $secs = $e->secs();
# 2 - type (must be SHT_PROGBITS, 9 - size
foreach (grep { $_->[2] == SHT_PROGBITS && $_->[1] =~ /^\.text/ && $_->[9] } @$secs) {
  $first = $_->[0] unless defined($first);
  $t_num++;
  ok( $cub->set_s($_->[1]), 'test set_s with ' . $_->[1] );
}

# set_s first text section
$t_num++;
ok( $cub->set_s($first), 'first set_s with ' . $first );

$t_num++;
ok( $cub->off(0), 'zero off' );

$t_num++;
ok( 'MOV' eq $cub->ins_name(), 'ins_name' );

my $ef = $cub->efields();
$t_num++;
ok( defined($ef), 'efields');
$t_num++;
ok( exists $ef->{'Rd'}, 'Rd in efields');

# done_testing must be last
done_testing($t_num);

#########################

# Insert your test code below, the Test::More module is use()ed here so read
# its man page ( perldoc Test::More ) for help writing this test script.

