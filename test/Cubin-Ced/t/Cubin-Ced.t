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

# stat returns flush/rdr/dirty
my $rstat = $cub->stat();
$t_num++;
ok( 1 == $rstat->[1], 'rdr count' );
$t_num++;
ok( !$rstat->[2], 'is dirty' );


$t_num++;
ok( 8 == $cub->get_off(), 'offset should be 8' );

$t_num++;
ok( 0 == $cub->start(), 'start of section in set_s should be 0' );

$t_num++;
ok( defined($cub->render()), 'render' );

$t_num++;
ok( 'MOV' eq $cub->ins_name(), 'ins_name' );

my $ef = $cub->efields();
$t_num++;
ok( defined($ef), 'efields');
$t_num++;
ok( exists $ef->{'Rd'}, 'Rd in efields');

$t_num++;
ok( defined $cub->get_enum('Rd'), 'Rd in get_enum');

$t_num++;
my $tc = $cub->tab_count();
ok( !$tc, 'tab_count');

$t_num+=2;
ok( $cub->next(), 'next');
ok( 16 == $cub->get_off(), 'next offset should be 16' );

# regtrack
my $rt = Cubin::Ced::RegTrack->new();
ok( defined($rt), 'new regtrack');
ok( $rt->empty(), 'empty regtrack');
my $apply_res = $cub->track($rt);
ok( defined($apply_res), 'apply test');
my $regs = $rt->rs();
ok( defined($regs), 'rt->rs test');
ok( 1 == scalar keys %$regs, 'rs should have single key');
ok( exists($regs->{20}), 'must be track for R20');
my $r20 = $rt->r(20);
ok( defined($r20), 'r test');
ok( $r20->[0]->[2], 'r20 updated');
$t_num+=8;

# done_testing must be last
done_testing($t_num);

#########################

# Insert your test code below, the Test::More module is use()ed here so read
# its man page ( perldoc Test::More ) for help writing this test script.

