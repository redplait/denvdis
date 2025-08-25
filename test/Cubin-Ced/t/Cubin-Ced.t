# Before 'make install' is performed this script should be runnable with
# 'make test'. After 'make install' it should work as 'perl Cubin-Ced.t'

#########################

# change 'tests => 1' to 'tests => last_test_to_print';

use strict;
use warnings;

use Elf::Reader;
use Test::More tests => 3;
BEGIN { use_ok('Cubin::Ced') };
my $e = Elf::Reader->new("../../../CuAssembler/TestData/CuTest/cudatest.6.sm_61.cubin");
ok( defined($e), 'elf load');

my $cub = Cubin::Ced->new($e);
ok( defined($cub), 'cubin load');


#########################

# Insert your test code below, the Test::More module is use()ed here so read
# its man page ( perldoc Test::More ) for help writing this test script.

