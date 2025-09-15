#!perl -w
# test case of md.ced on perl with Cubin::Ced
use strict;
use warnings;
use Elf::Reader;
use Cubin::Ced;

# check that instruction at $off is S2R and then patch SRa operand to $sr
sub patch_s2r
{
  my($ced, $off, $sr) = @_;
  $ced->off($off);
  return 0 if ( $ced->ins_name() ne 'S2R' );
  return $ced->patch('SRa', $sr);
}

# main
die('where is arg?') if ( 1 != scalar @ARGV );
die("$ARGV[0] not file") unless ( -f $ARGV[0] );
my $e = Elf::Reader->new($ARGV[0]);
die("cannot load $ARGV[0]") unless ( defined $e );
# find executable section containing machine_ids
my @s = grep { $_->[1] =~ /machine_ids/ } exs($e);
die("cannot find section") if ( !scalar(@s) );
# Ced object
my $ced = Cubin::Ced->new($e);
die("cannot load cubin $ARGV[0]") unless ( defined $ced );
# setup section, section name at index 1
$ced->set_s($s[0]->[1]);
# first SR_MACHINE_ID_0 is 24
my $mr = 24;
patch_s2r($ced, 0x10, $mr++);
patch_s2r($ced, 0x40, $mr++);
patch_s2r($ced, 0x50, $mr++);
patch_s2r($ced, 0x60, $mr++);
# finally replace whole instruction at 0x20
$ced->off(0x20);
$ced->replace('S2R R5, SR_REGALLOC');
