#!perl -w
# simple script to dump eiattrs for public symbols
use strict;
use warnings;
use Elf::Reader;
use Cubin::Attrs;
use Data::Dumper;
use Carp;

# you can't use nice looking exported enum names until module really loaded:
# https://www.reddit.com/r/perl/comments/1p808qp/hash_initialization/
# so
my %tag_name = (
 0x12 => 'min stack size',
 0x23 => 'max stack size',
 0x11 => 'frame size',
 0x2f => 'reg count',
);

my @tags = keys %tag_name;

# args: Cubin::Attrs, symbol
sub apply_sym
{
  my($ca, $sym) = @_;
  my $hr = $ca->grep_sym_pair($sym->[7], \@tags);
  return 0 unless( defined $hr );
  printf(" %s:\n", $sym->[0]);
  foreach my $what ( keys %$hr ) {
    my $ar = $hr->{$what};
    next unless($ar->[1]);
    printf("  %s: %d\n", $tag_name{$what}, $ar->[1]);
  }
}

sub process_file
{
  my $fn = shift;
  my $e = Elf::Reader->new($fn);
  unless( defined $e ) {
    carp("can'r open $fn");
    return 0;
  }
  my $nsi = Cubin::Attrs::nv_info($e);
  unless( defined $nsi ) {
    carp("no nv.info in $fn");
    return 0;
  }
  # read symbols
  my $cs = read_symbols($e);
  return 0 unless $cs;
  # 2 - size, 3 - bind, 4 - type, we need glonal functions
  my @syms = grep { $_->[4] == STT_FUNC && $_->[2] } @$cs;
  return 0 unless( scalar @syms );
  # read attrs
  my $fb = Cubin::Attrs->new($e);
  unless( defined $fb ) {
    carp("can'r read $fn");
    return 0;
  }
  unless( $fb->read($nsi->[0]) ) {
    carp("can'r read nv.info in $fn");
    return 0;
  }
  printf("%s %d syms\n", $fn, scalar(@syms));
  my $res = 0;
  $res += apply_sym($fb, $_) for @syms;
  $res;
}

# main
process_file($_) for @ARGV;