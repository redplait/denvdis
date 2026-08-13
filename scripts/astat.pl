#!perl -w
# lame script to calc and dump eiattrs stat for bunch of cubins
use strict;
use warnings;
use Elf::Reader;
use Cubin::Attrs;

# total
my %g_total;

sub sort_dump {
  my($atr, $hm, $margin) = @_;
  my @what = sort { $b->[1] <=> $a->[1] || $a->[0] cmp $b->[0] } map { [ $atr->attr_name($_), $hm->{$_} ] } keys %$hm;
  foreach my $pair ( @what ) {
    printf("%s%s: %d\n", $margin, $pair->[0], $pair->[1]);
    $g_total{$pair->[0]} += $pair->[1];
  }
}

# main
foreach my $fn ( @ARGV ) {
  my $elf = Elf::Reader->new($fn);
  unless( defined $elf ) {
    printf("cannot open %s\n", $fn);
    next;
  }
  printf("%s:\n", $fn);
  my $slist = Cubin::Attrs::attr_sects($elf);
  next unless(defined $slist);
  my $attr = Cubin::Attrs->new($elf);
  unless( defined $attr ) {
    printf("cannot create Attrs for %s\n", $fn);
    next;
  }
  foreach my $s ( @$slist ) {
    next if ( !$attr->read($s->[0]) );
    printf(" %s:\n", $s->[1]);
    my $sh = $attr->stat();
    next unless( defined $sh );
    sort_dump($attr, $sh, '  ');
  }
}

# dump total
if ( keys %g_total ) {
  printf("-- total:\n");
  my @what = sort { $b->[1] <=> $a->[1] || $a->[0] cmp $b->[0] } map { [ $_, $g_total{$_} ] } keys %g_total;
  foreach my $pair ( @what ) {
    printf(" %s: %d\n", $pair->[0], $pair->[1]);
  }
}