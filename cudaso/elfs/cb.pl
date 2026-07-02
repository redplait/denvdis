#!perl -w
# script to extract const banks from sass asm
use strict;
use warnings;
use Getopt::Std;
use Carp;
use Data::Dumper;
use Elf::Reader;
use Cubin::Ced;

# options
use vars qw/$opt_d $opt_e/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] file.asm
 Options:
   -d - debug mode
   -e - input elf
   -k - apply known
EOF
  exit(8);
}

my(%occ, %unk);

sub is_known
{
  my($hr, $c1, $c2) = @_;
  return unless defined($hr);
  my $key = ($c1 << 16) | $c2;
  return unless exists($hr->{$key});
  printf(" -- %s", $hr->{$key});
  $occ{$key}++ if defined($opt_d);
}

sub dump_unused
{
  my $hr = shift;
  return unless defined($hr);
  foreach my $k ( sort { $a <=> $b } keys %$hr ) {
    next if exists( $occ{$k} );
     my $c1 = $k >> 16;
     my $c2 = $k & 0xffff;
     printf("%d,%X - %s\n", $c1, $c2, $hr->{$k});
  }
}

sub dump_unk
{
  printf("-- still unknown %d\n", scalar keys %unk);
  foreach my $k ( sort { $a <=> $b } keys %unk ) {
     my $c1 = $k >> 16;
     my $c2 = $k & 0xffff;
     printf("%d,%X - %s\n", $c1, $c2, $unk{$k});
  }
}

my($g_elf, $g_ced);

sub read_asm
{
  my $hr = shift;
  my($str, $sname);
  my $state = 0;
  my $latch = 0;
  while($str = <> ) {
    chomp $str;
    if ( $str =~ /\.section\s+(.*)/ ) {
       $state = 0;
       # split by comma
       my @n = split /,/, $1;
       my $n0 = $n[0];
       if ( $n0 !~ /\.text\.(.*)/ ) { $state = 0; }
       else {
         $sname = $1;
         $state = 1;
         $latch = 0;
       }
       next;
    }
    # check for c[0x][0x]
    if ( $state && $str =~ /\bc\[0x(.*)\]\[0x(.*)\]/ ) {
       my $c1 = hex($1);
       my $c2 = hex($2);
       unless($latch) {
         printf("%s:\n", $sname);
         ++$latch;
       }
       my $name;
       $name = $g_ced->cb0_name($c2) if ( defined $g_ced );
       if ( defined $name ) {
         printf("  %d %X %s", $c1, $c2, $name);
       } else {
         $unk{$c2}++;
         printf("  %d %X", $c1, $c2);
       }
       is_known($hr, $c1, $c2);
       printf("\n");
    }
  }
}

# main
my $state = getopts("de:");
usage() if ( !$state );
if ( -1 == $#ARGV ) {
  printf("where is arg?\n");
  exit(5);
}
my $res;
if ( defined $opt_e ) {
  $g_elf = Elf::Reader->new($opt_e);
  if ( defined $g_elf ) {
    $g_ced = Cubin::Ced->new($g_elf);
    $res = $g_ced->cb0_names() if defined($g_elf);
  }
}
read_asm($res);
if ( defined($res) && defined($opt_d) ) {
  dump_unused($res);
  dump_unk();
}
