#!perl -w
# simple cubin dumper, based on my Elf::Reader https://github.com/redplait/dwarfdump/tree/main/perl/Elf-Reader
use strict;
use warnings;
use Elf::Reader;

# section attrs SHT_XXX
my %g_sht;

sub read_sht
{
  my $fh;
  if ( !open($fh, '<', "../sht.txt") ) {
    printf("cannot open sht.txt\n");
    return 0;
  }
  my $str;
  while( $str = <$fh> ) {
    chomp $str;
    if ( $str =~ /^([0-9a-f]+)\s+(.*)$/i ) {
      $g_sht{ hex($1) } = $2;
    }
  }
  close $fh;
  return 1;
}

# sm version
my %g_smver;

sub read_smver
{
   my $fh;
  if ( !open($fh, '<', "../sm_version.txt") ) {
    printf("cannot open sm_version.txt\n");
    return 0;
  }
  my $str;
  while( $str = <$fh> ) {
    chomp $str;
    if ( $str =~ /^([0-9a-f]+)\s+(.*)$/i ) {
      $g_smver{ hex($1) } = $2;
    }
  }
  close $fh;
  return 1;
}

sub dump_elf($$)
{
  my($e, $fname) = @_;
  # check elf machine
  if ( 190 != $e->machine() ) {
    printf("%s is not cubin\n", $fname);
    return 0;
  }
  # dump version
  my $ver = ($e->flags() >> 0x10) & 0xff;
  if ( exists $g_smver{$ver} ) {
    printf("version: %s\n", $g_smver{$ver});
  } else {
    printf("unkown version: %d\n", $ver);
  }
  # dump sections
  my $secs = $e->secs();
  printf("sections %d\n", scalar( @$secs ));
  foreach ( @$secs ) {
    my $sht_name = $g_sht{$_->[2]};
    if ( defined($sht_name) ) {
      printf("%s %s\n", $_->[1], $sht_name);
    }
  }
}

# main
read_sht(); read_smver();
foreach ( @ARGV ) {
 my $elf = Elf::Reader->new($_);
 dump_elf($elf, $_);
}