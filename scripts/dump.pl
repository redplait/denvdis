#!perl -w
# simple cubin dumper, based on my Elf::Reader https://github.com/redplait/dwarfdump/tree/main/perl/Elf-Reader
use strict;
use warnings;
use Elf::Reader;

my %g_sm = (
 0xa => 'sm10',
 0xb => 'sm11',
 0xc => 'sm12',
 0xd => 'sm13',
0x14 => 'sm20',
0x15 => 'sm21',
0x1E => 'sm30',
0x20 => 'sm32',
0x23 => 'sm35',
0x25 => 'sm37',
0x32 => 'sm5',
0x34 => 'sm52',
0x35 => 'sm52',
0x3c => 'sm55',
0x3d => 'sm57',
0x3e => 'sm57',
0x46 => 'sm70',
0x48 => 'sm72',
0x4b => 'sm75',
0x50 => 'sm80',
0x56 => 'sm86',
0x57 => 'sm87',
0x59 => 'sm89',
0x5a => 'sm90',
0x64 => 'sm100',
0x65 => 'sm101',
0x78 => 'sm120'
);

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
    printf("md %s\n", $g_sm{$ver}) if ( exists $g_sm{$ver} );
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