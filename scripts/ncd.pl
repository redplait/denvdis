#!perl -w
# script to dump nvidia code dumps
use strict;
use warnings;
use Elf::Reader;
use Cubin::Ced;
use Getopt::Std;
use Carp;
use Data::Dumper;

# options
use vars qw/$opt_D $opt_v $opt_k/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] file.nvcudmp
 Options:
  -D drv-version
  -k - keep extracted file(s)
  -v - verbose mode
EOF
  exit(8);
}

# globals
my ($g_strtab, $g_elf, $g_s);
$opt_D = Elf::Reader::DRV_VERSION;

# read string from strtab
sub get_str
{
  my $off = shift;
  return unless defined($g_strtab);
  $g_elf->strz($g_strtab, $off);
}

sub dump_str
{
  my($off, $pfx) = @_;
  my $s = get_str($off);
  return unless defined($s);
  printf($pfx, $s);
}

# args: section type, dev idx
sub grep_sec
{
  my($st, $didx) = @_;
  my @res;
  foreach my $s ( @$g_s ) {
    next if ( $s->[2] != $st );
    # check section name at [1]
    next if ( $s->[1] !~ /dev(\d+)/ );
    next if ( int($1) != $didx );
    push @res, $s->[0];
  }
  return scalar(@res) ? \@res: undef;
}

# read sections and find g_strtab, storing list of section into $g_s
# returns dev list section
sub read_sections
{
  $g_s = $g_elf->secs();
  return unless defined($g_s);
  my $res;
  foreach my $s ( @$g_s ) {
    $g_strtab = $s->[0] if ( $s->[2] == SHT_STRTAB );
    $res = $s->[0] if ( $s->[2] == Elf::Reader::CUDBG_SHT_DEV_TABLE );
    last if ( defined($res) && defined($g_strtab) );
  }
  $res;
}

sub parse_dev_tab
{
  my $idx = shift;
  my $ar = $g_elf->ncd_dev($idx, $opt_D);
  my $ar_size = scalar @$ar;
  foreach my $i ( 0 .. $ar_size - 1 ) {
    printf("device %d:\n", $i);
    my $d = $ar->[$i];
    dump_str($d->[0], " Name: %s\n");
    dump_str($d->[1], " Type: %s\n");
    dump_str($d->[2], " SM: %s\n");
    if ( defined $opt_v ) {
      printf(" major %d minor %d\n", $d->[11], $d->[12]);
      printf(" instr size: %d\n", $d->[13]);
      printf(" status: %d\n", $d->[14]);
      printf(" SM nums: %d\n", $d->[6]);
      printf(" Warps per SM: %d\n", $d->[7]);
      printf(" Lanes per Warp: %d\n", $d->[8]);
      printf(" Regs per lane: %d\n", $d->[9]);
      printf(" Preds per lane: %d\n", $d->[10]);
      printf(" URegs per lane: %d\n", $d->[15]) if ( $d->[15] );
      printf(" UPreds per lane: %d\n", $d->[16]) if ( $d->[16] );
      printf(" Barriers per Warp: %d\n", $d->[17]) if ( $d->[17] );
    }
  }
  $ar_size;
}

# args: list of SM_Tab sections indexes
# return errorPC from SMTableEntries if presents
sub check_sm
{
  my $sl = shift;
  my $sl_size = scalar(@$sl);
  my $res;
  return unless($sl_size);
  for( my $i = 0; $i < $sl_size; $i++ ) {
    my $sm = $g_elf->ncd_sm($sl->[$i], $opt_D);
    next unless defined($sm);
    printf("SM %d:\n", $i) if defined($opt_v);
    # check what we have
    if ( 'ARRAY' eq ref $sm->[0] ) {
      foreach my $csm ( @$sm ) {
        $res = $csm->[3] if ( !defined($res) && defined($csm->[3]) );
        # dump SM details
        if ( defined $opt_v ) {
          printf(" ID %d\n", $csm->[0]);
          printf(" exception: %d\n", $csm->[1]);
          dump_str($csm->[8], "  %s\n") if defined($csm->[8]);
          printf(" errorPC: %X\n", $csm->[3]) if defined($csm->[3]);
        }
      }
    } else {
      # old useless SM format with id only
      if ( defined $opt_v ) {
        printf(" %d\n", $_) for @$sm;
      }
    }
  }
  $res;
}

# almost like check_sm
# args: list of Warp_Tab sections indexes
# return errorPC from WarpTableEntries if presents
sub check_warp
{
  my $wl = shift;
  my $wl_size = scalar(@$wl);
  my $res;
  return unless($wl_size);
  for( my $i = 0; $i < $wl_size; $i++ ) {
    my $wt = $g_elf->ncd_wp($wl->[$i], $opt_D);
    next unless defined($wt);
    printf("Warp %d:\n", $i) if defined($opt_v);
    foreach my $warp ( @$wt ) {
      $res = $warp->[0] if ( !defined($res) && defined($warp->[0]) );
      if ( defined $opt_v ) {
        printf(" warp ID: %d\n", $warp->[1]);
        printf(" valid lanes mask: %X\n", $warp->[2]);
        printf(" active lanes mask: %X\n", $warp->[3]);
        printf(" warp broken: %d\n", $warp->[4]) if ( $warp->[4] );
        printf(" errorPC: %X\n", $warp->[0]) if defined($warp->[0]);
      }
    }
  }
  $res;
}

# args: dev index, error PC
sub analyse_mods
{
  my($d_id, $e_pc) = @_;
  # read contexts
  my $cts_s = grep_sec(Elf::Reader::CUDBG_SHT_CTX_TABLE, $d_id);
  unless( defined $cts_s ) {
    printf("Cannot get ctas for dev %d\n", $d_id);
    return;
  }
  my @tmp;
  foreach my $si ( @$cts_s ) {
    my $ctxs = $g_elf->ncd_ctx($si);
    next unless(defined $ctxs);
    my $c_len = scalar @$ctxs;
    foreach my $m ( 0 .. $c_len - 1 ) {
      printf(" ctx %d:\n", $m);
      my $c = $ctxs->[$m];
      push @tmp, $c;
      if ( defined $opt_v ) {
        printf("   ctx ID: %X\n", $c->[0]);
        printf("   shared %X\n", $c->[1]);
        printf("   local %X\n", $c->[2]);
        printf("   global %X\n", $c->[3]);
        printf("   dev_id %d tid %X\n", $c->[4], $c->[5]);
      }
    }
  }
  # read mods
  my $mod_s = grep_sec(Elf::Reader::CUDBG_SHT_MOD_TABLE, $d_id);
  unless( defined $mod_s ) {
    printf("Cannot get modules for dev %d\n", $d_id);
    return;
  }
  foreach my $ms ( @$mod_s ) {
    my $mt = $g_elf->ncd_mods($ms);
    next unless(defined $mt);
    my $m_len = scalar @$mt;
    foreach my $m ( 0 .. $m_len - 1 ) {
      my $mod = $mt->[$m];
      printf(" [%d] mod %X\n", $m, $mod);
    }
  }
}

# main
my $state = getopts("vkD:");
usage() if ( !$state );
if ( -1 == $#ARGV ) {
  printf("where is arg?\n");
  exit(5);
}
# open elf file
$g_elf = Elf::Reader->new($ARGV[0]);
die("cannot open $ARGV[0]") unless defined($g_elf);
die("$ARGV[0] not nvcudmp") unless $g_elf->is_ncd();
my $dev_idx = read_sections();
die("cannot find dev table in $ARGV[0]") unless defined($dev_idx);
my $devs = parse_dev_tab($dev_idx);
# try to find SMs for each device
my($fault_addr, $fault_dev_idx);
for my $i ( 0 .. $devs - 1 ) {
  my $slist = grep_sec(Elf::Reader::CUDBG_SHT_SM_TABLE, $i);
  next unless defined($slist);
  $fault_addr = check_sm($slist);
  if ( defined($fault_addr) ) { $fault_dev_idx = $i; last; }
}
unless( defined $fault_addr ) {
 # then try Warps
 for my $i ( 0 .. $devs - 1 ) {
   my $wlist = grep_sec(Elf::Reader::CUDBG_SHT_WP_TABLE, $i);
   next unless defined($wlist);
   $fault_addr = check_warp($wlist);
   if ( defined($fault_addr) ) { $fault_dev_idx = $i; last; }
 }
}
die("cannot find fault address") unless(defined $fault_addr);
analyse_mods($fault_dev_idx, $fault_addr);
