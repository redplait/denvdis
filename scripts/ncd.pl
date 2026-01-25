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
use vars qw/$opt_D $opt_e $opt_g $opt_r $opt_t $opt_v $opt_k/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] file.nvcudmp
 Options:
  -D drv-version
  -e - dump only threads with exception
  -g - dump grids
  -k - keep extracted file(s)
  -r - dump registers
  -t - dump threads
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

# args: section type, dev idx, ctx idx
sub grep_sec_ctx
{
  my($st, $didx, $cidx) = @_;
  my @res;
  foreach my $s ( @$g_s ) {
    next if ( $s->[2] != $st );
    # check section name at [1]
    next if ( $s->[1] !~ /dev(\d+)/ );
    next if ( int($1) != $didx );
    next if ( $s->[1] !~ /ctx(\d+)/ );
    next if ( int($1) != $cidx );
    push @res, $s->[0];
  }
  return scalar(@res) ? \@res: undef;
}

# boring sections selection logic. Hierarchy is
# 0 - device
# 1 - sm
# 2 - cta
# 3 - warp
# 4 - lane
# args: function/clojure to grep, section type
sub grep_sec_list
{
  my($cl, $st) = @_;
  my @res;
  foreach my $s ( @$g_s ) {
    next if ( $s->[2] != $st );
    # skip empty
    next unless ( $s->[9] );
    next unless ( $cl->($s) );
    if ( $s->[1] =~ /wp(\d+)/ ) {
      push @res, [ $s, int($1) ];
    } else {
      push @res, $s;
    }
  }
  return scalar(@res) ? \@res: undef;
}

# clojure factory for grep_sec_list
sub sm_filter
{
  my $ar = shift;
  my $res = sub {
    my $s = shift;
    # 0 - dev
    return 0 if ( $s->[1] !~ /dev(\d+)/ );
    return 0 if ( int($1) != $ar->[0] );
    # 1 - sm
    return 0 if ( $s->[1] !~ /sm(\d+)/ );
    return ( int($1) == $ar->[1] );
  };
  $res;
}

sub cta_filter
{
  my $ar = shift;
  my $res = sub {
    my $s = shift;
    # 0 - dev
    return 0 if ( $s->[1] !~ /dev(\d+)/ );
    return 0 if ( int($1) != $ar->[0] );
    # 1 - sm
    return 0 if ( $s->[1] !~ /sm(\d+)/ );
    return 0 if ( int($1) != $ar->[1] );
    # 2 - cta
    return 0 if ( $s->[1] !~ /cta(\d+)/ );
    return int($1) == $ar->[2];
  };
  $res;
}

# and for lane - reuse cta_filter to avoid copy-pasting
sub lane_filter
{
  my $ar = shift;
  my $cl = cta_filter($ar);
  my $res = sub {
    my $s = shift;
    return 0 unless ( $cl->($s) );
    # 3 - warp
    return 0 if ( $s->[1] !~ /wp(\d+)/ );
    return 0 if ( int($1) != $ar->[3] );
    # 4 - lane
    return 0 if ( $s->[1] !~ /ln(\d+)/ );
    return int($1) == $ar->[4];
  };
  $res;
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

# args: dev id, list of grid sections
sub dump_grids
{
  my($d_id, $glist) = @_;
  my $g_size = scalar(@$glist);
  return unless($g_size);
  for( my $i = 0; $i < $g_size; $i++ ) {
    my $grid = $g_elf->ncd_grid($glist->[$i], $opt_D);
    next unless defined($grid);
    printf("Dev %d: %d grids:\n", $d_id, scalar @$grid);
    foreach my $cg ( @$grid ) {
      printf(" GridID %X\n", $cg->[0]);
      printf(" context ID %X\n", $cg->[1]);
      printf(" function %X\n", $cg->[2]);
      printf(" functionEntry %X\n", $cg->[3]);
      printf(" module %X\n", $cg->[4]);
      printf(" parend GridID %X\n", $cg->[5]) if ( $cg->[5] );
      printf(" params off %X\n", $cg->[6]);
      printf(" kernel type %X\n", $cg->[7]);
      printf(" origin %X\n", $cg->[8]) if ( $cg->[8] );
      printf(" grid status %X\n", $cg->[9]) if ( $cg->[9] );
      printf(" num regs: %X\n", $cg->[10]);
      printf(" gridDim: X %X Y %X Z %X\n", $cg->[11], $cg->[12], $cg->[13]);
      printf(" blockDim: X %X Y %X Z %X\n", $cg->[14], $cg->[15], $cg->[16]);
      printf(" attrLaunchBlocking %X\n", $cg->[17]);
      printf(" attrHostId %X\n", $cg->[18]);
      next unless( defined $cg->[19]);
      printf(" clusterDim: X %X Y %X Z %X\n", $cg->[19], $cg->[20], $cg->[21]);
    }
  }
}

sub dump_ar
{
  my($pfx, $regs) = @_;
  return unless defined($regs);
  my $r_size = scalar(@$regs);
  return unless($r_size);
  printf("       %d %s:\n", $r_size, $pfx );
  printf("        [%d] %X\n", $_, $regs->[$_]) for( 0 .. $r_size - 1 );
}

# dump (u)regs/(u)preds for specific thread
# args: list of thread coordinates for lane_filter
sub dump_regs
{
  my $ar = shift;
  my $f = lane_filter($ar);
  # regs
  my $rlist = grep_sec_list($f, Elf::Reader::CUDBG_SHT_DEV_REGS);
  if ( defined($rlist) && 1 == scalar(@$rlist) ) {
    dump_ar("Regs", $g_elf->ncd_regs($rlist->[0]->[0]->[0]));
  }
  # preds
  my $plist = grep_sec_list($f, Elf::Reader::CUDBG_SHT_DEV_PRED);
  if ( defined($plist) && 1 == scalar(@$plist) ) {
    dump_ar("Preds", $g_elf->ncd_pred($plist->[0]->[0]->[0]));
  }
  my $urlist = grep_sec_list($f, Elf::Reader::CUDBG_SHT_DEV_UREGS);
  if ( defined($urlist) && 1 == scalar(@$urlist) ) {
    dump_ar("URegs", $g_elf->ncd_uregs($urlist->[0]->[0]->[0]));
  }
  my $uplist = grep_sec_list($f, Elf::Reader::CUDBG_SHT_DEV_UPRED);
  if ( defined($uplist) && 1 == scalar(@$uplist) ) {
    dump_ar("UPreds", $g_elf->ncd_upred($uplist->[0]->[0]->[0]));
  }
}

sub dump_uregs
{
  my $ar = shift;
  my $f = cta_filter($ar);
  my $urlist = grep_sec_list($f, Elf::Reader::CUDBG_SHT_DEV_UREGS);
  if ( defined $urlist ) {
    foreach my $r ( @$urlist ) {
      printf("    Warp %d:\n", $r->[1]);
      dump_ar("URegs", $g_elf->ncd_uregs($r->[0]->[0]));
    }
  }
  my $uplist = grep_sec_list($f, Elf::Reader::CUDBG_SHT_DEV_UPRED);
  if ( defined $uplist ) {
    foreach my $r ( @$uplist ) {
      printf("    Warp %d:\n", $r->[1]);
      dump_ar("UPreds", $g_elf->ncd_upred($r->[0]->[0]));
    }
  }
}

# dump lanes
# called only when -t option
# return fault PC if presents
sub dump_threads
{
  my $ar = shift;
  my $f = cta_filter($ar);
  my $res;
  my $tlist = grep_sec_list($f, Elf::Reader::CUDBG_SHT_LN_TABLE);
  return unless( defined $tlist );
  my $latch = 0;
  foreach my $l ( @$tlist ) {
    # here l is pair [ section, warp ID ]
    my $t = $g_elf->ncd_lanes($l->[0]->[0], $opt_D);
    next unless( defined $t );
    # store warp
    $ar->[3] = $l->[1];
    foreach my $ct ( @$t ) {
      # store lane id
      $ar->[4] = $ct->[2];
      next if ( defined($opt_e) && !$ct->[6] );
      if ( !$latch ) {
        $latch++;
        printf("   Lanes for warp %d:\n", $l->[1]);
      }
      printf("    Ln %d\n", $ct->[2]);
      # dump remained fields
      printf("     virtualPC: %X\n", $ct->[0]); $res = $ct->[0];
      printf("     physPC: %X\n", $ct->[1]);
      printf("     threadIdx: X %X Y %X Z %X\n", $ct->[3], $ct->[4], $ct->[5]);
      printf("     exception %X\n", $ct->[6]) if ( $ct->[6] );
      printf("     callDepth: %d\n", $ct->[7]) if ( $ct->[7] );
      printf("     syscallDepth: %d\n", $ct->[8]) if ( $ct->[8] );
      printf("     ccRegister: %X\n", $ct->[9]) if ( $ct->[9] );
      printf("     threadState: %X\n", $ct->[10]) if ( defined $ct->[10] );
      next unless( defined $opt_r );
      dump_regs($ar);
    }
  }
  $res;
}

# args: list of cta sections, filter
# called only when -t option
sub dump_cta
{
  my($clist, $ar) = @_;
  my $c_size = scalar(@$clist);
  return unless($c_size);
  for ( my $i = 0; $i < $c_size; $i++ ) {
    my $ctas = $g_elf->ncd_cta($clist->[$i]->[0], $opt_D);
    next unless defined($ctas);
    my $cta_size = scalar @$ctas;
    for ( my $j = 0; $j < $cta_size; $j++ ) {
      $ar->[2] = $j; # store cta num in filter
      printf(" CTA %d:\n", $j);
      my $c = $ctas->[$j];
      printf("  grid ID: %X\n", $c->[0]);
      printf("  blockIdx   X %X Y %X Z %X\n", $c->[1], $c->[2], $c->[3]);
      printf("  clusterIdx X %X Y %X Z %X\n", $c->[4], $c->[5], $c->[6]) if defined($c->[4]);
      my $res = dump_threads($ar);
      dump_uregs($ar) if ( defined($res) && defined($opt_r) );
    }
  }
}

# args: list of SM_Tab sections indexes
# return errorPC from SMTableEntries if presents
# problem with this dirty hack is that SM is not bound with context so we need to make full scan in analyse_mods
sub check_sm
{
  my($d_id, $sl) = @_;
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
    if ( defined $opt_t ) {
      # dump cta for each SM in $sm list
      for ( my $j = 0; $j < scalar(@$sm); $j++ ) {
        my @ar = ( $d_id, $j );
        my $f = sm_filter(\@ar);
        my $clist = grep_sec_list($f, Elf::Reader::CUDBG_SHT_CTA_TABLE);
        next unless( defined $clist );
# printf("CTA found %d size %d\n", $i, scalar(@$clist));
        dump_cta($clist, \@ar);
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

# check if extracted ELF contains $e_pc
sub check_elf
{
  my($fname, $e_pc) = @_;
  my $elf = Elf::Reader->new($fname);
  unless( defined $elf ) {
    carp("check_elf: cannot read $fname");
    return 0;
  }
  # quick check
  return 0 unless( $elf->in_elf($e_pc) );
  # yep, it's our precious-s-s
  my $slist = $elf->secs();
  unless ( defined $slist ) {
    carp("check_elf: cannot read sections from $fname");
    return 0;
  }
  printf("in %s\n", $fname);
  my $res_s;
  # enum sections
  foreach my $s ( @$slist ) {
    next unless($s->[8]); # no address
    next unless($s->[9]); # no size
    next if ($s->[2] != SHT_PROGBITS); # section type
    printf("[%d] %s addr %X size %X\n", $s->[0], $s->[1], $s->[8], $s->[9]) if ( defined $opt_v );
    # finally check if address inside this section
    next if ( $e_pc < $s->[8] );
    next if ( $e_pc >= ($s->[8] + $s->[9]) );
    $res_s = $s;
  }
  unless( defined $res_s ) {
    carp("cannot find target section in $fname");
    return 0;
  }
  printf("Addr %X: section %s off %X\n", $e_pc, $res_s->[1], $e_pc - $res_s->[8]);
  return 1;
}

# args: dev_id, context_id, error pc
sub dump_mods
{
  my($d_id, $ctx_id, $e_pc) = @_;
  my $mlist = grep_sec_ctx(Elf::Reader::CUDBG_SHT_RELF_IMG, $d_id, $ctx_id);
  unless( defined $mlist ) {
    carp("cannot extract rel elf images for dev $d_id ctx $ctx_id");
    return;
  }
  my $m_len = scalar @$mlist;
  foreach my $i ( 0 .. $m_len - 1 ) {
    my $fname = sprintf("eld%d.dev%d.ctx%d", $i, $d_id, $ctx_id);
    my $fh;
    open($fh, '>', $fname) or die("cannot create $fname");
    binmode($fh);
    $g_elf->save2fd($mlist->[$i], $fh);
    close $fh;
    my $res = check_elf($fname, $e_pc);
    unlink($fname) if ( !$res && !defined($opt_k) );
  }
}

# args: dev index, error PC
sub analyse_mods
{
  my($d_id, $e_pc) = @_;
  # read contexts
  my $cts_s = grep_sec(Elf::Reader::CUDBG_SHT_CTX_TABLE, $d_id);
  unless( defined $cts_s ) {
    printf("Cannot get ctx for dev %d\n", $d_id);
    return;
  }
  my $ctx_len = scalar @$cts_s;
  foreach my $si ( 0 .. $ctx_len - 1 ) {
    my $ctxs = $g_elf->ncd_ctx($cts_s->[$si]);
    next unless(defined $ctxs);
    my $c_len = scalar @$ctxs;
    foreach my $m ( 0 .. $c_len - 1 ) {
      printf(" ctx %d:\n", $m);
      my $c = $ctxs->[$m];
      if ( defined $opt_v ) {
        printf("   ctx ID: %X\n", $c->[0]);
        printf("   shared %X\n", $c->[1]);
        printf("   local %X\n", $c->[2]);
        printf("   global %X\n", $c->[3]);
        printf("   dev_id %d tid %X\n", $c->[4], $c->[5]);
      }
    }
    # read mods table
    my $mod_s = grep_sec_ctx(Elf::Reader::CUDBG_SHT_MOD_TABLE, $d_id, $si);
    unless( defined $mod_s ) {
      printf("Cannot get modules for dev %d ctx %d\n", $d_id, $si);
      next;
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
    # dump elf mods
    dump_mods($d_id, $si, $e_pc);
  }
}

# main
my $state = getopts("egrtvkD:");
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
  if ( defined $opt_g ) {
    my $glist = grep_sec(Elf::Reader::CUDBG_SHT_GRID_TABLE, $i);
    dump_grids($i, $glist) if ( defined $glist );
  }
  my $slist = grep_sec(Elf::Reader::CUDBG_SHT_SM_TABLE, $i);
  next unless defined($slist);
  $fault_addr = check_sm($i, $slist);
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
