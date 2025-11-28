#!perl -w
# Sample of using perl modules from https://redplait.blogspot.com/2025/10/perl-modules-for-cubins-patching.html
use strict;
use warnings;
use Elf::Reader;
use Cubin::Ced;
use Cubin::Attrs;
use Getopt::Std;
use Carp;
use Data::Dumper;

# options
use vars qw/$opt_b $opt_d $opt_g $opt_l $opt_p $opt_r $opt_s $opt_t $opt_u $opt_U $opt_v/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] file.cubin
 Options:
  -b - track read/write barriers
  -d - debug mode
  -g - build cfg
  -l - dump latency info
  -p - dump properties
  -r - dump relocs
  -s - try to find instructions to swap and reduce stall count
  -t - track registers
  -u - try detect register reuse cache
  -U - analyze possible registers reuse
  -v - verbose mode
EOF
  exit(8);
}

# globals
my($g_elf, $g_attrs, $g_ced, $g_syms, $g_w);
# stat for barriers, key is ins name, value is [ wait, read, write ] count
my %g_barstat;
# per code section globals
# syms inside section, curr_index and cached symbols
my(@gs_syms, $gs_cidx, $g_afsyms);
# relocs
my($gs_rel, $gs_rela);
# cb params
my(@gs_cbs, $gs_cb_size, $gs_cb_off);
# labels from attrs
my($gs_loffs, $gs_ibt);
# for -u
my $gu_max = 0;
my($gu_off, %gu_cache);
# for -U - reuse stat
my $gU_found = 0; # total possible reuse count
my $gU_solved = 0; # actual reuse count
my $gU_dis_failed = 0; # times of disasm failures
my $gU_not_found = 0;  # cannot find reg or mask
# latency stat, 0 - total, 1 - missed predicted stall, 2 - bad stall, 3 - map of missed opcodes, 4 - map of bad stalls opcodes
my @gl_stat = ( 0, 0, 0, (), () );
my @gl_pcols_stat = ( 0, 0, 0, (), () );
my @gl_prows_stat = ( 0, 0, 0, (), () );
# reordering stat - total amount of instructions, swappable pairs count, total stall gain
my($gs_total, $gs_ords, $gs_gain, $gs_old_stall);

sub dump_swap_stat
{
  return unless($gs_ords);
  printf("Reordering stat: total %d swappable %d (%f)\n", $gs_total, $gs_ords, $gs_ords * 1.0 / $gs_total);
  printf(" total gain %d (%f avg) gain/old ratio %f \n", $gs_gain, $gs_gain * 1.0 / $gs_ords, $gs_gain * 1.0 / $gs_old_stall);
}

# args: stall gain, total old stall in pair
sub upd_swap_stat
{
  my($gain, $old) = @_;
  $gs_ords++;
  $gs_gain += $gain;
  $gs_old_stall += $old;
}

# arg - one of 3 latency stat, header
sub dump_lstat
{
  my($gls, $hdr) = @_;
  return unless($gls->[0]);
  printf("; -- %s:\n", $hdr) if defined($hdr);
  printf("Missed latency: %d (%f)\n", $gls->[1], $gls->[1] * 1.0 / $gls->[0] ) if ( $gls->[1] );
  # dump missed opcodes
  my $missed = $gls->[3];
  if ( defined($opt_v) && defined($missed) ) {
    my @m = sort { $missed->{$b} <=> $missed->{$a} } keys %$missed;
    if ( scalar @m ) {
      printf(" missed stat:\n");
      printf("  %d - %s\n", $missed->{$_}, $_) for @m;
    }
  }
  printf("Mismatched latency: %d (%f)\n", $gls->[2], $gls->[2] * 1.0 / $gls->[0] ) if ( $gls->[2] );
  my $bad = $gls->[4];
  if ( defined($opt_v) && defined($bad) ) {
    my @m = sort { $bad->{$b} <=> $bad->{$a} } keys %$bad;
    if ( scalar @m ) {
      printf(" bad stalls stat:\n");
      printf("  %d - %s\n", $bad->{$_}, $_) for @m;
    }
  }
}

sub dump_lat_stat
{
  dump_lstat(\@gl_stat, 'curr columns with curr rows');
  dump_lstat(\@gl_pcols_stat, 'prev columns with curr rows');
  dump_lstat(\@gl_prows_stat, 'curr columns with prev rows');
}

sub dump_ruc
{
  return unless($gu_max);
  printf("max RUC:%d at %X:\n", $gu_max, $gu_off);
  printf(" %s%d\n", $_ & 0x8000 ? 'UR' : 'R', $_ & ~0x8000) for ( keys %gu_cache );
}

sub dump_rU
{
  return unless($gU_found);
  printf("Found %d reuse cases, solved %d, not found %d, dis fails %d\n",
    $gU_found, $gU_solved, $gU_not_found, $gU_dis_failed);
}

# args: block, off, regs from snap
sub add_ruc
{
  my($block, $off, $rs) = @_;
  return unless defined($rs);
  my $added = 0;
  while( my($r, $flag) = each(%$rs) ) {
    if ( $flag & 0x40 ) {
      $block->[10]->{$r} = $off;
      $added++;
    } else {
      delete $block->[10]->{$r};
    }
  }
  if ( $added ) {
    my $cur_size = scalar keys %{ $block->[10] };
    if ( $cur_size > $gu_max ) {
      $gu_max = $cur_size;
      $gu_off = $off;
      # copy RUC from block to gu_cache
      %gu_cache = %{ $block->[10] };
    }
  }
  $added;
}

sub sym_reset { $gs_cidx = 0; }

sub sym_name
{
  my $sidx = shift;
  return unless defined($g_syms);
  return unless defined($g_syms->[$sidx]);
  $g_syms->[$sidx]->[0];
}

sub sym
{
  my $sidx = shift;
  return unless defined($g_syms);
  $g_syms->[$sidx];
}

sub setup_syms
{
  my $sidx = shift;
  $gs_cidx = 0;
  return unless defined($g_afsyms);
  @gs_syms = sort { $a->[1] <=> $b->[1] }
   # grep named with right section at [5]
   grep { $_->[5] == $sidx } @$g_afsyms;
  if ( defined $opt_v ) {
    printf(" %d symbols:\n", scalar @gs_syms);
    foreach my $s ( @gs_syms ) {
      printf("  %X type %x size %x %s\n", $s->[1], $s->[4], $s->[2], $s->[0]);
    }
  }
}

sub dump_sym_cmn
{
  my $sym = shift;
  # global?
  printf("\t.global %s\n", $sym->[0]) if ( STB_GLOBAL == $sym->[3] );
  # size
  printf("\t.size %X\n", $sym->[2]) if ( $sym->[2] );
  # dump name label
  printf("%s:\n", $sym->[0]);
}

sub check_sym
{
  my $off = shift;
  my $res = 0;
  return if ( $gs_cidx >= scalar(@gs_syms) );
  while ( $gs_syms[$gs_cidx]->[1] <= $off ) {
    dump_sym_cmn($gs_syms[$gs_cidx]);
    $res++;
    last if ( ++$gs_cidx >= scalar(@gs_syms) );
  }
  $res;
}

# skip all symbols below boff, dump symbols up to off
sub head_syms
{
  my( $block_off, $off ) = @_;
  return if ( $gs_cidx >= scalar(@gs_syms) );
  # skip till block_off
  while ( $gs_syms[$gs_cidx]->[1] < $block_off ) {
    return if ( ++$gs_cidx >= scalar(@gs_syms) );
  }
  return check_sym($off);
}

# put first found symbol to br[off+2]
sub gcheck_sym
{
  my($br, $off) = @_;
  my $latch = 0;
  return if ( $gs_cidx >= scalar(@gs_syms) );
  while ( $gs_syms[$gs_cidx]->[1] <= $off ) {
    unless ( $latch ) {
      $br->{$off+2} = $gs_cidx;
      $latch++;
    }
    last if ( ++$gs_cidx >= scalar(@gs_syms) );
  }
  return $latch;
}

# lame binary search in compound array
# args: ref to array, index of field to compare, target
sub bin_sa
{
  my($ar, $idx, $what) = @_;
  my $low = 0;
  my $high = scalar @$ar; # Index of the last element
  while ($low < $high) {
     my $mid = int(($low + $high) / 2); # Calculate the middle index
     if ($ar->[$mid]->[$idx] == $what) {
        return wantarray ? ($mid, $ar->[$mid]) : $mid; # Target found
     } elsif ($ar->[$mid]->[$idx] < $what) {
        $low = $mid + 1; # Target is in the upper half
     } else {
        $high = $mid - 1; # Target is in the lower half
     }
  }
  undef;
}

# args: ref to array, sub with <=> like return, target
sub bin_sac
{
  my($ar, $cb, $what) = @_;
  my $low = 0;
  my $high = scalar @$ar; # Index of the last element
  while ($low < $high) {
     my $mid = int(($low + $high) / 2); # Calculate the middle index
     my $res = $cb->($ar->[$mid], $what);
     if (!$res) {
        return wantarray ? ($mid, $ar->[$mid]) : $mid; # Target found
     } elsif ($res < 0) {
        $low = $mid + 1; # Target is in the upper half
     } else {
        $high = $mid - 1; # Target is in the lower half
     }
  }
  undef;
}

sub dump_ext
{
  my $ext_idx = shift;
  my $exts = $g_attrs->value($ext_idx->{'id'});
  return unless defined($exts);
  printf(" %d externals:\n", scalar @$exts);
  foreach my $i ( @$exts ) {
    printf("  %d", $i);
    my $sn = sym_name($i);
    if ( defined $sn ) { printf(" %s\n", $sn); }
    else { printf("\n"); }
  }
}

sub dump_cparams
{
  $gs_cb_off = $g_attrs->cb_off();
  $gs_cb_size = $g_attrs->cb_size();
  my $cnt = $g_attrs->params_cnt();
  @gs_cbs = ();
  return unless $cnt;
  printf(" %d CParams off %X:\n", $cnt, $gs_cb_off);
  my @tmp;
  for ( my $ci = 0; $ci < $cnt; ++$ci ) {
    my $c = $g_attrs->param($ci);
    next unless defined($c);
    printf("  [%d] ord %d off %X size %X\n", $ci, $c->{'ord'}, $c->{'off'}, $c->{'size'});
    $c->{'off'} += $gs_cb_off;
    push @tmp, [ $c->{'off'}, $c ];
  }
  @gs_cbs = sort { $a->[0] <=> $b->[0] } @tmp;
}

# return ref to cparam hash at index $off or undef
sub find_cparam($)
{
  my $off = shift;
  return unless defined($gs_cb_off);
  return unless scalar(@gs_cbs);
  return if ( $off < $gs_cb_off );
  foreach my $c ( @gs_cbs ) {
    return $c->[1] if ( $off >= $c->[0] && $off < ($c->[0] + $c->[1]->{'size'}) );
  }
  undef;
}

sub dump_rels
{
  my($pfx, $s_idx, $r_idx) = @_;
  my %rels;
  my $is_a = $pfx =~ /A$/;
  my $rsub = $is_a ? \&read_rela : \&read_rel;
  my $res = $rsub->($g_attrs, $g_elf, $s_idx, \%rels);
  if ( $is_a ) {
    $gs_rela = $res ? \%rels: undef;
  } else {
    $gs_rel = $res ? \%rels: undef;
  }
  return if ( !$res );
  printf(" %s %d: %d\n", $pfx, $r_idx, $res);
  foreach my $r ( sort { $a <=> $b } keys %rels ) {
    printf("  %X [%d] type %d", $r, $rels{$r}->[0], $rels{$r}->[2]);
    my $rname = $g_ced->reloc_name($rels{$r}->[2]);
    printf(" %s", $rname) if ( defined $rname );
    $rname = sym_name($rels{$r}->[1]);
    if ( defined $rname ) { printf(" %s", $rname); }
    # addend
    if ( $is_a && $rels{$r}->[3] ) {
      printf(" add %X", $rels{$r}->[3]);
    }
    printf("\n");
  }
}

# check if some offset has reloc
# returns (reloc, is_rela)
sub has_rel($)
{
  my $off = shift;
  if ( defined $gs_rel ) {
    return ($gs_rel->{$off}, 0) if exists($gs_rel->{$off});
  }
  if ( defined $gs_rela ) {
    return ($gs_rela->{$off}, 1) if exists($gs_rela->{$off});
  }
  (undef, 0);
}

sub get_ins_cb0
{
  my $res = $g_ced->ins_cbank();
  return unless defined($res);
  return if ( $res->[0] );
  return unless defined($res->[1]);
  $res->[1];
}

# args: cols from l2map, rows from l2map, header
sub intersect_lat
{
  my($cs, $rs, $hdr) = @_;
  return if ( !defined($cs) || !defined($rs) );
  my $res;
  # clojure to assign max latency
  my $asmax = sub {
    my $what = shift;
    unless( defined($res) ) {
      $res = $what;
      return;
    }
    $res = $what if ( $what > $res );
  };
  my $latch = 0;
  foreach my $tab ( sort { $a cmp $b } keys %$cs ) {
    next unless exists($rs->{$tab});
    # extract value from col and row
    my $m_y = $cs->{$tab};
    my $their = $rs->{$tab};
    printf("; -- %s:\n", $hdr) if ( !$latch++ );
    printf(";    tab %s (%s line %d):\n", $m_y->[0]->tab_name(), $m_y->[0]->conn_name(), $m_y->[0]->line());
    foreach my $c ( @$m_y ) {
      foreach my $r ( @$their ) {
        my $what = $c->at($r);
        $asmax->($what);
        printf(";      col %d (%s) row %d (%s): %d\n", $c->idx(), $c->name(), $r->idx(), $r->name(), $what);
      }
    }
  }
  $res;
}

# update latency stat
# args: current predicted stall (or undef), sched ctx to extract 'curr' stall, ref to lstat data
sub update_lstat($$$)
{
  my($pred, $sctx, $gl) = @_;
  $gl->[0]++;
  unless( defined $pred ) {
    $gl->[1]++;
    $gl->[3]->{$g_ced->ins_name()}++;
    printf("; [!] Missed latency\n");
    return;
  }
  return unless defined($sctx);
  my $diff = abs( $pred - $sctx->{'curr'});
  if ( $diff > 1 ) {
    $gl->[2]++;
    printf("; [!] Mismatched latency\n");
    $gl->[4]->{$g_ced->ins_name()}++;
    return;
  }
}

# dump latency tables
sub dump_latmap
{
  my($t, $pfx) = @_;
  foreach my $tab ( sort { $a cmp $b } keys %$t ) {
    my $ar = $t->{$tab};
    my $ar_size = scalar @$ar;
    next unless($ar_size); # empty?
    if ( 1 == $ar_size ) {
      my $single = $ar->[0];
      printf(";  table %s (%s line %d) %s index %d (%s)\n", $single->tab_name(), $single->conn_name(), $single->line(),
       $pfx, $single->idx(), $single->name());
    } else {
      printf(";  table %s (%s line %d) %d %ss:\n", $ar->[0]->tab_name(), $ar->[0]->conn_name(), $ar->[0]->line(), $ar_size, $pfx);
      printf(";   %d (%s)\n", $_->idx(), $_->name()) for ( @$ar );
    }
  }
}

# args: block, sched ctx
sub dump_lat
{
  my($block, $sctx) = @_;
  # columns
  my $cols = $g_ced->lcols();
  my $l2cols;
  if ( defined $cols ) {
    printf("; %d latency tab columns\n", scalar @$cols);
    $l2cols = l2map($cols);
    dump_latmap($l2cols, 'col');
  }
  # rows
  my $rows = $g_ced->lrows();
  my $l2rows;
  if ( defined $rows ) {
    printf("; %d latency tab rows\n", scalar @$rows);
    $l2rows = l2map($rows);
    dump_latmap($l2rows, 'row');
  }
  # brute force 3 variants
  # 1) intersect with itself
  update_lstat( intersect_lat($l2cols, $l2rows, 'self cols with self rows'), $sctx, \@gl_stat ) if ( defined($l2cols) && defined($l2rows) );
  if ( defined $block ) {
    # 2) intersect current columns with rows from previous instruction
    update_lstat( intersect_lat($l2cols, $block->[12], 'current cols with previous rows'), $sctx, \@gl_prows_stat) if defined($block->[12]);
    # 3) intersect columns from previous instruction with current rows
    update_lstat( intersect_lat($block->[11], $l2rows, 'previous cols with current rows'), $sctx, \@gl_pcols_stat) if defined($block->[11]);
    # store
    $block->[11] = $l2cols;
    $block->[12] = $l2rows;
  }
}

# scheduler context
# for old 64/88 bit SM has 'dual' field
# for -b option this is just map where key is barrier index and value is [ offset, R/W ]
# current stall rolling sum stored in 'roll' field
# current stall count in field 'curr', previous in 'prev'
# for new 88/128 bit stall count for barriers insts map in field 'c', key is offset
sub make_sctx
{
  my %res;
  $res{'dual'} = 0 if ( $g_w < 128 );
  $res{'roll'} = 0;
  if ( $g_w > 64 ) {
    my %c;
    $res{'c'} = \%c;
  }
  \%res;
}

# get current dual state and update it for next instruction
sub get_dual
{
  my $ctx = shift;
  return 0 if ( $g_w == 128 );
  return 0 if ( !$ctx->{'dual'} );
  return 1 if ( 1 == $ctx->{'dual'}++ );
  $ctx->{'dual'} = 0;
  2;
}

sub get_spfx
{
  my $dual = shift;
  return ' ' if ( $g_w == 128 );
  return '   ' if ( 1 != $dual );
  ' { ';
}

sub get_ssfx
{
  my $dual = shift;
  return ' ' if ( $g_w == 128 );
  return ' } ' if ( 2 == $dual );
  ' ';
}

sub add_barstat
{
  my($iname, $ar) = @_;
  if ( exists $g_barstat{$iname} ) {
    my $old = $g_barstat{$iname};
    $old->[$_] += $ar->[$_] for ( 0..2 );
  } else {
    $g_barstat{$iname} = $ar;
  }
}

sub dump_barstat
{
  while( my($name, $ar) = each(%g_barstat) ) {
    printf("%s:\t%d %d %d\n", $name, $ar->[0], $ar->[1], $ar->[2]);
  }
}

# check bitset of barriers to wait
sub check_wait
{
  my($watdb, $sctx, $curr_stall) = @_;
  my $res = 0;
  for my $i ( 0 .. 5 ) {
    my $mask = 1 << $i;
    if ( $watdb & $mask ) {
      $res++;
      if ( exists $sctx->{$i} ) {
        printf("; wait %d (%s) at %X", $i, $sctx->{$i}->[1], $sctx->{$i}->[0]);
        printf(" stall diff %d", $curr_stall - $sctx->{'c'}->{ $sctx->{$i}->[0] });
        printf("\n");
      }
      delete $sctx->{$i};
    }
  }
  $res;
}

# prepare sched data in block for next instruction - swap array from 4 to 6 and map from 5 to 7
sub sched_next
{
  my $b = shift;
  $b->[6] = $b->[4];
  $b->[4] = undef;
  $b->[7] = $b->[5];
  $b->[7] = undef;
}

# check if array at index ai waits some barriers map at index mi
sub sched_cross_check
{
  my($b, $ai, $mi, $what) = @_;
  return 0 if ( !defined($b->[$ai]) || !defined($b->[$mi]) );
  my $ar = $b->[$ai];
# print 'cross_check ', Dumper($b->[$ai], $b->[$mi]);
  for my $i ( 0..5 ) {
    next unless($ar->[$i]);
    return $what if exists($b->[$mi]->{$i});
  }
  0;
}

# check if two adjacent instructions share common barriers
sub sched_check
{
  my $b = shift;
  my $res = sched_cross_check($b, 4, 7, 1);
  $res = sched_cross_check($b, 6, 5, 2) unless $res;
  sched_next($b);
  $res;
}

# args: block, index, value
sub store_s_idx($$$)
{
  my($block, $idx, $v) = @_;
  return unless defined($block);
  return unless defined($opt_s);
  my $ar = $block->[13];
  $ar->[$idx] = $v;
  1;
}

# check if two adjacent instructions can be swapped
# Warning! this function should be called after checking that they are truly independent - it just estimate if they can be swapped
# args: block ref
# returns still gains
sub can_swap
{
  my $b = shift;
  return unless defined($b);
  return if ( !defined($b->[13]) || !defined($b->[14]) );
  my $curr = $b->[13];
  my $prev = $b->[14];
  # check if any of instructions
  # 1) dual
  return 0 if ( $curr->[8] || $prev->[8] );
  # 2) brt
  return 0 if ( $curr->[5] || $prev->[5] );
  # 3) has rela
  return 0 if ( $curr->[4] || $prev->[4] );
  # check cond
  return 0 if ( !defined($curr->[6]) || !defined($prev->[6]) );
  return 0 if ( $curr->[6] != $prev->[6] );
  # if they have cond - check that they are the same
  if ( $curr->[6] ) {
    return 0 if ( $curr->[2] !~ /^@(\!?\w+)/ );
    my $c_cond = $1;
    return 0 if ( $prev->[2] !~ /^@(\!?\w+)/ );
    return 0 unless ( $c_cond eq $1 );
  }
  # check if we can get some gain from swapping
  return $prev->[7] if ( $curr->[7] > $prev->[7] );
  0;
}

# extract old stall count for pair of adjacent instructions
# arg - block
sub get_old_pair_stall
{
  my $b = shift;
  $b->[13]->[7] + $b->[14]->[7];
}

# args: offset, sched ctx, block
sub process_sched
{
  my($off, $sctx, $b) = @_;
  my $ctrl;
  my $is_dual = 0;
  my $stall = 0;
  if ( $g_w == 64 ) {
    $ctrl = $g_ced->ctrl();
    # from Understanding the GPU Microarchitecture to Achieve Bare-Metal Performance Tuning:
    #  bits 0-3 indicate the number of stall cycles before issuing the next instruction.
    #  0x2n means a warp is suspended for n cycles before issuing the next instruction, where n = 0, 1, . . . , 15.
    #  0x20 means the single-issue mode, while 0x04 means the dual-issue mode
    $stall = $ctrl & 0x2f;
    $is_dual = $stall == 0x4;
    $stall &= 0xf; # bit 0..3
    store_s_idx($b, 7, $stall);
    store_s_idx($b, 8, $is_dual);
    if ( defined $opt_b ) {
      # previous & current stall
      $sctx->{'prev'} = $sctx->{'curr'};
      $sctx->{'curr'} = $stall;
      # update rolling stall count
      $sctx->{'roll'} += $stall;
      # dump current stall counts
      printf("; stall %d total %d ctrl %X\n", $stall, $sctx->{'roll'}, $ctrl);
    }
  } else {
    $ctrl = $g_ced->cword();
    # low 5 bits
    if ( $g_w == 88 ) {
      $is_dual = $g_ced->ins_dual();
      store_s_idx($b, 8, $is_dual) if ( $is_dual );
    }
    # render
    if ( defined $opt_b ) {
      my @stat = (0, 0, 0);
      my $curr_stall = $sctx->{'roll'};
      my $s = $g_ced->render_cword($ctrl);
      $stall = ($ctrl & 0x0000f) >> 0;
      store_s_idx($b, 7, $stall);
      # previous & current stall
      $sctx->{'prev'} = $sctx->{'curr'};
      $sctx->{'curr'} = $stall;
      # dump current stall counts
      printf("; stall %d total %d cword %X %s\n", $stall, $curr_stall, $ctrl, $s);
      # track barriers - ripped from maxas printCtrl
      my $wrtdb = ($ctrl & 0x000e0) >> 5;  # 3bit write dependency barrier
      my $readb = ($ctrl & 0x00700) >> 8;  # 3bit read  dependency barrier
      my $watdb = ($ctrl & 0x1f800) >> 11; # 6bit wait on dependency barrier
      # store sched data in block - array at index 4, map at 5
      if ( defined $b ) {
        my $ar;
        if ( $watdb ) {
          $ar = [ (0) x 6 ];
          for my $idx ( 0..5 ) {
            $ar->[$idx] = 1 if ( $watdb & (1 << $idx) );
          }
        }
        # store wait array
        $b->[4] = $ar;
        if ( $wrtdb != 7 || $readb != 7 ) {
          my %mb;
          $mb{$wrtdb} = 1 if ( $wrtdb != 7 );
          $mb{$readb} = 1 if ( $readb != 7 );
          $b->[5] = \%mb;
        } else { $b->[5] = undef; }
      }
      # check barriers - if watdb non-zero
      if ( $watdb ) {
        $stat[0] = 1;
        check_wait($watdb, $sctx, $curr_stall);
      }
      if ( $wrtdb != 7 ) { $sctx->{$wrtdb} = [ $off, 'W' ]; $stat[2] = 1; }
      if ( $readb != 7 ) { $sctx->{$readb} = [ $off, 'R' ]; $stat[1] = 1; }
      # debpar has barrier index in field sbidx & wait mask in in scoreboard_list (dep_scbd in SM5xx)
      my $sbidx = $g_ced->get('sbidx');
      if ( defined($sbidx) ) {
        $stat[0] = 1;
        if ( exists $sctx->{$sbidx} ) {
          printf("; sync %d (%s) at %X", $sbidx, $sctx->{$sbidx}->[1], $sctx->{$sbidx}->[0]);
          printf(" stall diff %d", $curr_stall - $sctx->{'c'}->{ $sctx->{$sbidx}->[0] });
          printf("\n");
        }
        delete $sctx->{$sbidx};
        # check bitset
        my $bm = $g_ced->get('scoreboard_list');
        $bm = $g_ced->get('dep_scbd') unless defined($bm);
        check_wait($bm, $sctx, $curr_stall) if ( $bm );
      }
      add_barstat($g_ced->ins_name(), \@stat) if ( $stat[0] or $stat[1] or $stat[2] );
      # update rolling stall count
      $sctx->{'roll'} += $stall unless $is_dual;
      # and store in 'c' if need
      $sctx->{'c'}->{$off} = $sctx->{'roll'} if ( $wrtdb != 7 || $readb != 7 );
    }
  }
  # store dual flag in sched ctx
# printf(" dual %d %d\n", $is_dual, $sctx->{'dual'});
  $sctx->{'dual'} = 1 if ( $is_dual && !$sctx->{'dual'} );
}

sub is_skip { $g_ced->ins_false() || 'NOP' eq $g_ced->ins_name(); }

# main horror - dump single instruction
# args: offset, sched context, block (or undef), reg track
# returns 0 if this instruction should be skipped
sub dump_ins
{
  my($off, $sctx, $block, $rt) = @_;
  my $brt = $g_ced->ins_brt();
  my $scbd = $g_ced->ins_scbd();
  my $mw = $g_ced->ins_min_wait();
  my $i_text = $g_ced->ins_text();
  if ( defined $opt_v ) {
    my $cl = $g_ced->ins_class();
    my $ln = $g_ced->ins_line();
    printf("; %s line %d", $cl, $ln);
    printf(" ALT") if ( $g_ced->ins_alt() );
    printf(" Brt %d (%s)", $brt, brt_name($brt)) if $brt;
    printf(" Scbd %d (%s)", $scbd, scbd_name($scbd)) if $scbd;
    printf(" min_wait: %d", $mw) if $mw;
    printf("\n");
  }
  # store data for -s
  if ( defined($block) && defined($opt_s) ) {
    my $ar = $block->[13];
    $ar->[0] = $off;
    $ar->[1] = $g_ced->ins_name();
    $ar->[2] = $i_text;
    $ar->[5] = $brt;
    $ar->[6] = $g_ced->has_pred();
  }
  # is empty instruction - nop or with !@PT predicate
  my $skip = is_skip();
  # check instr for label
  if ( !$skip ) { # && $brt != Cubin::Ced::BRT_RETURN ) {
    my($rel, $is_a) = has_rel($off);
    # store data for -s
    if ( defined($block) && defined($opt_s) && defined($rel) ) {
      my $ar = $block->[13];
      if ( $is_a ) { $ar->[4] = 1; }
      else { $ar->[3] = 1; }
    }
    # ignore instr having relocs
    unless($rel) {
      unless( defined $block ) {
        my $addl = $g_ced->ins_clabs();
        if ( defined($addl) ) {
          printf(" ; add label %X\n", $addl) if defined($opt_v);
          $gs_loffs->{$addl} = 0;
        }
      }
    }
    if ( defined($rel) && defined($opt_v) ) {
      my $f_off = rel2foff($rel->[2]);
      if ( defined $f_off ) {
        printf("; has reloc%s type %X field offset %d\n", $is_a ? 'a' : '', $rel->[2], $f_off);
      } else {
        printf("; has reloc%s type %X\n", $is_a ? 'a' : '', $rel->[2]);
      }
    }
  }
  # dump label for current instr
  if ( defined($gs_loffs) && exists($gs_loffs->{$off}) ) {
    my $l = $gs_loffs->{$off};
    if ( !$l ) { printf("LABEL_%X:\n", $off); }
    else { printf("LABEL_%X: ; %s\n", $off, $g_attrs->attr_name($l)); }
  }
  # process scheduling/find dual instr
  process_sched($off, $sctx, $block);
  my $dual = get_dual($sctx);
  # dump body
  printf("/*%X*/%s", $off, get_spfx($dual));
  printf("%s%s;", $i_text, get_ssfx($dual));
  if ( $skip ) {
    printf("\n");
    return 0;
  }
  # track regs
  $g_ced->track($rt) if defined($rt);
  # check LUT
  my $lut = $g_ced->has_lut();
  if ( defined($lut) ) {
    printf(" LUT %x: %s", $lut, $g_ced->lut($lut));
  }
  printf("\n");
  # check const bank
  my $cb0 = get_ins_cb0();
  if ( defined $cb0 ) {
    printf("; CB %X", $cb0);
    my $cp = find_cparam($cb0);
    printf(" param off %X ord %d size %X", $cp->{'off'}, $cp->{'ord'}, $cp->{'size'}) if defined($cp);
    printf("\n");
  }
  if ( defined($opt_p) ) {
    # props
    my $props = $g_ced->ins_prop();
    if ( defined $props ) {
      printf("; Properties:\n");
      while( my($name, $pr) = each(%$props) ) {
        printf(" ; %s:", PR_name($name));
        # first - type, rest - list of fields names
        for ( my $fi = 0; $fi < scalar(@$pr); $fi++ ) {
          if ( !$fi ) {
            printf(" %s", PType_name($pr->[$fi]));
          } else { printf(" %s", $pr->[$fi]); }
        }
        printf("\n");
      }
    }
    # preds
    my $preds = $g_ced->ins_pred();
    if ( defined($preds) ) {
      printf("; Preds:\n");
      foreach my $pname ( sort keys %$preds ) {
        printf(" ; %s %s\n", $pname, $preds->{$pname});
      }
    }
  }
  # latency tabs
  dump_lat($block, $sctx) if defined ( $opt_l );
  1;
}

=pod

=head1 Registers reusing

I<reuse> attribute means hint for GPU scheduler that some register in an instruction can reuse the physical register
already allocated to one of its source operands, avoiding a full register allocation and reducing register pressure.
So having registers tracking (see L<https://redplait.blogspot.com/2025/11/barriers-registers-tracking-for-sass.html>)
it's easy to discover pattern for I<reuse> generated by ptxas:

=begin text

1a0: LOP3.LUT PT,R6,RZ,R15,RZ, 0x33,!PT ; R6 got some value
1d0: SHF.R.S32.HI R7,RZ, 0x1F,R6.reuse  ; R6 used as source operand and marked with reuse
1f0: IMAD.WIDE R6,PT,R4, 0x2,R6 ; and here it's actually reused as source operand
; Note that R6 also got new value: R6 (and R7 bcs of .WIDE) = R4 * 2 + R6

=end text

Track for R6 looks like

=begin text
;   1A0 mask 8000 <- INTEGER
;   1D0 mask 200 reuse INTEGER
;   1F0 mask 0 INTEGER
;   1F0 mask 8000 <- INTEGER

=end text

So we can build FSM to traverse registers track and try to find where we can insert I<reuse> attribute - when there are

=over

=item * two or more sequential reads

=item * not marked as reused by PTX yet

=item * instructions has no conditional @PX

=item * register is not wide

=item * and those instructions are located not very far from each other - bcs reuse cache size is unknown but obviously limited.
I arbitrary choosed value 0x70

=back

unfortunately ragel (L<https://www.colm.net/open-source/ragel/>) can't produce perl code, so lets make FSM in old-school - manually

FSM is actually very simple and has only 3 states:

=over

=item 0 - no previous read operation, initial state

=item 1 - single read operation, store it somewhere

=item 2 - second read operation, add it to result after check distance and go to state 1

=back

=head3 Function prototype

Argument - array of register track - from RegTrack r/ur method, see how to dump it in function B<dump_gpr>

returs array of [ offset, mask ]

For described above example if I<reuse> attribute missed function should return

=begin text

 [0x1D0, 0x200]

=end text
=cut
sub collect_reuse
{
  my $rh = shift;
  return unless defined($rh);
  my @res;
  my $state = 0;
  my($prev);
  foreach my $l ( @$rh ) {
    # if this is write operation
    if ( $l->[2] ) { $state = 0; undef $prev; next; }
    # if this is wide
    my $wi = rh_widx($l->[1]);
    if ( $wi ) { $state = 0; undef $prev; next; }
    # if it has predicate
    if ( defined $l->[3] ) { $state = 0; undef $prev; next; }
    # we have some read operation - check if it's not first
    if ( $state && defined($prev)) {
      # check distance and that this is not Nth operand in the same instruction
      if ( $l->[0] != $prev->[0] && $l->[0] - $prev->[0] <= 0x70 ) {
        # check if previous read not marked already with reuse
        push(@res, $prev) unless ( rh_reuse($prev->[1]) );
      }
    }
    # store current operation in $prev
    $state = 1;
    $prev = [ $l->[0], $l->[1] ];
  }
  # probably should add wantarray support
  return scalar(@res) ? \@res : undef;
}

sub disasm
{
  my $s_size = shift;
  my $sctx = make_sctx();
  do {
    my $off = $g_ced->get_off();
    check_sym($off);
    dump_ins($off, $sctx);
  } while( $g_ced->next_off() < $s_size && $g_ced->next() );
}

# dump reg track snapshot for current instruction
sub dump_snap
{
  my($g, $pr) = @_;
  if ( defined $g ) {
    printf("; used regs:\n");
    while( my($r, $flag) = each(%$g) ) {
      printf(";  %sR%d: %X", $r & 0x8000 ? 'U' : '', $r & 0xff, $flag);
      printf(" write") if ( $flag & 0x80 );
      printf(" reuse") if ( $flag & 0x40 );
      printf(" read")  if ( $flag & 0x20 );
      printf("\n");
    }
  }
  if ( defined $pr ) {
    printf("; used predicates:\n");
    while( my($r, $flag) = each(%$pr) ) {
      printf(";  %sP%d: %d\n", $r & 0x8000 ? 'U' : '', $r & 0x7, $flag);
    }
  }
}

# check if found instruction for reuse really has reusage mask
# args:
#  [ offset, mask ] from collect_reuse
#  register index
#  is universal register
# returns [ attribute_name, tab_index if attr in table ]
sub resolve_rusage
{
  my($rt, $ridx, $is_u) = @_;
  my $off = $rt->[0];
  printf("resolve_rusage at %X for %X is_u %d\n", $off, $ridx, $is_u) if defined($opt_d);
  if ( !$g_ced->off($off) ) {
    carp("off $off failed");
    $gU_dis_failed++;
    return 0;
  }
  my $pr = rh_ops($rt->[1]);
  if ( defined $pr ) {
    my $rname = rkey($pr, $is_u);
    return 0 unless defined($rname);
    my $r_value = $g_ced->get($rname);
    if ( defined $r_value && $g_ced->get($rname) == $ridx ) { # sanity check
      my $reuse_name = reuse_attr($pr);
      # reuse_src_e and next are efields while reuse_src_a is table field, so check both
      my $t_idx = $g_ced->has_tfield($reuse_name);
      if ( defined($t_idx) || $g_ced->efield($reuse_name) ) {
        printf(" %s", $reuse_name) if defined($opt_v);
        return [ $reuse_name, $t_idx ];
      }
    }
  } else {
    # well, lets do some brute-force
    for my $t ( Cubin::Ced::ISRC_A() .. Cubin::Ced::ISRC_I() ) {
      my $rname = rkey($t, $is_u);
      next unless $rname;
      my $r_value = $g_ced->get($rname);
      next if ( !defined($r_value) || $r_value != $ridx );
      my $reuse_name = reuse_attr($t);
      my $t_idx = $g_ced->has_tfield($reuse_name);
      if ( defined($t_idx) || $g_ced->efield($reuse_name) ) {
        printf(" %s", $reuse_name) if defined($opt_v);
        return [ $reuse_name, $t_idx ];
      }
    }
  }
  $gU_not_found++;
  0;
}

# dump track of predicates
# args: RegTrack, map where keys are predicates and prefix U for universal
sub dump_ps
{
  my($rt, $rs, $pfx) = @_;
  my $is_u = $pfx ne '';
  my $res = 0;
  foreach my $k ( sort { $a <=> $b } keys %$rs ) {
    my $l = $is_u ? $rt->up($k) : $rt->p($k);
    next unless defined($l);
    printf("; %sP%d\n", $pfx, $k);
    $res++;
    foreach my $ar ( @$l ) {
      printf(";   %X", $ar->[0]); # offset
      printf(" mask %X", $ar->[1]) if defined($opt_d);
      # check predicate
      if ( defined $ar->[3] ) {
        my $pr = rh_pred($ar->[1]);
        # for @P there will be rh_pred having the same value as our key k - then ingore it
        # else you can't use pfx bcs there can be instruction like @UP1 setp p2
        printf(" @%sP%d", rh_upred($ar->[1]) ? 'U' : '', $pr)  if ( $pr != $k );
      }
      printf(" <-") if ( $ar->[2] );
      # finally add new line
      printf("\n");
    }
  }
  $res;
}

# dump track of (u)gprs
# args: RegTrack, map where keys are regs and prefix U for universal
# format of each list entry see in fill_reg file Ced.xs
sub dump_gpr
{
  my($rt, $rs, $pfx) = @_;
  my $is_u = $pfx ne '';
  my $res = 0;
  foreach my $k ( sort { $a <=> $b } keys %$rs ) {
    my $l = $is_u ? $rt->ur($k) : $rt->r($k);
    next unless defined($l);
    printf("; %sR%d\n", $pfx, $k);
    $res++;
    foreach my $ar ( @$l ) {
      printf(";   %X", $ar->[0]); # offset
      printf(" mask %X", $ar->[1]) if defined($opt_d);
      # check predicate
      if ( defined $ar->[3] ) {
        my $pr = rh_pred($ar->[1]);
        printf(" @%sP%d", rh_upred($ar->[1]) ? 'U' : '', $pr);
      }
      my $is_comp = rh_comp($ar->[1]);
      if ( $ar->[2] ) { # check is_write
        printf(" %s", $is_comp ? 'W' : '<-');
      }
      # windex
      my $wi = rh_widx($ar->[1]);
      printf(" w %d", $wi) if ( $wi );
      # for compound also dump it's prop
      if ( $is_comp ) {
        my $pr = rh_ops($ar->[1]);
        if ( defined $pr ) {
          my $pr_name = PR_name($pr);
          printf(" [%s]", $pr_name) if defined($pr_name);
        }
      }
      printf(" reuse") if ( rh_reuse($ar->[1]) );
      # last is type
      if ( defined $ar->[4] ) {
        my $t_name = PType_name($ar->[4]);
        printf(" %s", $t_name) if defined($t_name);
      }
      # finally add new line
      printf("\n");
    }
    # check possibly registers reuse
    if ( defined $opt_U ) {
      my $reus = collect_reuse($l);
      if ( defined $reus ) {
        my $r_size = scalar @$reus;
        printf("; %d reuses:\n", $r_size);
        $gU_found += $r_size;
        for my $ru ( @$reus ) {
          printf(";   %X mask %d", $ru->[0], $ru->[1]);
          if ( resolve_rusage($ru, $k, $is_u) ) {
            $gU_solved++;
            printf(" can patch") if defined($opt_v);
          }
          printf("\n");
        }
      }
    }
  }
  $res;
}

sub dump_rt
{
  my $rt = shift;
  return if ( $rt->empty() );
  my $cbs = $rt->cbs();
  if ( defined $cbs ) {
    printf(";;; Const banks\n");
    foreach my $cb ( @$cbs ) {
      printf(";  at %X idx %d off %X %X\n", $cb->[0], $cb->[1], $cb->[2], $cb->[3]);
    }
  }
  my $ps = $rt->ps();
  if ( defined $ps && scalar keys %$ps ) {
    printf(";;; Predicates\n");
    dump_ps($rt, $ps, '');
  }
  my $ups = $rt->ups();
  if ( defined $ups && scalar keys %$ups ) {
    printf(";;; UPredicates\n");
    dump_ps($rt, $ups, 'U');
  }
  my $rs = $rt->rs();
  if ( defined $rs && scalar keys %$rs ) {
    printf(";;; GPRs\n");
    dump_gpr($rt, $rs, '');
  }
  my $urs = $rt->urs();
  if ( defined $urs && scalar keys %$urs) {
    printf(";;; Universal Regs\n");
    dump_gpr($rt, $urs, 'U');
  }
}

# logic for checking regs/predicates interleaving
# preds filling in merge_preds - they are map where key is pred number and value is int - we need check 2 for write operation
sub dep_preds
{
  my($cur, $up) = @_;
  # reset bcs we have return in middle of while each
  keys %$cur;
  while( my($k, $v) = each %$cur ) {
    next if ( !($v & 1) ); # ignore non-read
    next if ( !exists $up->{$k} );
    if ( $up->{$k} & 2 ) {
      printf("; dep from pred %d\n", $k) if defined($opt_d);
      return 1;
    }
  }
  0;
}

# regs filling in gprs - they are map where key is reg number
# value is 0x80 for write, 0x20 for read
sub dep_regs
{
  my($cur, $up) = @_;
  # reset bcs we have return in middle of while each
  keys %$cur;
  while( my($k, $v) = each %$cur ) {
    next if ( ($v & 0x80) && !($v & 0x20) ); # write-only
    next if ( !exists $up->{$k} );
    if ( $up->{$k} & 0x80 ) {
      printf("; dep from reg %d\n", $k) if defined($opt_d);
      return 1;
    }
  }
  0;
}

sub int_preds
{
  my($cur, $up) = @_;
  dep_preds($cur, $up) || dep_preds($up, $cur);
}

sub int_regs
{
  my($cur, $up) = @_;
  dep_regs($cur, $up) || dep_regs($up, $cur);
}

# curr instr snap in block->[8], prev in block->[9]
# return 1 if 2 instrs are interleaved on some register/predicate
sub is_interleaved
{
  my $b = shift;
  return 0 unless(defined($b->[8]) && defined($b->[9]));
  my $curr = $b->[8];
  my $prev = $b->[9];
  my $res = 0;
  $res += int_preds($curr->[1], $prev->[1]) if ( defined($curr->[1]) && defined($prev->[1]) );
  $res += int_regs($curr->[0], $prev->[0]) if ( defined($curr->[0]) && defined($prev->[0]) );
  $res;
}

sub gdisasm
{
  my $dg = shift;
  for my $block ( @$dg ) {
    my $off = $block->[0];
    my $block_off = $g_ced->block_off($off);
    # check if off starts at block boundary
    $off += 8 if ( $g_w < 128 && $off == $block_off );
    if ( !$g_ced->off($off) ) {
      carp("cannot set offset $off");
      next;
    }
    # per-block data
    my $sctx = make_sctx();
    head_syms($block_off, $off);
    my $rt;
    $rt = Cubin::Ced::RegTrack->new() if defined($opt_t);
    my $idx = 0;
    # disasm every instruction in this block
    do {
      my $may_swap = 0;
      $off = $g_ced->get_off();
      my $res = dump_ins($off, $sctx, $block, $rt);
      # check shared barriers
      if ( defined $opt_b ) {
        my $sc = sched_check($block);
        printf("; shared barrier(s) %d\n", $sc) if ( $sc );
        # check if we don't share any barriers
        $may_swap = 1 if ( defined($opt_s) && $res && !$sc );
      }
      # dump snap
      if ( $res && defined($rt) ) {
        printf("; mask %X mask2 %X\n", $rt->mask(), $rt->mask2()) if defined($opt_v);
        my($g, $pr) = $rt->snap();
        dump_snap($g, $pr) if ( defined($g) || defined($pr) );
        add_ruc($block, $off, $g) if ( defined($g) && defined($opt_u) );
        # compare snap from current at index 8 with prev at index 9
        if ( !$idx ) {
          # no prev - just store current in ->[9]
          $block->[9] = [ $g, $pr ];
        } else {
          $block->[8] = [ $g, $pr ];
          my $inter = is_interleaved($block);
          if ( $may_swap && !$inter ) {
            my $gain = can_swap($block);
            if ( defined($gain) && $gain > 0 ) {
              upd_swap_stat($gain, get_old_pair_stall($block));
              printf("; Can swap to reduce %d\n", $gain);
            }
          }
          # shift for next instruction
          $block->[9] = $block->[8];
          $block->[8] = undef;
        }
      }
      # swap -s data
      if ( defined $opt_s ) {
        if ( !$res ) {
          $block->[13] = [];
          $block->[14] = [];
        } else {
          $gs_total++;
          $block->[14] = $block->[13];
          $block->[13] = [];
        }
      }
      $rt->snap_clear() if ( defined $rt );
      $idx++;
    } while( $g_ced->next_off() < $block->[1] && $g_ced->next() );
    # do block post-processing of block here
    if ( defined $rt ) {
      $rt->finalize();
      dump_rt($rt);
    }
  }
}

=pod

=head1 CFG functions

Seems that recovering of CFG belongs to the category 'everyone has known for a long time'
In reality google gives hard to implement algos like L<https://nicolo.dev/en/blog/role-control-flow-graph-static-analysis/>
So I invented my own - sure bcs of NIH syndrome

=head2 Some theory

Basic block typically can have 1 or 2 out edges - like in case of conditional branch you will have link from those branch and
link to next instruction
However in SASS Indirect Branches have corresponding IBT in attributes and can contain several targets

The next problem is how to split already found blocks. Lets check couple of code fragments:

=begin text

  STG.E [R10], R12                Block A              BRA L_1
label: ; can be obtained from code located below
  ..                              Block B

=end text

for left code we need to add link from A to B
for right code - no, bcs it ends with unconditional branch
So at least we must keep addresses of unconditional branches to avoid linking with next block

And also there are strange dead-loops like

=begin text

.L_x_4:
 BRA `(.L_x_4)
 NOP

=end text

I don't want to add such blocks at all

And finally we can have symbols in unpredictable places - lets check another couple of blocks

=begin text

  STG.E [R10], R12      Block A     @P1 BRA L_X
symbol: ; some local/global function
   ...                  Block B

=end text

Seems that we must close previous block and start new one

=head2 Complexity of algorithm

=over

=item Pass 1 - collect labels, complexity is O(N) where N is number of instructions and we need to lookup in Ibt each processed instruction -
this can be done with sorted list of IBT sources

=item Pass 2 - sort O(m * log(m)) + O(m) where m is amount of blocks + markers

=item Pass 3 - resolve block back-links. If we have M blocks - each can have M - 1 back references, resolving can use binary search, so
total O(M * M * log(M))

=back

=cut
# args: back-refs map, to addr, from
sub add_label
{
 my($br, $addr, $from) = @_;
 # check if it exists
 if ( exists $br->{$addr} ) {
   push @{ $br->{$addr} }, $from;
 } else {
   $br->{$addr} = [ $from ];
 }
}

sub dump_blocks
{
  my($br, $resolved) = @_;
  foreach my $b ( @$br ) {
    printf("block %X till %X", $b->[0], $b->[1]);
    printf(" sym %d", $b->[2]) if ( defined $b->[2] );
    printf(":\n");
    if ( $resolved ) {
      printf(" %X", $_->[0]) for ( values %{ $b->[3] } );
    } else {
     printf(" %X", $_) for ( keys %{ $b->[3] } );
    }
    printf("\n");
  }
}

# merge IBTs from $gs_ibt with back-refs
sub merge_ibts
{
  my $br = shift;
  return unless defined($gs_ibt);
# print "Ibt " . Dumper($gs_ibt);
  # format of ibt from collect - key is address of destination, value is [ list of sources ]
  # we just copy it to br map to avoind patching of original $gs_ibt
  $br->{$_} = $gs_ibt->{$_} for ( keys %$gs_ibt );
  # get original IBTs
  my $id = $g_attrs->grep(0x34);
  return unless defined($id);
  my $ib_values = $g_attrs->value($id->[0]->{'id'});
  return unless defined($ib_values);
  # and make sorted array of source instructions
  my @res;
  push @res, $_ for ( sort { $a <=> $b } keys %$ib_values );
  \@res;
}

# check if some instruction with label is 'pre' - see details at
#  https://docs.nvidia.com/cuda/archive/12.2.1/cuda-binary-utilities/index.html#maxwell-and-pascal-instruction-set
sub is_pre
{
  return 0 if ( $g_w > 88 );
  my $n = $g_ced->ins_name();
  return ($n eq 'PRET' || $n eq 'PBK' || $n eq 'PCNT' || $n eq 'PEXIT' );
}

sub dg
{
  my($code_off, $s_size) = @_;
  my %br; # map of branches and markers
  my $ibs = merge_ibts(\%br);
  my($ib_curr, $ib_size);
  if ( defined $ibs ) {
    $ib_curr = 0;
    $ib_size = scalar(@$ibs);
  }
  my $check_ibt = sub {
    my $off = shift;
    return 0 unless defined($ibs);
    return 0 if ( $ib_curr >= $ib_size );
    if ( $ibs->[$ib_curr] == $off ) {
      $ib_curr++;
      return 1;
    }
    0;
  };
  # first pass - collect links and marks in br map
  my $has_prev; # if we should add link from previous instruction
  my $add_prev = sub {
   my $of = shift;
   if ( defined $has_prev ) {
     add_label(\%br, $of, $has_prev);
     undef $has_prev;
   }
  };
  my $cnd_sub = sub {
   my($cond, $off) = @_;
   if ( $cond ) { $has_prev = $off; }
   else { undef $has_prev; }
  };
  my $off; # at end will hold last processed address
  do {
    $off = $g_ced->get_off();
    gcheck_sym(\%br, $off);
    my $skip = is_skip();
    if ( $skip ) { $add_prev->($off); }
    else {
      my $brt = $g_ced->ins_brt();
      my $cond = $g_ced->has_pred();
      my $link_prev = sub {
        $add_prev->($off);
        $cnd_sub->($cond, $off);
        $br{$off+1} = 1 if ( !$cond );
      };
      # check if this is IBT
      if ( $check_ibt->($off) ) {
        printf("ibt at %X\n", $off) if defined($opt_d);
        # nothing to add - br arleady has IBT labels
        $link_prev->();
      } else {
        # check if have some branch
        my $is_dl = 0;
        my $added = 0;
        my $pre = 0;
        if ( $brt == Cubin::Ced::BRT_RETURN || $brt == Cubin::Ced::BRT_BRANCHOUT ) {
          # return/exit don't have address - so logic is the same as for IBT
          $link_prev->();
        } else {
          my($rel, $is_a) = has_rel($off);
          # ignore instr having relocs
          unless($rel) {
            my $addl = $g_ced->ins_clabs();
            if ( defined($addl) ) {
              $added = 1;
              if ( $addl == $off ) { # this is dead-loop
                $is_dl = 1;
              } else {
                add_label(\%br, $addl, $off);
                $pre = is_pre();
                $gs_loffs->{$addl} = 0; # this is really some new labal - store it for later disasm too
              }
            }
          }
          # link with prev instr
          $add_prev->($off) unless($is_dl);
          if ( $added ) {
            # check if we have conditional branch
            if ( $pre ) { $cnd_sub->($pre, $off); }
            else { $cnd_sub->($cond, $off) if ( $brt ); }
            if ( $is_dl ) {
              $br{$off+1} = -1; # put dead-loop marker
            } elsif ( $brt != Cubin::Ced::BRT_CALL && !$cond ) { $br{$off+1} = 1; }
          }
        }
      }
    }
  } while( $g_ced->next_off() < $s_size && $g_ced->next() );
  # make sorted array
  my @sorted = sort { $a->[0] <=> $b->[0] } map {
   [ $_, $br{$_} ]
   } keys %br;
  return unless scalar(@sorted);
  # dump what we collected
  if ( defined $opt_d ) {
    foreach my $s (@sorted) {
      printf("%X:", $s->[0]);
      if ( 'ARRAY' eq ref $s->[1] ) {
        printf(" %X", $_) for @{ $s->[1] };
      } else {
        if ( 2 == ($s->[0] & 7) ) { printf(" symbol %d", $s->[1]); }
        else { printf(" marker %d", $s->[1]); }
      }
      printf("\n");
    }
  }
=comment
  pass 2 - make blocks, complexity O(m) where m is number of marks/symbols/back references in @sorted array
  indexes in block:
   [0] - start address
   [1] - last address
   [2] - symbol index or undef
   [3] - map with back-refs
  below is data for barriers tracking - used in process_sched & check_sched
   [4] - array of wait or undef
   [5] - map of read/write
   [6] - array of wait for previous instruction
   [7] - map of read/write for previous instruction
  registers tracking (-t option):
   [8] - snap array from current instruction
   [9] - snap array from previous instruction
  [10] - map with currently reused registers - for -u option
   latency tables (-l option):
  [11] - col indexes of previous instruction from l2map
  [12] - row indexes of previous instruction from l2map
   to find swappable instructions (-s option) we need more sophisticated data to keep some instructions properties
   I chose array, indexes (first 7 from dump_ins) are
    * 0 - offset
    * 1 - instruction name
    * 2 - full instruction text
    * 3 - has rel offset
    * 4 - has rela offset
    * 5 - brt
    * 6 - has cond
    * 7 - stall count - filled in process_sched
    * 8 - is dual - filled in process_sched
    * 9 - TBC
  [13] - properties for current instruction
  [14] - properties for previous instruction
=cut
  my @bbs;
  my $add_block = sub {
    my $off = shift;
    my @res = ( $off );
    my %bl;
    $res[3] = \%bl;
    push @bbs, \@res;
    if ( defined $opt_u ) {
      my %ruc;
      $res[10] = \%ruc;
    }
    if ( defined $opt_s ) {
      my @tmp;
      $res[13] = \@tmp;
    }
    \@res;
  };
  my $cb;
  my $close_block = sub {
    $cb->[1] = shift;
    undef $cb;
  };
  # check if we need to add first block
  my $need_firstb = 1;
  $need_firstb = 0 if ( scalar(@sorted) && 'ARRAY' eq $sorted[0]->[1] && $sorted[0]->[0] <= $code_off );
  $cb = $add_block->($code_off) if ( $need_firstb );
  foreach my $cop ( @sorted ) {
# we have 8 cases here
# has block  current operand  what to do
#   N          sym            add new block and add symbol
#   Y          sym            close prev block and start new, also add symbol
#   N        dead loop        skip
#   Y        dead loop        close prev block
#   N        back ref         add new block and back ref to it
#   Y        back ref         put back ref to current block
#   N         marker          wtf? add new block with single instruction - don't know if it has sence
#   Y         marker          close prev block
    if ( 'ARRAY' eq ref $cop->[1] ) {
      my $need_close = 0;
      if ( defined $cb ) {
        # check if all labels located in current block
        for ( @{ $cop->[1] } ) {
          if ( $_ < $cb->[0] || $_ > $cop->[0] ) {
            $need_close++;
            last;
          }
        }
      }
      $close_block->($cop->[0]-1) if ( $need_close );
      $cb = $add_block->($cop->[0])  unless $cb;
      $cb->[3]->{$_} = 0 for ( @{ $cop->[1] } );
      next;
    }
    my $kind = $cop->[0] & 7;
    my $curr_off = $cop->[0] - $kind;
    if ( 2 == $kind ) { # symbol
      unless($cb) {
        $cb = $add_block->($curr_off);
        $cb->[2] = $cop->[1];
        next;
      }
      unless(defined $cb->[2]) {
        $cb->[2] = $cop->[1];
        next;
      }
      # we have some block with symbol and now new symbol - wtf?
      # ok, close prev and make new block
      $close_block->(1+$g_ced->prev_off($curr_off));
      $cb = $add_block->($curr_off);
      $cb->[2] = $cop->[1];
      next;
    }
    # marker or dead loop
    if ( -1 == $cop->[1] ) {
      $close_block->(1+$g_ced->prev_off($curr_off)) if defined($cb);
      next;
    }
    if ( defined $cb ) {
      $close_block->(1+$curr_off);
      next;
    }
    # Как будто, как будто... Только я зачем тут-то?
  }
  # check if last block has last addr
  if ( scalar @bbs ) {
    $bbs[-1]->[1] = $off + 1 unless( defined $bbs[-1]->[1] );
  }
  dump_blocks(\@bbs, 0) if ( defined $opt_d );
  # pass 3 - resolve all back-references to blocks (now they are just offsets)
  # surprisingly this is most compute-intensive part - if we have M blocks then complexity will be O(M * M * log(M))
  # clojure for bin_sac
  my $bs_cb = sub {
    my($br, $addr) = @_;
    return 0 if ( $addr >= $br->[0] && $addr < $br->[1] );
    return -1 if ( $addr > $br->[0] );
    1;
  };
  foreach $cb ( @bbs ) {
    my $brs = $cb->[3];
    my %blinks;
    foreach my $addr ( keys %$brs ) {
      my($idx, $found) = bin_sac(\@bbs, $bs_cb, $addr);
      next unless defined($found);
      next if ( $found == $cb ); # skip self references
      $blinks{$addr} = $found;
    }
    $cb->[3] = \%blinks;
  }
  dump_blocks(\@bbs, 1) if ( defined $opt_d );
  # finally return blocks
  \@bbs;
}

# main
my $state = getopts("bdglprstUuv");
usage() if ( !$state );
if ( -1 == $#ARGV ) {
  printf("where is arg?\n");
  exit(5);
}
# some options validation
croak("you can track registers only with -g option") if ( defined($opt_t) && !defined($opt_g) );
croak("-u must be used with -t option") if ( defined($opt_u) && !defined($opt_t) );
croak("-U must be used with -t option") if ( defined($opt_U) && !defined($opt_t) );
if ( defined $opt_s ) {
 croak("you can use -s only with CFG -g option") unless defined($opt_g);
 croak("-s must be used with -t option") unless defined($opt_t);
 croak("-s must be used with -b option") unless defined($opt_b);
}

# load elf, symbols, ced & attrs
$g_elf = Elf::Reader->new($ARGV[0]);
die("cannot open $ARGV[0]") unless defined($g_elf);
$g_syms = read_symbols($g_elf);
if ( defined $g_syms ) {
  my @tmp = grep { $_->[0] ne '' && $_->[4] != STT_SECTION && $_->[4] != STT_FILE } @$g_syms;
  $g_afsyms = \@tmp;
}
$g_ced = Cubin::Ced->new($g_elf);
die("cannot load cubin $ARGV[0]") unless defined($g_ced);
$g_w = $g_ced->width();
if ( defined $opt_v ) {
  printf("SM %s width %d block_mask %d\n", $g_ced->sm_name(), $g_w, $g_ced->block_mask());
}
my @es = exs($g_elf);
die("Empty cubin $ARGV[0]") unless scalar(@es);
$g_attrs = Cubin::Attrs->new($g_elf);
dir("Attrs failed on $ARGV[0]") unless defined($g_attrs);
printf("%d sections with code\n", scalar(@es)) if defined($opt_v);

# we have list of sections in @es
foreach my $s ( @es ) {
 my $a_idx = $g_attrs->try($s->[0]);
 # dump current section
 printf("[%d] attrs in %d size %X %s\n", $s->[0], $a_idx, $s->[9], $s->[1]);
 # try to extract externals
 next if ( !$g_attrs->read($a_idx) );
 my $ext = $g_attrs->grep(0xf);
 # section setup
 setup_syms($s->[0]);
 dump_ext($ext->[0]) if defined($ext);
 dump_cparams();
 ($gs_loffs, $gs_ibt) = $g_attrs->collect();
 # relocs
 if ( defined $opt_r ) {
   my $rel_idx = $g_attrs->try_rel($s->[0]);
   if ( defined($rel_idx) ) {
     dump_rels('REL', $s->[0], $rel_idx);
   } else { undef $gs_rel; }
   $rel_idx = $g_attrs->try_rela($s->[0]);
   if ( defined($rel_idx) ) {
     dump_rels('RELA', $s->[0], $rel_idx);
   } else { undef $gs_rela; }
 }
 # setup ced
 die("cannot setup section $s->[0]") unless $g_ced->set_s($s->[0]);
 my $off = $g_w == 128 ? 0: 8;
 die("initial offset") unless $g_ced->off($off);
 if ( defined $opt_g ) {
   my $graph = dg($off, $s->[9]); # args - start of code offset bcs we need at least 2 passes, section size
   sym_reset();
   gdisasm($graph);
 } else { disasm($s->[9]); }  # arg - section size
}
dump_ruc() if defined($opt_u);
dump_rU() if ( defined $opt_U );
dump_barstat() if defined($opt_b);
dump_lat_stat() if defined($opt_l);
dump_swap_stat() if defined($opt_s);