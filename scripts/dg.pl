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
use vars qw/$opt_b $opt_d $opt_g $opt_p $opt_r $opt_v/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] file.cubin
 Options:
  -b - track read/write barriers
  -d - debug mode
  -g - build cfg
  -p - dump properties
  -r - dump relocs
  -v - verbose mode
EOF
  exit(8);
}

# globals
my($g_elf, $g_attrs, $g_ced, $g_syms, $g_w);
# stat for barriers, key is ins name, value is [ wait, read, write ] count
my %g_barstat;
# per code section globals
# syms inside section & curr_index
my(@gs_syms, $gs_cidx);
# relocs
my($gs_rel, $gs_rela);
# cb params
my(@gs_cbs, $gs_cb_size, $gs_cb_off);
# labels from attrs
my($gs_loffs, $gs_ibt);

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
  @gs_syms = sort { $a->[1] <=> $b->[1] }
   # grep named with right section at [5] and type at [4]
   grep { $_->[5] == $sidx && $_->[0] ne '' && $_->[4] != STT_SECTION } @$g_syms;
  $gs_cidx = 0;
  if ( defined $opt_v ) {
    printf(" %d symbols:\n", scalar @gs_syms);
    foreach my $s ( @gs_syms ) {
      printf("  %X type %x size %x %s\n", $s->[1], $s->[4], $s->[2], $s->[0]);
    }
  }
}

sub check_sym
{
  my $off = shift;
  my $res = 0;
  return if ( $gs_cidx >= scalar(@gs_syms) );
  while ( $gs_syms[$gs_cidx]->[1] <= $off ) {
    my $sym = $gs_syms[$gs_cidx];
    # global?
    printf("\t.global %s\n", $sym->[0]) if ( STB_GLOBAL == $sym->[3] );
    # size
    printf("\t.size %X\n", $sym->[2]) if ( $sym->[2] );
    $res++;
    # dump name label
    printf("%s:\n", $sym->[0]);
    last if ( ++$gs_cidx >= scalar(@gs_syms) );
  }
  $res;
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

# args: ref to array, sub with <=> like return
sub bin_sac
{
  my($ar, $cb) = @_;
  my $low = 0;
  my $high = scalar @$ar; # Index of the last element
  while ($low < $high) {
     my $mid = int(($low + $high) / 2); # Calculate the middle index
     my $res = $cb->($ar->[$mid]);
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
  printf(" %s %d: %d\n", $pfx, $r_idx, $res);
  if ( $is_a ) {
    $gs_rela = $res ? \%rels: undef;
  } else {
    $gs_rel = $res ? \%rels: undef;
  }
  return if ( !$res );
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
    return (undef, 0) unless exists($gs_rel->{$off});
    return ($gs_rel->{$off}, 0);
  }
  if ( defined $gs_rela ) {
    return (undef, 0) unless exists($gs_rela->{$off});
    return ($gs_rela->{$off}, 1);
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

# scheduler context
# for old 64/88 bit SM has 'dual' field
# for -b option this is just map where key is barrier index and value is [ offset, R/W ]
# current stall count stored in 'roll' field
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

sub process_sched
{
  my($off, $sctx) = @_;
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
    if ( defined $opt_b ) {
      # update rolling stall count
      $sctx->{'roll'} += $stall;
      # dump current stall counts
      printf("; stall %d total %d ctrl %X\n", $stall, $sctx->{'roll'}, $ctrl);
    }
  } else {
    $ctrl = $g_ced->cword();
    # low 5 bits
    $is_dual = $g_ced->ins_dual() if ( $g_w == 88 );
    # render
    if ( defined $opt_b ) {
      my @stat = (0, 0, 0);
      my $curr_stall = $sctx->{'roll'};
      my $s = $g_ced->render_cword($ctrl);
      $stall = ($ctrl & 0x0000f) >> 0;
      # dump current stall counts
      printf("; stall %d total %d cword %X %s\n", $stall, $curr_stall, $ctrl, $s);
      # track barriers - ripped from maxas printCtrl
      my $wrtdb = ($ctrl & 0x000e0) >> 5;  # 3bit write dependency barrier
      my $readb = ($ctrl & 0x00700) >> 8;  # 3bit read  dependency barrier
      my $watdb = ($ctrl & 0x1f800) >> 11; # 6bit wait on dependency barrier
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
# printf(" dual %d %d\n", $is_dual, $sctx->{'dual'});
  $sctx->{'dual'} = 1 if ( $is_dual && !$sctx->{'dual'} );
}

sub dump_ins
{
  my($off, $sctx) = @_;
  my $brt = $g_ced->ins_brt();
  my $scbd = $g_ced->ins_scbd();
  my $mw = $g_ced->ins_min_wait();
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
  # is empty instruction - nop or with !@PT predicate
  my $skip = $g_ced->ins_false() || 'NOP' eq $g_ced->ins_name();
  # check instr for label
  if ( !$skip ) { # && $brt != Cubin::Ced::BRT_RETURN ) {
    my($rel, $is_a) = has_rel($off);
    # ignore instr having relocs
    unless($rel) {
      my $addl = $g_ced->ins_clabs();
      if ( defined($addl) ) {
        printf(" ; add label %X\n", $addl) if defined($opt_v);
        $gs_loffs->{$addl} = 0;
      }
    } else {
      printf("; has reloc%s\n", $is_a ? 'a' : '') if defined($opt_v);
    }
  }
  # dump label for current instr
  if ( defined($gs_loffs) && exists($gs_loffs->{$off}) ) {
    my $l = $gs_loffs->{$off};
    if ( !$l ) { printf("LABEL_%X:\n", $off); }
    else { printf("LABEL_%X: ; %s\n", $off, $g_attrs->attr_name($l)); }
  }
  # process scheduling/find dual instr
  process_sched($off, $sctx);
  my $dual = get_dual($sctx);
  # dump body
  printf("/*%X*/%s", $off, get_spfx($dual));
  printf("%s%s;", $g_ced->ins_text(), get_ssfx($dual));
  if ( $skip ) {
    printf("\n");
    return;
  }
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

=pod

=head1 CFG functions

Seems that recovering of CFG belongs to the category 'everyone has known for a long time'
In reality google gives hard to implement algos like https://nicolo.dev/en/blog/role-control-flow-graph-static-analysis/
So I invented my own - sure bcs of NIH syndrome

=head2 Some theory

Basic block typically can have 1 or 2 out edges - like in case of conditional branch you will have link from those branch and
link to next instruction
However in SASS Indirect Branches have corresponding IBT in attributes and can contain several targets

The next problem is how to split already found blocks. Lets check couple of code fragments:

  STG.E [R10], R12                Block A              BRA L_1
label: ; can be obtained from code located below
  ..                              Block B

for left code we need to add link from A to B
for right code - no, bcs it ends with unconditional branch
So at least we must keep addresses of unconditional branches to avoid linking with next block

And also there are strange dead-loops like

.L_x_4:
 BRA `(.L_x_4)
 NOP

I don't want to add such blocks at all

=head2 Complexity of algorithm

Pass 1 - collect labels, complexity is O(N) where N is number of instructions and we need to lookup in Ibt each processed instruction -
this can be done with sorted list of IBT sources

Pass 2 - sort O(m * log(m)) + O(m) where m is amount of blocks + markers

Pass 3 - resolve block back-links. If we have M blocks - each can have M - 1 back references, resolving can use binary search, so
total O(M * M * log(M))

=cut
# merge IBTs from $gs_ibt with back-refs
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
  # dead-loops
  my %dl;
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
  do {
    my $off = $g_ced->get_off();
    my $skip = $g_ced->ins_false() || 'NOP' eq $g_ced->ins_name();
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
        printf("ibt at %X\n", $off) if defined($opt_v);
        # nothing to add - br arleady has IBT labels
        $link_prev->();
      } else {
        # check if have some branch
        my $is_dl = 0;
        my $added = 0;
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
                $dl{$off} = 1;
              } else {
                add_label(\%br, $addl, $off);
              }
            }
          }
          # link with prev instr
          $add_prev->($off) unless($is_dl);
          # check if we have conditional branch
          $cnd_sub->($cond, $off) if ( $added );
          if ( $is_dl ) {
            $br{$off+1} = -1; # put dead-loop marker
          } elsif ( $added && $brt != Cubin::Ced::BRT_CALL && !$cond ) { $br{$off+1} = 1; }
        }
      }
    }
  } while( $g_ced->next_off() < $s_size && $g_ced->next() );
  # make sorted array
  my @sorted = sort { $a->[0] <=> $b->[0] } map {
   [ $_, $br{$_} ]
   } keys %br;
  # dump what we collected
  if ( defined $opt_d ) {
    foreach my $s (@sorted) {
      printf("%X:", $s->[0]);
      if ( 'ARRAY' eq ref $s->[1] ) {
        printf(" %X", $_) for @{ $s->[1] };
      } else {
        printf(" marker %d", $s->[1]);
      }
      printf("\n");
    }
  }
}

# main
my $state = getopts("bdgprv");
usage() if ( !$state );
if ( -1 == $#ARGV ) {
  printf("where is arg?\n");
  exit(5);
}

# load elf, symbols, ced & attrs
$g_elf = Elf::Reader->new($ARGV[0]);
die("cannot open $ARGV[0]") unless defined($g_elf);
$g_syms = read_symbols($g_elf);
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
 if ( defined $opt_g ) { dg($off, $s->[9]); } # args - start of code offset bcs we need at least 2 passes, section size
 else { disasm($s->[9]); }  # arg - section size
}
dump_barstat() if defined($opt_b);