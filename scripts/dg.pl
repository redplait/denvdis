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
use vars qw/$opt_b $opt_g $opt_p $opt_r $opt_v/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] file.cubin
 Options:
  -b - track read/write barriers
  -p - dump properties
  -r - dump relocs
  -v - verbose mode
EOF
  exit(8);
}

# globals
my($g_elf, $g_attrs, $g_ced, $g_syms, $g_w);
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
# for -b option this is just map where key is barrier index and value is [ offset. R/W ]
sub make_sctx
{
  my %res;
  $res{'dual'} = 0 if ( $g_w < 128 );
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

sub process_sched
{
  my $sctx = shift;
  my $ctrl;
  my $is_dual = 0;
  if ( $g_w == 64 ) {
    $ctrl = $g_ced->ctrl();
    my $stall = $ctrl & 0x2f;
    $is_dual = $stall == 0x20;
    if ( defined $opt_b ) {
      printf("; ctrl %X\n", $ctrl);
    }
  } else {
    $ctrl = $g_ced->cword();
    # low 5 bits
    $is_dual = $g_ced->ins_dual() if ( $g_w == 88 );
    # render
    if ( defined $opt_b ) {
      my $s = $g_ced->render_cword($ctrl);
      printf("; cword %X %s\n", $ctrl, $s);
    }
  }
# printf(" dual %d %d\n", $is_dual, $sctx->{'dual'});
  $sctx->{'dual'} = 1 if ( $is_dual && !$sctx->{'dual'} );
}

sub dump_ins
{
  my($off, $sctx) = @_;
  my $brt = $g_ced->ins_brt();
  if ( defined $opt_v ) {
    my $cl = $g_ced->ins_class();
    my $ln = $g_ced->ins_line();
    printf("; %s line %d", $cl, $ln);
    printf(" ALT") if ( $g_ced->ins_alt() );
    printf(" Brt %d (%s)", $brt, brt_name($brt)) if $brt;
    printf("\n");
  }
  # is empty instruction - nop or with !@PT predicate
  my $skip = $g_ced->ins_false() or 'NOP' eq $g_ced->ins_name();
  # check instr for label
  if ( !$skip && $brt != Cubin::Ced::BRT_RETURN ) {
    my($rel, $is_a) = has_rel($off);
    # ignore instr having relocs
    unless($rel) {
      my $addl = $g_ced->ins_clabs();
      if ( defined($addl) ) {
        printf(" ; add label %X\n", $addl) if defined($opt_v);
        $gs_loffs->{$addl} = 0;
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
  process_sched($sctx);
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

# main
my $state = getopts("bgprv");
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
  printf("SM %s width %d\n", $g_ced->sm_name(), $g_w);
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
   dump_rels('REL', $s->[0], $rel_idx) if defined($rel_idx);
   $rel_idx = $g_attrs->try_rela($s->[0]);
   dump_rels('RELA', $s->[0], $rel_idx) if defined($rel_idx);
 }
 # setup ced
 die("cannot setup section $s->[0]") unless $g_ced->set_s($s->[0]);
 my $off = $g_w == 128 ? 0: 8;
 die("initial offset") unless $g_ced->off($off);
 disasm($s->[9]); # arg - section size
}