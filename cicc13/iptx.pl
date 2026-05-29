#!perl -w
# try intersect ptx.txt from cicc with ptx_ops2.txt extracted from ptxas
# -f for mask frequency analysis
use strict;
use warnings;
use Getopt::Std;
use Data::Dumper;

# options
use vars qw/$opt_a $opt_B $opt_b $opt_d $opt_f $opt_i $opt_o $opt_k $opt_t $opt_U $opt_w/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] md.txt
 Options:
 -b idx:shift
 -B list of idx:shift
 -d - dump known tables rows
 -f - mask frequency analysis
 -a ins1 ins2 ... - make and mask of instructions
 -i ins1 ins2 ... - make and mask of instructions - remained
 -o ins1 ins2 ... - make or mask of instructions - remained
 -k - exclude known
 -t - verify tabs and dump still unused
 -U - dump instruction not presented in cucc
 -w - ignore obscure _mma.warpgroup & _mma
EOF
  exit(8);
}

# instr names from ptx_ops2.txt
my %g_ins;
# array of ops, each element is [ line number, mask array, rest of op, op name ]
my @g_ops;
# neg mask of known
my $gk_neg;

sub apply_k
{
  my $ar = shift;
  return $ar unless($gk_neg);
  my @tmp = @$ar;
  my $res = 0;
  foreach my $i ( 0 .. 15 ) {
    $tmp[$i] &= $gk_neg->[$i];
    $res++ if ( $tmp[$i] );
  }
  return undef unless($res);
  \@tmp;
}

sub do_freq
{
  my(%lsk, %ins_k, $u_name);
  foreach my $i ( 0 .. 15 ) {
    foreach my $bi ( 0 .. 7 ) {
      my $mask = 1 << $bi;
      my $latch = 0;
      my @and_mask;
      foreach my $op ( @g_ops ) {
        my $ar = apply_k($op->[1]);
        next unless $ar;
        next unless( $ar->[$i] & $mask );
        if ( defined $opt_k ) {
          $lsk{ $op->[0] }++;
          $ins_k{ $op->[3] }++;
        }
        unless($latch) {
          printf("idx %d bit %d:\n", $i, $bi);
          $latch++;
          @and_mask = @$ar;
        } else {
          foreach my $mi ( 0 .. 15 ) {
            $and_mask[$mi] &= $ar->[$mi];
          }
        }
        printf(" line %d: %s\n", $op->[0], $op->[2]);
      }
      # dump and mask
      if ( $latch ) {
        printf("and_mask: ");
        dump_mask(\@and_mask);
        dump_mtabs(\@and_mask, '') if defined($opt_d);
      }
    }
  }
  # dump stat
  if ( $opt_k ) {
    my $o_size = scalar(@g_ops);
    my $l_size = scalar keys %lsk;
    printf("%d forms from %d, %f\n", $l_size, $o_size, 100.0 * $l_size / $o_size);
    # try to find most unknown instruction
    my $u_max = 0;
    while ( my($in, $iv) = each %ins_k ) {
      if ( $iv > $u_max ) {
        $u_name = $in;
        $u_max = $iv;
      }
    }
    my $ik = scalar keys %ins_k;
    printf("%d ins, %f\n", $ik, 100.0 * $ik / scalar(keys %g_ins));
    printf("obscuriest %s (%d)\n", $u_name, $u_max) if $u_name;
  }
}

# for -b. args: index shift
sub try_mask
{
  my($idx, $sh) = @_;
  my $mask = 1 << $sh;
  my @and_mask;
  foreach my $op ( @g_ops ) {
    my $ar = apply_k($op->[1]);
    next unless($ar);
    next unless( $ar->[$idx] & $mask );
    printf(" line %d: %s\n", $op->[0], $op->[2]);
    unless( scalar @and_mask ) {
      @and_mask = @$ar;
    } else {
      foreach my $mi ( 0 .. 15 ) {
        $and_mask[$mi] &= $ar->[$mi];
      }
    }
  }
  if ( scalar @and_mask ) {
    printf("and_mask: ");
    dump_mask(\@and_mask);
    dump_mtabs(\@and_mask, '') if defined($opt_d);
  }
}

# for -B. args - hash where keys are indexes and values - masks
sub try_maskB
{
  my($hr) = @_;
  my @and_mask;
OUTER:
  foreach my $op ( @g_ops ) {
    my $ar = apply_k($op->[1]);
    next unless($ar);
    foreach my $idx ( keys %$hr ) {
      my $mask = $hr->{$idx};
      next OUTER unless( $ar->[$idx] & $mask );
    }
    printf(" line %d: %s\n", $op->[0], $op->[2]);
    unless( scalar @and_mask ) {
      @and_mask = @$ar;
    } else {
      foreach my $mi ( 0 .. 15 ) {
        $and_mask[$mi] &= $ar->[$mi];
      }
    }
  }
  if ( scalar @and_mask ) {
    printf("and_mask: ");
    dump_mask(\@and_mask);
    dump_mtabs(\@and_mask, '') if defined($opt_d);
  }
}

# boring mask logic
sub dump_mask
{
  my $mr = shift;
  foreach my $mi ( @$mr ) {
    printf("%2.2X ", $mi);
  }
  printf("\n");
}

sub bcnt
{
  my $ar = shift;
  my $res = 0;
  foreach my $m ( @$ar ) {
    foreach my $i ( 0 .. 7 ) {
      $res++ if ( $m & (1 << $i) );
    }
  }
  $res;
}

# apply not and of remaning masks in $rm
sub nand
{
  my($m, $rm) = @_;
  foreach my $mi ( @$rm ) {
    foreach my $i ( 0 .. 15 ) {
      $m->[$i] &= ~( $mi->[$i] );
    }
  }
}

sub filter_and
{
  my $hr = shift;
  my @res;
  my $found = 0;
  foreach my $o ( @g_ops ) {
    # check name in hash
    next unless ( exists $hr->{ $o->[3] } );
    my $ar = apply_k($o->[1]);
    next unless($ar);
    if ( !$found ) {
      @res = @$ar; # copy first mask array
    } else {
      # apply and
      foreach my $ai ( 0 .. 15 ) {
        $res[$ai] &= $ar->[$ai];
      }
    }
    $found++;
  }
  if ( !$found ) {
    printf("not found\n");
  } else {
    dump_mask(\@res);
    dump_mtabs(\@res, '') if defined($opt_d);
  }
  $found;
}

sub filter_or
{
  my $hr = shift;
  my(@res, @rem);
  my $found = 0;
  foreach my $o ( @g_ops ) {
    my $ar = apply_k($o->[1]);
    next unless($ar);
    # check name in hash
    unless ( exists $hr->{ $o->[3] } ) {
      push @rem, $ar;
      next;
    }
    if ( !$found ) {
      @res = @$ar; # copy first mask array
    } else {
      # apply and
      foreach my $ai ( 0 .. 15 ) {
        $res[$ai] |= $ar->[$ai];
      }
    }
    $found++;
  }
  if ( !$found ) {
    printf("not found\n");
  } else {
    nand(\@res, \@rem);
    dump_mask(\@res);
    dump_mtabs(\@res, '') if defined($opt_d);
  }
  $found;
}

sub filter_ins
{
  my $hr = shift;
  my(@res, @rem);
  my $found = 0;
  foreach my $o ( @g_ops ) {
    # check name in hash
    unless ( exists $hr->{ $o->[3] } ) {
      push @rem, $o->[1];
      next;
    }
    my $ar = $o->[1];
    if ( !$found ) {
      @res = @$ar; # copy first mask array
    } else {
      # apply and
      foreach my $ai ( 0 .. 15 ) {
        $res[$ai] &= $ar->[$ai];
      }
    }
    $found++;
  }
  if ( !$found ) {
    printf("not found\n");
  } else {
    nand(\@res, \@rem);
    dump_mask(\@res);
    dump_mtabs(\@res, '') if defined($opt_d);
  }
  $found;
}

sub read_ops2
{
  my $fname = shift;
  my @mask = ( 0 ) x16;
  my($fh, $str, $m, $rest, $iname, $max_op);
  open($fh, '<', $fname) or die("Cannot open $fname, error $!");
  my $ln = 0;
  my $add = defined($opt_f) || defined($opt_b) || defined($opt_B) || defined($opt_i) || defined($opt_a) || defined($opt_o);
  my $max_mask = 0;
  while( $str = <$fh> ) {
    chomp $str;
    $ln++;
    next if ( $str !~ /^\d+\s+(.*)$/ );
    $str = $1;
    # opcode name and tail starts at 48
    my $tail = substr($str, 48);
    $iname = (split /\t/, $tail)[0];
    # skip _mma.warpgroup
    next if ( defined($opt_w) && ( $iname eq '_mma.warpgroup' or $iname eq '_mma' ) );
    $g_ins{$iname}++; # insert into g_ins
    # make mask
    $m = substr($str, 0, 47);
    my @tmp = map { hex $_; } split /\s+/, $m;
    # calc bitlen
    my $blen = bcnt(\@tmp);
    if ( $blen > $max_mask ) {
      $max_mask = $blen;
      $max_op = [ $ln, \@tmp, $tail ];
    }
    # add to g_ops if needed
    push @g_ops, [ $ln, \@tmp, $tail, $iname ] if ( $add );
    # or with total mask
    foreach my $mi ( 0 .. 15 ) {
      $mask[$mi] |= $tmp[$mi] if $tmp[$mi];
    }
  }
  close $fh;
  # dump results
  dump_mask(\@mask);
  printf("length of mask %d\n", bcnt(\@mask));
  printf("%d uniq ins\n", scalar keys %g_ins);
  if ( $max_op ) {
    printf("longest mask: %d\n", $max_mask);
    printf("at %d %s\n", $max_op->[0], $max_op->[2]);
  }
}

# read ptx.txt and check in g_ins every instruction
sub apply_ptx
{
  my $fname = shift;
  my($fh, $str);
  open($fh, '<', $fname) or die("Cannot open $fname, error $!");
  my $ln = 0;
  # some stat
  my $found = 0;
  my $bad = 0;
  my %in_cicc;
OUTER:
  while( $str = <$fh> ) {
    chomp $str;
    $ln++;
    my $last;
    # filter some trash
    next if ( $str eq ';' || $str eq '' );
    next if ( $str =~ /\{/ );
    $str =~ s/^\s+//;
    my $body = ( split /\s+/, $str )[0];
    next unless $body;
    $body =~ s/\s+$//;
    $body =~ s/;$//;
    next if ( $body =~ /\.reg/ );
    next if ( $body eq ',' or $body eq '}' or $body eq ';' or $body eq '//' or $body eq '@' or $body eq '#' or $body eq '' );
    next if ( $body =~ /^\}/ );
    next if ( $body =~ /^\]/ );
    next if ( $body =~ /^\.row/ );
    next if ( $body =~ /^\.col/ );
    next if ( $body =~ /^\.v2/ );
    next if ( $body =~ /^\.v4/ );
    next if ( $body =~ /^\.pred/ );
    next if ( $body =~ /^\.param/ );
    # split on '.'
    my $need_dot = 0;
    if ( $body =~ /^\./ ) {
      $need_dot = 1;
      $body = substr($body, 1);
    }
    my @chain = split /\./, $body;
    my $name = $chain[0];
    $name = '.' . $name if ( $need_dot );
    if ( exists $g_ins{$name} ) {
      if ( defined $opt_U ) {
        $last = $name;
      } else {
        $found++;
        next;
      }
    }
    # try all
    foreach my $i ( 1 .. scalar(@chain) - 1 ) {
      $name = $name . '.' . $chain[$i];
      if ( exists $g_ins{$name} ) {
        if ( defined $opt_U ) {
          $last = $name;
        } else {
          $found++;
          next OUTER;
        }
      }
    }
    if ( defined($opt_U) && $last ) {
      $in_cicc{$last}++;
      $found++;
      next OUTER;
    }
    # finally dump
    printf("line %d: %s\n", $ln, $str) unless( defined $opt_U );
    $bad++;
  }
  close $fh;
  printf("found %d bad %d\n", $found, $bad);
  if ( defined $opt_U ) {
    foreach my $o ( sort keys %g_ins ) {
      next if ( exists $in_cicc{ $o } );
      printf("%s\n", $o);
    }
  }
}

=pod
=begin text
table for bits decoding
 01 - 0
 02 - 1
 04 - 2
 08 - 3
 10 - 4
 20 - 5
 40 - 6
 80 - 7

good example is suld.b.geom{.cop}.vec.dtype.clamp

there geom is tab282E760 - also used in tex/tld4/tex.base/tex.level/tex.grad/sured.b

cop is tab282E960

vec also used in many instructions like ld/st/atom/red/

clamp is tab282E480 - it's position is known 9:5
                            |
suld.b has mask             V
20 00 00 00 20 04 0E 00 00 20 00 00 00 00 00 00

tld4 & suld.b & sured.b gives mask:
00 00 00 00 00 00 04 00 00 00 00 00 00 00 00 00

so geom must have index 6:2

mask for suld/ld/st to include vec is 4:5

As you cab see the hypothesis that indices should preserve order is not confirmed - here we have
geom at 6.2
and vec 4.5, but in suld geom must precede vec

5:2 then must be cop

sured.p & sured.b
00 00 00 00 20 00 0C 00 00 24 00 00 00 00 00 00 sured.p	is0	B32
00 00 00 00 20 00 0C 00 00 24 00 00 00 00 00 00 sured.p	is0	B64
00 00 00 00 20 00 0C 00 00 24 00 00 00 00 00 00 sured.b	is0	B32
00 00 00 00 20 00 0C 00 00 22 00 00 00 00 00 00 sured.b	is0	I[32|64]


multimem.red multimem.ld_reduce cp.reduce
00 00 00 00 80 08 01 00 00 00 00 00 00 00 00 00

=end text
=cut

# key idx * 8 + shift, value - name of table in tabs sub-dir without .txt extension
my %gk_tabs = (
# idx 0
  0 => 'tab282FC80', # BoolOp
  2 => 'tab282FBC0', # CmpOp
  3 => 'no_atexit',  # index might as well be 4
  5 => 'tab282F560',
  7 => 'approx',
  1 * 8 + 0 => 'relu', # cvt/fma/min/max
  1 * 8 + 1 => 'ftz',
  1 * 8 + 2 => 'noftz',
  1 * 8 + 3 => 'satfinite', # cvt with floats only
  1 * 8 + 4 => 'tab282F560', # int types like s32
  2 * 8 + 0 => 'block_scale', # mma & tcgen05.mma
  2 * 8 + 3 => 'tab282FA00', # kind for tcgen05.mma & tcgen05.mma.ws
  2 * 8 + 4 => 'tab282F8E0', # scale_vectorsize
  2 * 8 + 1 => 'ashift',
  2 * 8 + 2 => 'tab282F900', # .collector_usage
  3 * 8 + 0 => 'sat',
  3 * 8 + 1 => 'cc',
  3 * 8 + 2 => 'shiftamt',
  3 * 8 + 3 => 'tab282E820', # (f)rnd
  3 * 8 + 7 => 'uni',
  4 * 8 + 5 => 'vec',
  4 * 8 + 6 => 'tab282DFE0', # mov.type & cvt 01 - type with .pred
  4 * 8 + 7 => 'tab282EC40', # scope/ss like .gpu .cluster
  5 * 8 + 0 => 'tab282E6A0', # load_mode
  5 * 8 + 1 => 'testp',
  5 * 8 + 2 => 'tab282E960', # .cop
  5 * 8 + 3 => 'tab282E900', # .sem + barrier.cluster
  5 * 8 + 4 => 'tab282E8D0', # .to_proxykind::from_proxykind = {.tensormap::generic}
  5 * 8 + 5 => 'mmio',       # ld/st/red.async
  5 * 8 + 6 => 'tab282EC20', # cache level like .l1
  5 * 8 + 7 => 'tab282EB80', # eviction, since v7.4 also for ld/st/prefetch
  6 * 8 + 3 => 'tab282E760', # geom
  6 * 8 + 4 => 'tab282E720', # .dim = { .1d, .2d, .3d, .4d, .5d }
  6 * 8 + 5 => 'b1024',      # ? oficially only for tensormap.replace
  6 * 8 + 7 => 'tab282E620', # cta_group
  7 * 8 + 1 => 'tab282E5E0', # src_fmt/dst_fmt for tcgen05.cp & ldmatrix
  7 * 8 + 0 => 'tab282E600', # multicast for tcgen05.cp
  7 * 8 + 4 => 'multicast',  # tcgen05.commit & cp.async.bulk.tensor
  8 * 8 + 0 => 'tab282E520', # .completion_mechanism
  8 * 8 + 3 => 'tab282E4C0', # .comp = { .r, .g, .b, .a };
  8 * 8 + 4 => 'squery',     # common for txq & suq
  8 * 8 + 6 => 'tquery',     # tlquery is first 3 row from tquery
  8 * 8 + 7 => 'tab282E360', # vote mode
  9 * 8 + 0 => 'tab282FC80', # redOp
  9 * 8 + 2 => 'tab282F3A0', # redOp with popc
  9 * 8 + 5 => 'tab282E480', # clamp
  9 * 8 + 6 => 'po',
  9 * 8 + 7 => 'tab282E460', # scale = { .shr7, .shr15 }
 10 * 8 + 0 => 'prmt',
 10 * 8 + 1 => 'tab282E400', # bfly
 10 * 8 + 2 => 'down', # for tcgen05.shift
 10 * 8 + 5 => 'sync', # from setmaxnreg.inc
 10 * 8 + 6 => 'noinc',
 11 * 8 + 0 => 'tab282F800', # isspacep/cvta/cvt.to
 11 * 8 + 2 => 'aligned',
 11 * 8 + 6 => 'tab282ECE0', # shapes like m16n
 12 * 8 + 1 => 'trans',
 12 * 8 + 2 => 'tab282ECE0', # shape3/shape4
 12 * 8 + 5 => 'tab282F7A0', # num for tcgen05.ld/tcgen05.st
 13 * 8 + 6 => 'xorsign',
 14 * 8 + 6 => 'abs',
 15 * 8 + 1 => 'tab282F510', # alias for fence.proxy & membar.proxy
 15 * 8 + 4 => 'tab282F4E0', # sync_restrict::shared:*
 15 * 8 + 5 => 'tab282F460', # launch_dependents
 15 * 8 + 6 => 'tab282F4A0', # get_first_ctaid{::dimension}
 15 * 8 + 7 => 'read',
);

use constant TabsDir => 'tabs/';
use constant TabLen => 96;

sub tab_fname { TabsDir . shift . '.txt'; }

sub read_tab {
  my $tn = shift;
  my $fn = tab_fname($tn);
  my(@res, $fh, $str);
  open($fh, '<', $fn) or die("cannot open $fn for table $tn, error $!");
  my $ln = -1;
  while( $str = <$fh> ) {
    chomp $str;
    $ln++;
    # skip first empty line
    next if ( !$ln && $str eq '' );
    push @res, $str;
  }
  close $fh;
  return \@res;
}

# args: table name, prefix string
sub read_dump_tab {
  my($tn, $pfx) = @_;
  my $ar = read_tab($tn);
  next unless defined $ar;
  foreach my $str ( @$ar ) { printf("%s%s\n", $pfx, $str); }
}

# tabs cache, key - name, value - array of rows
my %g_tcache;

# return array of table rows
# if not in cache - read it
# args: name of table
sub get_trows {
  my $tn = shift;
  return $g_tcache{$tn} if ( exists $g_tcache{$tn} );
  my $ar = read_tab($tn);
  return unless defined $ar;
  $g_tcache{$tn} = $ar;
  $ar;
}

sub make_trow {
  my $row = shift;
  my $res = $row->[0];
  my $rlen = scalar @$row;
  my $len = length($res);
  for ( my $i = 1; $i < $rlen && $len < TabLen; ++$i ) {
    $res .= ' ' . $row->[$i];
    $len = length($res);
  }
  $res;
}

# dump tabs
# args: bitmask, prefix string
sub dump_mtabs {
  my($mask, $pfx) = @_;
  my $res = 0;
  my $idx = 0;
  foreach my $m ( @$mask ) {
    foreach my $i ( 0 .. 7 ) {
      next unless ( $m & (1 << $i) );
      my $tab_idx = $idx * 8 + $i;
      next unless exists $gk_tabs{$tab_idx};
      my $ar = get_trows($gk_tabs{$tab_idx});
      next unless defined($ar);
      # dump them
      printf("%s%2.2d:%d: %s\n", $pfx, $idx, $i, make_trow($ar));
      ++$res;
    }
    ++$idx;
  }
  $res;
}

sub v_tabs
{
  my($str, $dh, %tabs);
  opendir($dh, TabsDir) or die("cannot open tabs sub-dir, error $!");
  while($str = readdir($dh)) {
    next if ( $str eq '.' || $str eq '..' );
    if ( $str !~ /^(.*)\.txt$/ ) {
      printf("bad tab %s\n", $str);
      next;
    }
    $tabs{$1}++;
  }
  closedir($dh);
  # traverse gk_tabs values
  foreach my $t ( values %gk_tabs ) {
    if ( exists $tabs{$t} ) {
      delete $tabs{$t};
    } else {
      printf("unknown tab %s\n", $t);
    }
  }
  my @rem = sort { $a cmp $b } keys %tabs;
  return unless( scalar @rem );
  printf("%d unused tabs:\n", scalar @rem);
  foreach my $tn ( @rem ) {
    printf(" %s\n", $tn);
    read_dump_tab($tn, '   ') if ( $tn =~ /^tab/ );
  }
}

# main
my $status = getopts("Bb:adfikotUw");
usage() if ( !$status );

read_ops2('ptx_ops2.txt');
v_tabs() if ( defined $opt_t );
# build neg known mask
if ( defined $opt_k ) {
  my @k = ( 0 ) x 16;
  foreach my $kt ( keys %gk_tabs ) {
    my $idx = $kt >> 3;
    my $mask = 1 << ($kt & 0x7);
    $k[$idx] |= $mask;
  }
  # dump it
  printf("size of known mask %d\n", bcnt(\@k));
  printf("known mask:    ");
  dump_mask(\@k);
  # make negative
  my @kn = map { (~$_) & 0xff; } @k;
  printf("negative mask: ");
  dump_mask(\@kn);
  $gk_neg = \@kn;
}

my %ins;
my $ga = sub {
  my $what = shift;
  $ins{$_} = 1 foreach @ARGV;
  die("no ins for $what") unless( scalar keys %ins );
};
if ( defined $opt_i ) {
 $ga->('-i');
 filter_ins(\%ins);
} elsif ( defined $opt_a ) {
  $ga->('-a');
  filter_and(\%ins);
} elsif ( defined $opt_o ) {
  $ga->('-o');
  filter_or(\%ins);
} elsif ( defined $opt_b ) {
 # parse and check -b option
 die("bad -b option") if ( $opt_b !~ /^(\d+):(\d)$/ );
 my $idx = int($1);
 die("bad idx") if ( $idx > 15 );
 my $sh = int($2);
 die("bad shift") if ( $sh > 7 );
 try_mask($idx, $sh);
} elsif ( defined $opt_B ) {
  my %mh;
   # parse list of args for -B
  foreach my $ba ( @ARGV ) {
     die("bad -B option $ba") if ( $ba !~ /^(\d+):(\d)$/ );
     my $idx = int($1);
     die("bad idx in $ba") if ( $idx > 15 );
     my $sh = int($2);
     die("bad shift in $ba") if ( $sh > 7 );
     $mh{$idx} |= 1 << $sh;
  }
  die("no args for -B") unless( scalar keys %mh );
  try_maskB(\%mh);
} elsif ( defined $opt_f ) {
  do_freq();
} else { apply_ptx('ptx.txt'); }
