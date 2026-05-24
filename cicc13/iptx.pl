#!perl -w
# try intersect ptx.txt from cicc with ptx_ops2.txt extracted from ptxas
# -f for mask frequency analysis
use strict;
use warnings;
use Getopt::Std;
use Data::Dumper;

# options
use vars qw/$opt_a $opt_b $opt_f $opt_i $opt_o $opt_k $opt_t/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] md.txt
 Options:
 -b idx:shift
 -f - mask frequency analysis
 -a ins1 ins2 ... - make and mask of instructions
 -i ins1 ins2 ... - make and mask of instructions - remained
 -o ins1 ins2 ... - make or mask of instructions - remained
 -k - exclude known
 -t - verify tabs and dump still unused
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
  foreach my $i ( 0 .. 15 ) {
    foreach my $bi ( 0 .. 7 ) {
      my $mask = 1 << $bi;
      my $latch = 0;
      foreach my $op ( @g_ops ) {
        my $ar = apply_k($op->[1]);
        next unless $ar;
        next unless( $ar->[$i] & $mask );
        unless($latch) {
          printf("idx %d bit %d:\n", $i, $bi);
          $latch++;
        }
        printf(" line %d: %s\n", $op->[0], $op->[2]);
      }
    }
  }
}

sub try_mask
{
  my($idx, $sh) = @_;
  my $mask = 1 << $sh;
  foreach my $op ( @g_ops ) {
    my $ar = apply_k($op->[1]);
    next unless($ar);
    next unless( $ar->[$idx] & $mask );
    printf(" line %d: %s\n", $op->[0], $op->[2]);
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
    dump_mask(\@res);
  }
  $found;
}

sub filter_or
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
  my $add = defined($opt_f) || defined($opt_b) || defined($opt_i) || defined($opt_a) || defined($opt_o);
  my $max_mask = 0;
  while( $str = <$fh> ) {
    chomp $str;
    $ln++;
    next if ( $str !~ /^\d+\s+(.*)$/ );
    $str = $1;
    # opcode name and tail starts at 48
    my $tail = substr($str, 48);
    $iname = (split /\t/, $tail)[0];
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
OUTER:
  while( $str = <$fh> ) {
    chomp $str;
    $ln++;
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
      $found++;
      next;
    }
    # try all
    foreach my $i ( 1 .. scalar(@chain) - 1 ) {
      $name = $name . '.' . $chain[$i];
      if ( exists $g_ins{$name} ) {
        $found++;
        next OUTER;
      }
    }
    # finally dump
    printf("line %d: %s\n", $ln, $str);
    $bad++;
  }
  close $fh;
  printf("found %d bad %d\n", $found, $bad);
}

# key idx * 8 + shift, value - name of table in tabs sub-dir
my %gk_tabs = (
# idx 0
  2 => 'tab282FBC0', # CmpOp
  5 => 'tab282F560',
  7 => 'approx',
  1 * 8 + 1 => 'ftz',
  3 * 8 + 0 => 'sat',
  3 * 8 + 1 => 'cc',
  3 * 8 + 2 => 'shiftamt',
  3 * 8 + 3 => 'tab282E820', # (f)rnd
  3 * 8 + 7 => 'uni',
  5 * 8 + 1 => 'testp',
  6 * 8 + 3 => 'tab282E760', # geom
  6 * 8 + 4 => 'tab282E720', # .dim = { .1d, .2d, .3d, .4d, .5d }
  8 * 8 + 3 => 'tab282E4C0', # .comp = { .r, .g, .b, .a };
  8 * 8 + 7 => 'tab282E360', # vote mode
  9 * 8 + 5 => 'tab282E480', # clamp
  9 * 8 + 6 => 'po',
  9 * 8 + 7 => 'tab282E460', # scale = { .shr7, .shr15 }
  10 * 8 + 0 => 'prmt',
  10 * 8 + 1 => 'tab282E400', # bfly
  10 * 8 + 5 => 'sync', # from setmaxnreg.inc
  10 * 8 + 6 => 'noinc',
  11 * 8 + 2 => 'aligned',
  13 * 8 + 6 => 'xorsign',
  14 * 8 + 6 => 'abs',
  15 * 8 + 5 => 'tab282F460', # launch_dependents
  15 * 8 + 6 => 'tab282F4A0', # is_canceled
);

sub v_tabs
{
  my($str, $dh, %tabs);
  opendir($dh, 'tabs/') or die("cannot open tabs sub-dir, error $!");
  while($str = readdir($dh)) {
    next if ( $str eq '.' || $str eq '..' );
    next if ( $str !~ /^(.*)\.txt$/ );
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
  printf(" %s\n", $_) for ( @rem );
}

# main
my $status = getopts("b:afikot");
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
 die("bad shoft") if ( $sh > 7 );
 try_mask($idx, $sh);
} elsif ( defined $opt_f ) {
  do_freq();
} else { apply_ptx('ptx.txt'); }
