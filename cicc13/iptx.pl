#!perl -w
# try intersect ptx.txt from cicc with ptx_ops2.txt extracted from ptxas
# -f for mask frequency analysis
use strict;
use warnings;
use Getopt::Std;
use Data::Dumper;

# options
use vars qw/$opt_a $opt_b $opt_f $opt_i $opt_o/;

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
EOF
  exit(8);
}

# instr names from ptx_ops2.txt
my %g_ins;
# array of ops, each element is [ line number, mask array, rest of op, op name ]
my @g_ops;

sub do_freq
{
  foreach my $i ( 0 .. 15 ) {
    foreach my $bi ( 0 .. 7 ) {
      my $mask = 1 << $bi;
      my $latch = 0;
      foreach my $op ( @g_ops ) {
        my $ar = $op->[1];
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
    my $ar = $op->[1];
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
  my($fh, $str, $m, $rest, $iname);
  open($fh, '<', $fname) or die("Cannot open $fname, error $!");
  my $ln = 0;
  my $add = defined($opt_f) || defined($opt_b) || defined($opt_i) || defined($opt_a) || defined($opt_o);
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

# main
my $status = getopts("b:afio");
usage() if ( !$status );

read_ops2('ptx_ops2.txt');
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
