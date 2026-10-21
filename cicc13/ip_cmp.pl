#!perl -w
# script to compare ptx instructions from 13.4 with old format implemented in iptx.pl - in file ptx_ops2.txt
# borrowed some functions like bcnt & read_ops2
use strict;
use warnings;
use Getopt::Std;
use Data::Dumper;

# options
use vars qw/$opt_f $opt_v $opt_w/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] ptx_ops
 Options:
   -f - compare forms
   -v - verbose mode
   -w - skip wmma & friends
EOF
  exit(8);
}

use constant MaskSize => 19;

# boring mask logic
sub _dump_mask {
  my $mr = shift;
  foreach my $mi ( @$mr ) {
    printf("%2.2X ", $mi);
  }
}

sub dump_mask
{
  _dump_mask(@_);
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

sub read_ops
{
  my($fname, $hri) = @_;
  my @mask = ( 0 ) x (MaskSize + 1);
  my($fh, $str, $m, $rest, $iname, $max_op);
  open($fh, '<', $fname) or die("Cannot open $fname, error $!");
  my $ln = 0;
  my $res_cnt = 0;
  my $max_mask = 0;
  while( $str = <$fh> ) {
    chomp $str;
    $ln++;
    next if ( $str !~ /^\d+\s+(.*)$/ );
    $str = $1;
    # opcode name and tail starts at 48
    my $tail = substr($str, 60);
    $iname = (split /\t/, $tail)[0];
    # skip _mma.warpgroup
    next if ( defined($opt_w) && ( $iname eq '_mma.warpgroup' or $iname eq '_mma' ) );
    # make mask
    $m = substr($str, 0, 59);
    my @tmp = map { hex $_; } split /\s+/, $m;
    # calc bitlen
    my $blen = bcnt(\@tmp);
    if ( $blen > $max_mask ) {
      $max_mask = $blen;
      $max_op = [ $ln, \@tmp, $tail ];
    }
    # make array with mask & tail
    ++$res_cnt;
    my $curr_i = [ $ln, \@tmp, $tail, $iname ];
    if ( exists $hri->{$iname} ) {
      my $ar = $hri->{$iname};
      push @$ar, $curr_i;
    } else {
      $hri->{$iname} = [ $curr_i ];
    }
    # or with total mask
    foreach my $mi ( 0 .. MaskSize ) {
      $mask[$mi] |= $tmp[$mi] if $tmp[$mi];
    }
  }
  close $fh;
  # dump results
  if ( defined $opt_v ) {
    printf("%d uniq ins\n", scalar keys %$hri);
    dump_mask(\@mask);
    printf("length of mask %d\n", bcnt(\@mask));
    if ( $max_op ) {
      printf("longest mask: %d ", $max_mask);
      printf("at %d %s\n", $max_op->[0], $max_op->[2]);
    }
  }
  $res_cnt;
}

sub cmp_forms {
  my($oar, $nar) = @_;
  # to avoid quadratic lookup for each new form in nar against all forms in oar lets build dictionary of tails from oar
  my %tails;
  $tails{$_->[2]}++ for @$oar;
  foreach my $f ( @$nar ) {
    next if ( exists $tails{ $f->[2] } );
    printf(" line %d: %s\n", $f->[0], $f->[2]);
  }
}

# compare new instructions in nh missed in oh
sub cmp_ops {
 my($oh, $nh) = @_;
 my %visited;
 my $latch = 0;
 my $added = 0;
 my $removed = 0;
 my $n_forms = 0;
 # grab new instructions
 foreach my $iname ( sort keys %$nh ) {
   if ( exists $oh->{$iname} ) {
     $visited{$iname}++;
     if ( defined $opt_f ) {
       # check if we have the same number of forms
       my $old_ar = $oh->{$iname};
       my $new_ar = $nh->{$iname};
       next if ( scalar(@$old_ar) == scalar(@$new_ar) );
       printf("%s: %d old forms, %d new\n", $iname, scalar(@$old_ar), scalar(@$new_ar));
       $n_forms += scalar(@$new_ar) - scalar(@$old_ar);
       cmp_forms($old_ar, $new_ar);
     }
     next;
   }
   next if ( defined $opt_f );
   printf("--- new:\n") if ( !$latch++ );
   my $ar = $nh->{$iname};
   my $f_size = scalar @$ar;
   $n_forms += $f_size;
   printf(" %s (%d forms)\n", $iname, $f_size);
   ++$added;
 }
 printf("Total %d new instructions (%d forms)\n", $added, $n_forms) if ( $n_forms );
 # check removed
 $latch++;
 foreach my $iname ( sort keys %$oh ) {
   next if ( exists $visited{$iname} );
   ++$removed;
   printf("--- rewmoved:\n") if ( !$latch++ );
   printf(" %s\n", $iname);
  }
 printf("Total %d removed instructions\n", $removed) if $removed;
}

# main
my $status = getopts("fvw");
usage() if ( !$status );
if ( 1 != scalar @ARGV ) {
  printf("where is file name?\n");
  exit(8);
}
my(%old, %n);
my $ocnt = read_ops('ptx_ops2.txt', \%old);
my $ncnt = read_ops($ARGV[0], \%n);
printf("--- %d lines\n", $ncnt - $ocnt);
cmp_ops(\%old, \%n);