#!perl -w
# code for fast scan looks like
#   yy_cp = (yy_c_buf_p);
#   *yy_cp = (yy_hold_char);
#   yy_bp = yy_cp;
# ;; yy_start is 1 and seems that yy_start_state_list[1] is always 3
#   yy_current_state = yy_start_state_list[(yy_start)];
# yy_match:
#   {
#      const struct yy_trans_info *yy_trans_info;
#      YY_CHAR yy_c;
#
#      for ( yy_c = YY_SC_TO_UI(*yy_cp);
#              (yy_trans_info = &yy_current_state[yy_c])->yy_verify == yy_c;
#              yy_c = YY_SC_TO_UI(*++yy_cp) )
#       {
#          yy_current_state += yy_trans_info->yy_nxt;
#          if ( yy_current_state[-1].yy_nxt )
#          {
#             (yy_last_accepting_state) = yy_current_state;
#             (yy_last_accepting_cpos) = yy_cp;
#           }
#       }
# yy_find_action:
#       yy_act = yy_current_state[-1].yy_nxt;
#       YY_DO_BEFORE_ACTION;
# do_action:
#       switch ( yy_act )
#        { /* beginning of action switch */
#            case 0: /* must back up */
#            /* undo the effects of YY_DO_BEFORE_ACTION */
#            *yy_cp = (yy_hold_char);
#            yy_cp = (yy_last_accepting_cpos) + 1;
#            yy_current_state = (yy_last_accepting_state);
#             goto yy_find_action;

use strict;
use warnings;
use Data::Dumper;
use Getopt::Std;
use Storable;
# options
use vars qw/$opt_b $opt_c $opt_d $opt_t $opt_v/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] action(s)
 Options:
  -b - read blob
  -c - name of lex generated file (default lex.yy.c)
  -d - debug mode
  -t - test mode - read simgle string and try to apply lexer on it
  -v - verbose mode
EOF
  exit(8);
}

# globals - array of pairs [ 0 - yy_verify, 1 - yy_nxt ]
my @yy_trans;
my $upto;

sub read_tab
{
  my($fh, $ar) = @_;
  my $str;
  while ( $str = <$fh> ) {
    chomp $str;
    $str =~ s/^\s+//;
    $str =~ s/\s+$//;
    next if ( $str eq '{' || $str eq '' );
    # read pair of digits
    while ( $str =~ /\b(\d+)\s*,\s*(\d+)/g ) {
      push @$ar, [ int($1), int($2) ];
    }
    # closing };
    if ( $str =~ /};$/ ) {
      # print Dumper($ar) if ( defined $opt_d );
      return;
    }
  }
}

my $BlobFName = 'yy_state_list';

sub parse_blob
{
  my($fh, $str);
  open($fh, '<', $BlobFName) or die("cannot open $BlobFName, error $!");
  while($str = <$fh>) {
    chomp $str;
    if ( $str =~ /\b(\d+)\s*,\s*(\d+)/ ) {
      my $n1 = int($1);
      my $n2 = int($2);
      if ( $n2 > 0xf0000000 ) {
        $n2 = -( 1 + ((~$n2) & 0xffffffff) );
      }
      push @yy_trans, [ $n1 , $n2 ];
    }
  }
  if ( defined $opt_v ) {
    printf("readed %d\n", scalar @yy_trans);
  }
  $upto = scalar @yy_trans - 1;
  close($fh);
}

my $DefFName = 'lex.yy.c';

sub parse
{
  my($fh, $str, $ar);
  open($fh, '<', $DefFName) or die("cannot open $DefFName, error $!");
  while($str = <$fh>) {
    chomp $str;
    last if ( $str =~ /YY_INPUT/ );
    next if ( $str !~ /^static\s+const\s+.*\s+(\w+)\[(\d+)\]/ );
    my $size = int($2);
    my $name = $1;
    next if ( $name ne 'yy_transition' );
    read_tab($fh, \@yy_trans);
    my $rsize = scalar @yy_trans;
    printf("bad %s, %d vs readed %d\n", $name, $size, $rsize) if ( $size != $rsize );
    last; # in fast version there is single table
  }
  close($fh);
}

sub dump_pair
{
  my $idx = shift;
  printf("%d sym %d", $yy_trans[$idx]->[1], $yy_trans[$idx]->[0]);
  printf(" %s", chr($yy_trans[$idx]->[0])) if ( $yy_trans[$idx]->[0] );
  printf("\n");
}

sub dump_state
{
  my $idx = shift;
  printf("ycs %d ", $idx);
  dump_pair($idx);
}

# test of fast version
# yy_trans is array of pair [ 0 - yy_verify, 1 - yy_nxt]
sub ftest
{
  my $str = shift;
  my $yy_act;
  my $yy_trans_info;
  my $yy_last_accepting_state;
  my $yy_last_accepting_cpos = 0;
  my @ar = split //, $str;
  push @ar, 0;
  my $ar_len = scalar @ar;
  my $yy_pos = 0;
  while( $yy_pos < $ar_len ) {
    my $yy_c; # 0x84
    my $yy_current_state = defined($opt_b) ? 2: 3; # seems that yy_start_state_list[1] is always 3
dump_state($yy_current_state);
    for ( $yy_c = ord($ar[$yy_pos]); $yy_c && $yy_pos < $ar_len; $yy_c = ord($ar[++$yy_pos]) ) {
printf("pos %d %c ", $yy_pos, $yy_c); dump_state($yy_current_state);
      $yy_trans_info = $yy_trans[$yy_current_state + $yy_c]; # index in yy_trans
printf("trans "); dump_state($yy_current_state + $yy_c);
printf("idx %d, [ %d, %d]\n", $yy_current_state + $yy_c, $yy_trans_info->[0], $yy_trans_info->[1]);
      last if ( $yy_trans_info->[0] != $yy_c );
      $yy_current_state += $yy_trans_info->[1];
printf("new ycs "); dump_state($yy_current_state);
      if ( $yy_trans[$yy_current_state - 1]->[1] ) {
printf("store %d pos %d\n", $yy_current_state, $yy_pos);
        $yy_last_accepting_state = $yy_current_state;
        $yy_last_accepting_cpos = $yy_pos;
      }
    }
yy_find_action:
    $yy_act = $yy_trans[$yy_current_state - 1]->[1];
 printf("curr_state %d act %d, letter %d, %d\n", $yy_current_state, $yy_act, $yy_trans[$yy_current_state - 1]->[0], $yy_trans[$yy_current_state]->[0]);
    if ( !$yy_act ) {
      $yy_pos = $yy_last_accepting_cpos + 1;
      $yy_current_state = $yy_last_accepting_state;
      goto yy_find_action;
    }
 printf("-- %d\n", $yy_act);
  }
}

# brute-force logic
# key idx + nxt, value [ list of idx ]
my $dist = {};
# key num, value [ list of [ base, letter] ]
my $cache = {};

sub try_acts
{
  my $act = shift;
  my @res;
  my $idx = 0;
  if ( exists $cache->{$act} ) {
    return $cache->{$act};
  }
  printf("try_acts: %d failed\n", $act);
  foreach my $pair ( @yy_trans ) {
    if ( $pair->[1] == $act ) {
      printf("idx %d: ", $idx);
      printf("%c ", chr($pair->[0])) if ( $pair->[0] );
      printf("next %d", $yy_trans[$idx+1]->[1]);
      printf(" %c", chr($yy_trans[$idx+1]->[0])) if ( $yy_trans[$idx+1]->[0] );
      printf("\n");
      push @res, $idx + 1;
    }
    ++$idx;
  }
  return scalar(@res) ? \@res : undef;
}

sub calc_letters
{
  my $ar = shift;
  my $res = 0;
  foreach my $a ( @$ar ) {
    ++$res if ( $yy_trans[$a]->[0] );
  }
  $res;
}

sub fill_dist
{
  my $cnt = 0;
  foreach my $idx ( 1 .. $upto ) {
    # fill cache
    if ( $yy_trans[$idx]->[1] > 0 ) {
      my $v = $yy_trans[$idx]->[1];
      if ( exists $cache->{$v} ) {
        push @{ $cache->{$v} }, $idx;
      } else {
        $cache->{$v} = [ $idx ];
      }
    }
    # dist
    last if ( $upto - $idx < 128 );
    for my $i ( 32 .. 127 ) {
      my $next = $yy_trans[$idx + $i];
      next unless $next->[0];
      next unless $next->[0] != $i;
      my $d = $idx + $next->[1];
      next if ( $d < 2 );
      $cnt++;
      if ( exists $dist->{$d} ) {
        push @{ $dist->{$d} }, [ $idx, $i ];
      } else {
       $dist->{$d} = [ [ $idx, $i ] ];
      }
    }
  }
  printf("cache keys %d\n", scalar keys %$cache);
  if ( defined $opt_v ) {
    my $with_l = 0;
    printf("dist %d, uniq %d\n", $cnt, scalar keys %$dist);
    foreach my $i ( sort { $a <=> $b } keys %$dist ) {
      my $ar = $dist->{$i};
      my $l_cnt = calc_letters($ar);
      $with_l += $l_cnt;
      printf(" %d: %d letters %d\n", $i, scalar @$ar, $l_cnt);
    }
    printf("with letters %d\n", $with_l);
  }
}

sub td_dist
{
  my $idx = shift;
  unless( exists $dist->{$idx} ) {
    printf("td_dist: no %d\n", $idx);
    return;
  }
  my $ar = $dist->{$idx};
  printf("td_dist %d:\n", $idx);
  printf(" %d (%d)", $_->[0], $_->[1]) for ( @$ar );
  printf("\n");
}

# args: level, prev idx, [ array of letters ], hash with visited
sub rec_chain
{
  my($lvl, $old, $l, $hv) = @_;
  return 0 if ( $lvl > 15 );
  return unless exists $dist->{$old};
 printf("old %d letters %s\n", $old, join '', map { chr($_); } reverse @$l) if ( defined $opt_v );
  my $ar = $dist->{$old};
  foreach my $prev ( @$ar ) {
    next if ( exists $hv->{ $prev->[0] } );
    next if ( $prev->[0] < 2 );
    if ( 2 == $prev->[0] ) {
      printf("%d: %s\n", $old, join '', map { chr($_); } reverse @$l);
      return 1;
    } else {
      my @cl = @$l;
      push @cl, $prev->[1];
      my %lv = %$hv;
      $lv{ $prev->[0] }++;
      return if rec_chain($lvl + 1, $prev->[0], \@cl, \%lv);
    }
  }
  return 0;
}

sub try_prev
{
  my $old = shift;
  $old++;
  if ( exists $dist->{$old} ) {
    my $ar = $dist->{$old};
  printf("has dist: %d (%d)\n", $old, scalar @$ar);
    my %v;
    $v{$old}++;
    foreach my $prev ( @$ar ) {
      my @l = ( $prev->[1] );
      $v{ $prev->[0] }++;
      return if rec_chain( 0, $prev->[0], \@l, \%v );
      delete $v{ $prev->[0] };
    }
  }
  if ( exists $cache->{$old} ) {
  printf("has cache: %d\n", $old);
    return $cache->{$old};
  }
  undef;
}

sub dump_to
{
  my $idx = shift;
  my $ar = $dist->{$idx};
  foreach my $a ( @$ar ) {
    printf(" %d:", $a);
    printf(" CHAIN") if exists($dist->{$a});
    dump_pair($a);
  }
}

sub try_letter
{
  my $l = shift;
  printf("--- %s\n", chr($l));
  foreach my $idx ( 1 .. $upto ) {
    next if ( $l != $yy_trans[$idx]->[0]);
# next unless($yy_trans[$idx - $yy_trans[$idx]->[0]]->[0]);
    printf(" %d: %d %d\n", $idx, $idx - $yy_trans[$idx]->[0], $idx + $yy_trans[$idx]->[1]) if ( defined $opt_v );
    if ( exists $dist->{$idx} ) {
      printf("--> %d\n", $idx);
      dump_to($idx);
    }
# dump_pair($yy_trans[$idx]->[1]);
  }
}

# main
my $status = getopts("bdvtc:");
usage() if ( !$status );
if ( defined $opt_b ) {
  $BlobFName = $opt_c if defined($opt_c);
  parse_blob();
} else {
  $DefFName = $opt_c if defined($opt_c);
  parse();
}
if ( defined $opt_t ) {
  my $str = <>;
  chomp $str;
  ftest($str);
  exit;
}
if ( -f 'cached' ) {
 my $ret = retrieve('cached');
 $dist = $ret->[0];
 $cache = $ret->[1];
} else {
 fill_dist();
 store([$dist, $cache], 'cached');
}
# try_letter(ord('k'));
foreach my $rule ( @ARGV ) {
  my $r = int($rule);
printf("try %d\n", $r);
  my $ar = try_acts($r);
  next unless defined($ar);
  foreach my $l1 ( @$ar ) {
    printf("act at %d\n", $l1);
    td_dist($l1 + 1);
    my $l2 = try_prev($l1);
  }
}