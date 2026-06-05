#!perl -w
# hopeless attempt to re lex generated shitcode
#  what is looks like in source:
#     yy_current_state = (yy_start); <-- initial value 1
# yy_match:
#     do
#     {
#          YY_CHAR yy_c = yy_ec[YY_SC_TO_UI(*yy_cp)] ;
#          if ( yy_accept[yy_current_state] )
#          {
#             (yy_last_accepting_state) = yy_current_state;
#             (yy_last_accepting_cpos) = yy_cp;
#          }
#          while ( yy_chk[yy_base[yy_current_state] + yy_c] != yy_current_state )
#          {
#             yy_current_state = (int) yy_def[yy_current_state];
#             if ( yy_current_state >= 5165 ) <-- sizeof(yy_accept)
#                 yy_c = yy_meta[yy_c];
#          }
#          yy_current_state = yy_nxt[yy_base[yy_current_state] + yy_c];
#          ++yy_cp;
#      } while ( yy_base[yy_current_state] != 9873 ); <-- sizeof(yy_nxt) - sizeof(yy_meta)
#
# yy_find_action:
#      yy_act = yy_accept[yy_current_state];
#      if ( yy_act == 0 )
#      { /* have to back up */
#         yy_cp = (yy_last_accepting_cpos);
#         yy_current_state = (yy_last_accepting_state);
#         yy_act = yy_accept[yy_current_state];
#      }
# so we need array
#  yy_ec - conversion for symbols to yy_c
#  yy_accept
#  yy_base
#  yy_def
#  yy_chk
#  yy_nxt
#  yy_meta
use strict;
use warnings;
use Data::Dumper;
use Getopt::Std;
# for given
use feature qw( switch );
no warnings qw( experimental::smartmatch );

# options
use vars qw/$opt_c $opt_d $opt_t $opt_v/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] action(s)
 Options:
  -c - name of lex generated file (default lex.yy.c)
  -d - debug mode
  -t - test mode - read simgle string and try to apply lexer on it
  -v - verbose mode
EOF
  exit(8);
}

# globals
my(@yy_ec, @yy_accept, @yy_base, @yy_def, @yy_chk, @yy_nxt, @yy_meta);
# max value in yy_ecc & hash of uniq symbols
my($yy_ec_max, %yy_syms);
# terminal indexes in yy_base
my %yy_bterm;
# terminal indexes in yy_nxt (reffered to indexes in yy_bterm)
my %yy_nxtf;

sub endm() { scalar(@yy_nxt) - scalar(@yy_meta); }

sub dump_arr
{
  my $ar = shift;
  printf(" %d", $_) foreach @$ar;
  printf("\n");
}

sub find_nxtf
{
  my $size = scalar @yy_nxt;
  my $res = 0;
  foreach my $i ( 0 .. $size - 1 ) {
    my $s = $yy_nxt[$i];
    next unless exists($yy_bterm{$s});
    $yy_nxtf{$i}++;
    ++$res;
  }
  if ( $res && $opt_v ) {
    printf("nxt %d:", $res);
    my @k = sort { $a <=> $b } keys %yy_nxtf;
    dump_arr(\@k);
  }
  $res;
}

# lets naively assume that we known prev yy_current_state and want to find yy_nxt[yy_base[yy_current_state] + yy_c]
# (here yy_c is unknown) such that yy_base[found index] is term - so in yy_bterm
sub try_naive
{
  my $prev = shift;
  my $size = scalar @yy_nxt;
  my $b = $yy_base[$prev];
  for ( my $i = 0; $i <= $yy_ec_max && $b + $i < $size; ++$i ) {
    next unless exists($yy_syms{$i});
    my $v = $yy_nxt[$b + $i];
    next unless exists $yy_bterm{$v};
    printf("b %d uu_c %d %c\n", $b + $i, $i, $yy_syms{$i} );
  }
}

sub naive_brute
{
  my $size = scalar @yy_nxt;
  my $upto = scalar @yy_base;
  for ( my $idx = 0; $idx < $upto; ++$idx ) {
     my $b = $yy_base[$idx];
     for ( my $i = 0; $i <= $yy_ec_max && $b + $i < $size; ++$i ) {
       next unless exists($yy_syms{$i});
       next unless $yy_accept[$b + $i];
       my $v = $yy_nxt[$b + $i];
       next unless exists $yy_bterm{$v};
       printf("idx %d b %d uu_c %d %c\n", $idx, $b + $i, $i, $yy_syms{$i} );
     }
  }
}

sub find_bterms
{
  my $end = endm();
  my $size = scalar @yy_base;
  my $res = 0;
  foreach my $i ( 0 .. $size - 1 ) {
    my $s = $yy_base[$i];
    next if ( $s != $end );
    $yy_bterm{$i}++;
    ++$res;
  }
  if ( $res && $opt_v ) {
    printf("base %d:", $res);
    my @k = sort { $a <=> $b } keys %yy_bterm;
    dump_arr(\@k);
  }
  find_nxtf() if $res;
  $res;
}

sub process_ec {
  find_bterms();
  $yy_ec_max = 0;
  my $size = scalar @yy_ec;
  my %cache;
  foreach my $i ( 0 .. $size - 1 ) {
    my $s = $yy_ec[$i];
    next unless $s;
    if ( exists $cache{$s} ) { # not uniq
      $cache{$s}->[0]++;
    } else {
      $cache{$s} = [ 0, $i ];
    }
    $yy_ec_max = $s if ( $s > $yy_ec_max );
  }
  # fill yy_syms
  foreach my $k ( keys %cache ) {
    my $ar = $cache{$k};
    next if ( $ar->[0] );
    $yy_syms{$k} = $ar->[1];
  }
  # verbose mode
  if ( $opt_v ) {
    printf("ec_max %d, uniq symbols %d\n", $yy_ec_max, scalar keys %yy_syms);
    print Dumper(\%yy_syms) if ( $opt_d );
  }
}

sub scan_accept {
  my $what = shift;
  my $size = scalar @yy_accept;
  my @res;
  foreach my $i ( 0 .. $size - 1 ) {
    push @res, $i if ( $yy_accept[$i] == $what );
  }
  my $fsize = scalar @res;
  if ( defined $opt_v ) {
    if ( $fsize ) {
      printf("found %d:", $what);
      dump_arr(\@res);
    } else {
      printf("%d not found in yy_accept\n", $what);
    }
  }
  return $fsize ? \@res : undef;
}

# parse logic - very boring
sub yy_by_name {
  my $name = shift;
  given($name) {
    when('yy_ec')     { return \@yy_ec; }
    when('yy_accept') { return \@yy_accept; }
    when('yy_base')   { return \@yy_base; }
    when('yy_def')    { return \@yy_def; }
    when('yy_chk')    { return \@yy_chk; }
    when('yy_nxt')    { return \@yy_nxt; }
    when('yy_meta')   { return \@yy_meta; }
  }
  undef;
}

sub read_array {
  my($fh, $ar) = @_;
  my($str);
  while( $str = <$fh> ) {
    chomp $str;
    $str =~ s/^\s+//;
    $str =~ s/^\{\s+//;
    next if ( $str eq '' );
    last if ( $str =~ /^}/ );
    my @spl = split(/,\s*/, $str);
    foreach my $i ( @spl ) {
      next if ( $i eq '' );
      push @$ar, int($i);
    }
  }
  return scalar @$ar;
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
    $ar = yy_by_name($name);
    unless( defined $ar ) {
      printf("unknown table %s\n", $name);
      next;
    }
    my $rsize = read_array($fh, $ar);
    if ( $rsize != $size ) {
      printf("read of %s failed: %d vs %d\n", $name, $rsize, $size);
    }
  }
  close($fh);
}

# test
sub ltest
{
  my $str = shift;
  my $yy_start = 1;
  my $yy_last_accepting_state;
  my $yy_last_accepting_cpos = 0;
  my @ar = split //, $str;
  push @ar, 0;
  my $ar_len = scalar @ar;
  my $yy_pos = 0;
  while( $yy_pos < $ar_len ) {
     my $yy_current_state = $yy_start;
     do
     {
        my $sym = $ar[$yy_pos];
        my $yy_c = $yy_ec[ord($sym)];
printf("pos %d yy_cp %s c %d\n", $yy_pos, $sym ? $sym : ' ', $yy_c);
          if ( $yy_accept[$yy_current_state] )
          {
printf("accept curr state %d\n", $yy_current_state);
             $yy_last_accepting_state = $yy_current_state;
             $yy_last_accepting_cpos = $yy_pos;
          }
          while ( $yy_chk[$yy_base[$yy_current_state] + $yy_c] != $yy_current_state )
          {
             $yy_current_state = $yy_def[$yy_current_state];
printf("state %d\n", $yy_current_state);
             if ( $yy_current_state >= scalar(@yy_accept) ) {
                 $yy_c = $yy_meta[$yy_c];
printf("new yy_c %d\n", $yy_c);
             }
          }
          $yy_current_state = $yy_nxt[$yy_base[$yy_current_state] + $yy_c];
          $yy_pos++;
printf("pos %d curr_state %d base %d\n", $yy_pos, $yy_current_state, $yy_base[$yy_current_state]);
      } while ( $yy_base[$yy_current_state] != endm() );

yy_find_action:
      my $yy_act = $yy_accept[$yy_current_state];
      if ( !$yy_act )
      { # have to back up
printf("--- backup to %d pos\n", $yy_last_accepting_cpos);
         $yy_pos = $yy_last_accepting_cpos;
         $yy_current_state = $yy_last_accepting_state;
         $yy_act = $yy_accept[$yy_current_state];
      }
printf("--- yy_act %d state %d\n", $yy_act, $yy_current_state);
 # switch
      if ( !$yy_act ) {
        $yy_pos = $yy_last_accepting_cpos;
        $yy_current_state = $yy_last_accepting_state;
        goto yy_find_action;
      }
      printf("--- curr_state %d act %d pos %d\n", $yy_current_state, $yy_act, $yy_pos);
#      if ( 1 == $yy_act ) {
#        $yy_pos = 0;
#        $yy_start = 1;
#        next;
#      }
      $yy_start = 1;
   }
}

# main
my $status = getopts("dvtc:");
usage() if ( !$status );
$DefFName = $opt_c if defined($opt_c);
parse();
process_ec();
if ( defined $opt_t ) {
  my $str = <>;
  chomp $str;
  ltest($str);
  exit;
}
naive_brute();
die("where is actions?") if ( -1 == $#ARGV );
foreach my $a ( @ARGV ) {
  my $ai = int($a);
  unless($ai) {
    printf("bad action %s\n", $a);
    next;
  }
  my $s1 = scan_accept($ai);
  next unless defined($s1);
  # check than yy_base[] == endm
  my @next;
  foreach my $s ( @$s1 ) {
printf("base[%d] %d\n", $s, $yy_base[$s]);
    push @next, $s if ( $yy_base[$s] == endm() );
  }
  # check result
  my $nsize = scalar @next;
  if ( $opt_v ) {
     if ( $nsize ) {
       printf("next %d:", $nsize);
       dump_arr(\@next);
     } else {
       printf("next is empty\n");
     }
  }
  unless($nsize) {
    try_naive($_) for @$s1;
  }
}