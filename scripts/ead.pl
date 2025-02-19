#!perl -w
# some nvdisasm encoding analysis
use strict;
use warnings;

# ENCODING WIDTH
my $g_size;

# for masks we need 2 map - first with name as key, second as mask
# both contains as value array where
# [0] - name
# [1] - mask
# [2] - size of significant bits
# [3] - list for decoding
my(%g_mnames, %g_mmasks);

sub dump_decode
{
  my $d = shift;
  my $res = '';
  my $first;
  foreach ( @$d ) {
    if ( defined $first ) {
      $res .= sprintf("%d:%d ", $first, $_);
      undef $first;
    } else {
      $first = $_;
    }
  }
  chop $res;
  $res;
}


sub parse_mask
{
  my $str = shift;
  return 0 if ( $str !~ /^\s*(\S+)\s+\'([^\']+)\'/ );
  my $name = $1;
  my $v = lc($2);
  if ( length($2) != $g_size ) {
    printf("length of %s is %d\n", $name, length($2));
    return 0;
  }
  if ( exists $g_mnames{$name} ) {
    printf("%s duplicated\n", $name);
    return 0;
  }
  # check if we already have such mask
  if ( exists $g_mmasks{$v} ) {
    printf("%s has the same mask as %s (%s)\n", $name, $g_mmasks{$v}->[0], dump_decode( $g_mmasks{$v}->[3] ));
    $g_mnames{$name} = $g_mmasks{$v};
    return 1;
  }
  my @a = split //, $v;
  my $cnt = 0;
  foreach ( @a ) {
    $cnt++ if ( $_ eq 'x' );
  }
  if ( $cnt > 64 ) {
    printf("%s has too long value %d\n", $name, $cnt);
  }
  # split for decoding
  my @d;
  my $idx = -1;
  for ( my $i = 0; $i < $g_size; $i++ ) {
    if ( $a[$i] ne 'x' ) {
      if ( $idx != -1 ) {
        push @d, $idx; # start index
        push @d, $i - $idx; # length
        $idx = -1;
      }
      next;
    } else {
      $idx = $i if ( $idx == -1 );
    }
  }
  # last item
  if ( -1 != $idx ) {
    push @d, $idx; # start index
    push @d, $g_size - $idx; # length
  }
  # make new mask
  my @res = ( $name, $v, $cnt, \@d );
  $g_mnames{$name} = $g_mmasks{$v} = \@res;
  1;
}

# opcodes divided into 2 group - first start with some zero mask (longest if there are > 1) and second with longest opcode
# key is mask value is map of second type
my(%g_zero, %g_ops);
# format hash where key is opcode, value - array where
# [0] - class name
# [1] - name
# [2] - opcode
# [3] - encoding list

sub parse0b
{
  my $s = shift;
  my $res = 0;
  my @arr = split //, $s;
  for ( my $i = 0; $i < length($s); $i++ ) {
    if ( $arr[$i] eq '0' ) { $res <<= 1; }
    elsif ( $arr[$i] eq '1' ) { $res <<= 1; $res |= 1; }
  }
  return $res;
}

# main
if ( 1 == $#ARGV ) {
  printf("where is arg?\n");
  exit(5);
}

my($fh, $state, $str);
open($fh, '<', $ARGV[0]) or die("cannot open, error $!");
$state = 0;
my($cname, $has_prev, @op);
while( $str = <$fh> ) {
  chomp $str;
  if ( !$state ) {
    if ( $str =~ /ENCODING\s+WIDTH\s+(\d+)\s*\;/ ) {
       $g_size = int($1);
       $state = 1;
    }
    next;
  }
printf("%d %s\n", $state, $str);
  if ( $str =~ /CLASS\s+\"([^\"]+)\"/ ) {
    if ( $has_prev ) {
      printf("%s %s %X\n", $cname, $op[0], $op[1]);
    }
    $has_prev = 1;
    $cname = $1;
    $state = 2;
    next;
  }
  if ( $state == 1 ) {
    parse_mask($str); next;
  }
  if ( $state == 2 && $str =~ /OPCODES/ ) {
    $state = 3;
    next;
  }
  # parse opcode
  if ( $state == 3 && $str =~ /^\s*(\S+)\s*=\s*(\S+);/ ) {
    my $name = $1;
    my $value = parse0b($2);
    # skip pipe version
    next if ( $name =~ /_pipe/ );
    $op[0] = $name;
    $op[1] = $value;
  }
  # encoding
  if ( $str =~ /ENCODING/ ) {
    $state = 4;
    next;
  }
}
close $fh;