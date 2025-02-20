#!perl -w
# some nvdisasm encoding analysis
use strict;
use warnings;
use Getopt::Std;

# options
use vars qw/$opt_v/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] md.txt
EOF
  exit(8);
}

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
# [3] - opcode mask
# [4] - line number
# [5] - encoding list (not includes opcode mask)
# [6] - !encoding list

sub insert_ins
{
  my($iname, $op) = @_;
  # find longest nenc
  my $nmax = 0;
  my $nmask;
  foreach ( @{ $op->[5] } ) {
    if ( $_->[2] > $nmax ) {
      $nmask = $_;
      $nmax = $_->[2];
    }
  }
  my $tree = \%g_ops;
  # compare to where to insert
  if ( $nmax > $op ) {
    if ( exists $g_zero{$nmask->[1]} ) {
      $tree = $g_zero{$nmask->[1]};
    } else {
      $g_zero{$nmask->[1]} = ();
      $tree = $g_zero{$nmask->[1]};
    }
  }
}

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

### main
my $status = getopts("v");
usage() if ( !$status );
if ( 1 == $#ARGV ) {
  printf("where is arg?\n");
  exit(5);
}

my($fh, $state, $str, $line);
open($fh, '<', $ARGV[0]) or die("cannot open, error $!");
$state = $line = 0;
# op indexes
# [0] - name
# [1] - opcode
# [2] - opcode mask
# [3] - line no (for debugging)
# [4] - ref to enc
# [5] - ref to nenc
my($cname, $has_op, $op_line, @op, @enc, @nenc);
# reset current instruction
my $reset = sub {
  $has_op = $op_line = 0;
  @op = @enc = @nenc = ();
};
my $ins_op = sub {
  printf("%s %s %X\n", $cname, $op[0], $op[1]) if ( defined $opt_v );
  $op[3] = $op_line;
  $op[4] = \@enc;
  $op[5] = \@nenc;
  insert_ins($cname, \@op);
};
while( $str = <$fh> ) {
  chomp $str;
  $line++;
  if ( !$state ) {
    if ( $str =~ /ENCODING\s+WIDTH\s+(\d+)\s*\;/ ) {
       $g_size = int($1);
       $state = 1;
    }
    next;
  }
# printf("%d %s\n", $state, $str);
  if ( $str =~ /CLASS\s+\"([^\"]+)\"/ ) {
    if ( $has_op ) {
      $ins_op->(); $reset->();
    }
    $has_op = 1;
    $op_line = $line;
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
  if ( $str =~ /^\s*ENCODING/ ) {
    $state = 4;
    next;
  }
  # encodings
  if ( 4 == $state ) {
    if ( $str =~ /^\s*\!(\S+)\s*;/ ) {
      # put ref from g_mnames into nenc
      if ( !exists $g_mnames{$1} ) {
        printf("%s not exists, line %d op %s\n", $1, $line, $op[0]);
      } else {
        push( @nenc, $g_mnames{$1} );
      }
      next;
    }
    # check for =Opcode
    if ( $str =~ /^\s*(\S+)\s*=\s*Opcode/ ) {
      if ( !exists $g_mnames{$1} ) {
        printf("opcode mask %s not exists, line %d op %s\n", $1, $line, $op[0]);
        $reset->();
      } else {
        $op[2] = $g_mnames{$1};
      }
      next;
    }
    # check and put to enc
    if ( $str =~ /^\s*(\S+)\s*=/ ) {
      if ( !exists $g_mnames{$1} ) {
        printf("encode mask %s not exists, line %d op %s\n", $1, $line, $op[0]);
        $reset->();
      } else {
        push(@enc, $str);
      }
    }
  }
}
close $fh;
# check last instr
$ins_op->() if ( $has_op );