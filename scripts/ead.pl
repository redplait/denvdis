#!perl -w
# some nvdisasm encoding analysis
use strict;
use warnings;
use Getopt::Std;
use Data::Dumper;

# options
use vars qw/$opt_a $opt_m $opt_v $opt_w/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] md.txt
 Options:
 - a - add alternates
  -m - generate masks
  -v - verbose
  -w - dump warnings
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
  if ( exists $g_mmasks{$v} && defined($opt_w) ) {
    printf("%s has the same mask as %s (%s)\n", $name, $g_mmasks{$v}->[0], dump_decode( $g_mmasks{$v}->[3] ));
    $g_mnames{$name} = $g_mmasks{$v};
    return 1;
  }
  my @a = split //, $v;
  my $cnt = 0;
  foreach ( @a ) {
    $cnt++ if ( $_ eq 'x' );
  }
  if ( $cnt > 64 && defined($opt_w) ) {
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

sub zero_mask
{
 my($a, $list) = @_;
 for( my $i = 0; $i < scalar(@$list); $i += 2 ) {
   my $base = $list->[$i];
   for ( my $j = 0; $j < $list->[$i + 1]; $j++ ) {
     $a->[$base + $j] = '0';
   }
 }
}

# args: a - ref to result array, v - int value, mask - ref to value in g_mnames
sub mask_value
{
  my($a, $v, $mask) = @_;
  my $list = $mask->[3];
  for( my $i = 0; $i < scalar(@$list); $i += 2 ) {
   my $base = $list->[$i];
   for ( my $j = 0; $j < $list->[$i + 1]; $j++ ) {
     my $sym = $v & 1 ? '1' : '0';
     $a->[$base + $j] = $sym;
     $v >>= 1;
   }
 }
}

sub gen_inst_mask
{
  my $op = shift;
  my @res = ('-') x $g_size;
  foreach my $mask ( @{ $op->[6] } ) {
    zero_mask(\@res, $mask->[3]);
  }
  # opcode
  mask_value(\@res, $op->[2], $op->[3]);
  # encodings
  my @new;
  my $altered = 0;
  foreach my $emask ( @{ $op->[5] } ) {
    if ( $emask =~ /^\s*(\S+)\s*=\s*(\d+)/ ) {
      $altered++;
      mask_value(\@res, int($2), $g_mnames{$1});
    } elsif ( $emask =~ /^\s*(\S+)\s*=\*\s*(\d+)/ ) {
      $altered++;
      mask_value(\@res, int($2), $g_mnames{$1});
    } else {
      push @new, $emask;
    }
  }
  $op->[5] = \@new if $altered;
  return join('', @res);
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
  my($cname, $op) = @_;
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
  if ( $nmax > $op->[2]->[2] ) {
    $g_zero{$nmask->[1]} = { } if ( !exists $g_zero{$nmask->[1]} );
    $tree = $g_zero{$nmask->[1]};
  }
  # put class name to op
  unshift @$op, $cname;
  # insert with opcode mask
  $tree->{ $op->[3]->[1] } = { } if ( !exists $tree->{ $op->[3]->[1] } );
  my $mpail = $tree->{ $op->[3]->[1] };
  # and then by opcode
  if ( !exists $mpail->{ $op->[2] } ) {
    $mpail->{ $op->[2] } = [ $op ];
  } else {
    # check if they have the same names
    my $aref = $mpail->{ $op->[2] };
    printf("duplicated ins %s (old %s) mask %s value %X\n", $op->[1], $aref->[0]->[1], $op->[3]->[1], $op->[2])
      if ( $op->[1] ne $aref->[0]->[1] );
    push @{ $aref }, $op;
  }
}

sub dump_tree
{
  my($t, $level) = @_;
  my $id = $level ? ' ' x $level : '';
  foreach ( keys %{ $t } ) {
    printf("%s%s %s\n", $id, $_, $g_mmasks{ $_ }->[0]);
    # value is int
    while( my($v, $ops) = each %{ $t->{$_} } ) {
      printf("  %s %X", $id, $v);
      # check size of ops array
      my $ops_size = scalar @$ops;
      if ( 1 == $ops_size ) {
        printf(" %s\n", $ops->[0]->[1]);
      } else {
        # for duplicates dump also line & encodings
        printf(" %d items:\n", $ops_size);
        for( my $i = 0; $i < $ops_size; $i++ ) {
          printf("    %s %s line %d:\n", $id, $ops->[$i]->[1], $ops->[$i]->[4]);
          # dump encodings
          foreach my $enc ( @{ $ops->[$i]->[5] } ) {
            printf("      %s\n", $enc);
          }
        }
      }
    }
  }
}

sub dump_negtree
{
  my $t = shift;
  foreach ( keys %{ $t } ) {
    printf("%s %s\n", $_, $g_mmasks{ $_ }->[0]);
# print Dumper($t->{$_});
    dump_tree( $t->{$_}, 2 );
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

# mask map, key - mask string, value - instruction like in g_ops
my(%g_masks);
my $g_dups = 0;

sub dump_dup_masks
{
  while( my($v, $ops) = each %g_masks ) {
    my $size = scalar @$ops;
    next if ( 1 == $size );
    printf("%s: %d items\n", $v, $size);
    # dump duplicated instructions
    foreach my $op ( @$ops ) {
      printf("  %s line %d\n", $op->[1], $op->[4]);
      # dump encodings
      foreach my $enc ( @{ $op->[5] } ) {
        printf("    %s\n", $enc);
      }
    }
  }
}


sub insert_mask
{
  my($cname, $op) = @_;
  # put class name to op
  unshift @$op, $cname;
  my $mask = gen_inst_mask($op);
  if ( exists $g_masks{$mask} ) {
     # skip alternate classes
     $g_dups++ if ( !$op->[7] );
     if ( !$op->[7] || defined($opt_a) ) {
       my $ops = $g_masks{$mask};
       push @$ops, $op;
     }
   } else {
    $g_masks{$mask} = [ $op ];
  }
}

### main
my $status = getopts("amvw");
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
# [6] - is alternate class
my($cname, $has_op, $op_line, @op, @enc, @nenc, $alt);
# reset current instruction
my $reset = sub {
  $cname = '';
  $alt = $has_op = $op_line = 0;
  @op = @enc = @nenc = ();
};
# insert copy of current instruction
my $ins_op = sub {
  printf("%s %s %X\n", $cname, $op[0], $op[1]) if ( defined $opt_v );
  if ( !scalar( @enc ) && !scalar( @nenc ) ) {
    printf("%s %s has empty encoding\n", $cname, $op[0]);
    return;
  }
  my @c = @op;
  my @cenc = @enc;
  my @cnenc = @nenc;
  $c[3] = $op_line;
  $c[4] = \@cenc;
  $c[5] = \@cnenc;
  $c[6] = $alt;
  if ( defined($opt_m) ) {
    insert_mask($cname, \@c);
   } else {
    insert_ins($cname, \@c);
   }
};
$reset->();
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
  if ( $str =~ /(ALTERNATE\s+)?CLASS\s+\"([^\"]+)\"/ ) {
    if ( $has_op ) {
      $ins_op->(); $reset->();
    }
    $has_op = 1;
    $op_line = $line;
    $cname = $2;
    $alt = defined($1);
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
        # $reset->();
      } else {
        push(@enc, $str);
      }
    }
  }
}
close $fh;
# check last instr
$ins_op->() if ( $has_op );

# dump trees
if ( defined($opt_m) ) {
# results
#                sm3 sm4 sm5 sm57 sm72 sm75 sm100 sm101 sm120
# !enc + opcode  205 258 273 275  310  548
# total          279 261 321 363  365  681
# -alternate     135 183 200 194  135  241
# witn encoded = const
# total          330 310 374 411  433  807
# duplicated     113 160 171 173   74  128
# with encoded =* const - I don't know what this means
# total          330 310 374 415  538 1010    992   927  1016
# duplicated     113 160 171 173   60   96    141   129   156
  dump_dup_masks();
  printf("%d duplicates, total %d\n", $g_dups, scalar keys %g_masks);
} else {
  dump_negtree(\%g_zero);
  printf("--- opcodes tree\n");
  dump_tree(\%g_ops, 0);
}