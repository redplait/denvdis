#!perl -w
# some nvdisasm encoding analysis
use strict;
use warnings;
use Getopt::Std;
use Data::Dumper;

# options
use vars qw/$opt_a $opt_e $opt_m $opt_v $opt_w/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] md.txt
 Options:
 - a - add alternates
  -e - dump enums
  -m - generate masks
  -v - verbose
  -w - dump warnings
EOF
  exit(8);
}

# some hardcoded tabs
my %fmz = (
 nofmz => 0,
 noFTZ => 0,
 FTZ => 1,
 FMZ => 2,
 INVALIDFMZ3 => 3
);

my %pmode = (
 IDX => 0,
 F4E => 1,
 B4E => 2,
 RC8 => 3,
 ECL => 4,
 ECR => 5,
 RC16 => 6,
 INVALID7 => 7
);

my %hilo = (
 LO => 0,
 HI => 1
);
my %b1b0 = (
 H0 => 0,
 H1 => 2,
 B0 => 0,
 B1 => 1,
 B2 => 2,
 B3 => 3
);
my %b3b0 = (
 B0 => 0,
 B1 => 1,
 B2 => 2,
 B3 => 3
);

my %cwmode = (
 C => 0,
 W => 1
);
my %bval = (
 BM => 0,
 BF => 1
);
my %store_cache = (
 WB => 0,
 CG =>1,
 CS => 2,
 WT => 3
);
my %reuse = (
 noreuse => 0,
 reuse => 1
);

my %Tabs = (
 FMZ => \%fmz,
 PMode => \%pmode,
 HILO => \%hilo,
 CWMode => \%cwmode,
 BVal => \%bval,
 B1B0 => \%b1b0,
 B3B0 => \%b3b0,
 REUSE => \%reuse,
 StoreCacheOp => \%store_cache
);

# global enums hash map - like Tabs but readed dynamically
my %g_enums;
my $g_rz = 0;

sub dump_enums
{
  printf("-- Enums\n");
  foreach my $e_name ( sort keys %g_enums ) {
    my $enum = $g_enums{$e_name};
    next if ( !scalar keys %$enum );
    printf("%s:\n", $e_name);
    while( my($n, $v) = each %$enum ) {
      printf("  %s\t%d\n", $n, $v);
    }
    printf("\n");
  }
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
    if ( $emask =~ /^(\S+)\s*=\s*0b(\S+)/ ) { # enc = 0bxxx
      my $mask = $g_mnames{$1};
      mask_value(\@res, parse0b($2), $mask);
      $altered++;
    } elsif ( $emask =~ /^(\S+)\s*=\s*(\d+)/ ) {
      $altered++;
      mask_value(\@res, int($2), $g_mnames{$1});
    } elsif ( $emask =~ /^(\S+)\s*=\*\s*(\d+)/ ) {
      $altered++;
      mask_value(\@res, int($2), $g_mnames{$1});
    } else {
      push @new, $emask;
    }
  }
  $op->[5] = \@new if $altered;
  # enc = `const - in op->[9]
  foreach my $q ( @{ $op->[9] } ) {
    if ( $q =~ /^(\S+)\s*=\s*\`(\S+)/ ) {
      my $v = $2;
      if ( $v eq 'Register@RZ' ) {
        mask_value(\@res, $g_rz, $g_mnames{$1});
        next;
      }
      # check enums
      my $mask = $g_mnames{$1};
      if ( $v =~ /^(\S+)@(\w+)/ ) {
        if ( !exists $g_enums{$1} ) {
          printf("cannot find quoted enum %s for %s line %d: %s\n", $1, $op->[1], $op->[4], $q);
          next;
        }
        my $tab = $g_enums{$1};
        if ( !exists $tab->{$2} ) {
          printf("cannot find quoted enum %s in %s for %s line %d: %s\n", $2, $1, $op->[1], $op->[4], $q);
          next;
        }
        mask_value(\@res, $tab->{$2}, $mask);
      }
    }
  }
  # check /Group(Value):alias in format
  while ( $op->[8] =~ /\/(\w+)\(\"?([^\"\)]+)\"?\)\:(\w+)/g ) {
    if ( exists $Tabs{$1} && exists $Tabs{$1}->{$2} ) {
      my $value = $Tabs{$1}->{$2};
      my $what = check_enc($op->[5], $1, $3);
      if ( defined($what) ) {
        mask_value(\@res, $value, $what);
      }
      next;
    }
    # check in enums
    if ( !exists $g_enums{$1} ) {
      printf("cannot find /enum %s for %s line %d\n", $1, $op->[1], $op->[4]);
      next;
    }
    my $tab = $g_enums{$1};
    if ( !exists $tab->{$2} ) {
      printf("cannot find /enum %s in %s for %s line %d\n", $2, $1, $op->[1], $op->[4]);
      next;
    }
    my $value = $tab->{$2};
    my $what = check_enc($op->[5], $1, $3);
    if ( defined($what) ) {
      mask_value(\@res, $value, $what);
    }
  }
  # and again check for ZeroRegister(RZ) in format - in worst case just assign it yet one more time
  if ( $op->[8] =~ /\bZeroRegister\(\"?RZ\"?\)\:(\w+)/ ) {
    my $what = check_enc($op->[5], $1, $1);
    mask_value(\@res, $g_rz, $what) if ( defined $what );
  }
  return join('', @res);
}

sub check_enc
{
  my($e, $name, $alias) = @_;
  $alias =~ s/\}\s*$//;
# printf("check_enc %s %s\n", $name, $alias);
  for my $enc ( @$e ) {
    if ( $enc =~ /^\s*(\S+)\s*=\s*\*?\s*(\S+)\s*/ ) {
      if ( $2 eq $name or $2 eq $alias ) {
         return $g_mnames{$1};
      }
    }
  }
  return undef;
}

# opcodes divided into 2 group - first start with some zero mask (longest if there are > 1) and second with longest opcode
# key is mask value is map of second type
my(%g_zero, %g_ops);
my $g_diff_names = 0;
# format hash where key is opcode, value - array where
# [0] - class name
# [1] - name
# [2] - opcode
# [3] - opcode mask
# [4] - line number
# [5] - encoding list (not includes opcode mask)
# [6] - !encoding list
# [7] - is alternate class
# [8] - format string
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
    if ( $op->[1] ne $aref->[0]->[1] ) {
      printf("duplicated ins %s (old %s) mask %s value %X\n", $op->[1], $aref->[0]->[1], $op->[3]->[1], $op->[2]);
      $g_diff_names++;
    }
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
    my $name1 = $ops->[0]->[1];
    foreach my $op ( @$ops ) {
      if ( $name1 ne $op->[1] ) {
        printf(" !!%s line %d %s\n", $op->[1], $op->[4], $op->[8]);
      } else {
        printf("   %s line %d %s\n", $op->[1], $op->[4], $op->[8]);
      }
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
       $g_diff_names++ if ( $op->[1] ne $g_masks{$mask}->[0]->[1] );
       my $ops = $g_masks{$mask};
       push @$ops, $op;
     }
   } else {
    $g_masks{$mask} = [ $op ];
  }
}

### main
my $status = getopts("aemvw");
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
# [7] - format string
# [8] - ref to tabs
my($cname, $has_op, $op_line, @op, @enc, @nenc, @tabs, $alt, $format);

# enum state
my($curr_enum, $eref);
# 0 - don't parse, 1 - expect start of enum, 2 - continue with next line
my $estate = 0;
my $reset_enum = sub {
  $curr_enum = 0;
};
my $parse_pair = sub {
  my $s = shift;
  $s =~ s/\;\s*$//;
  $s =~ s/^\s+//;
  $s =~ s/\s+$//;
  return 0 if ( $s eq '' );
  # simplest case - just some enum
  if ( $s =~ /^\"?([\w\.]+)\"?$/ ) {
    $eref->{$1} = $curr_enum++;
    return 1;
  }
  # enum = 0b
  if ( $s =~ /^\"?([\w\.]+)\"?\s*=\s*0b(\w+)\s*$/ ) {
    my $name = $1;
    $curr_enum = parse0b($2);
    $eref->{$1} = $curr_enum++;
    return 1;
  }
  # enum = number
  if ( $s =~ /^\"?([\w\.]+)\"?\s*=\s*(\d+)\s*$/ ) {
    $curr_enum = int($2);
    $eref->{$1} = $curr_enum++;
    return 1;
  }
 printf("bad enum %s on line %d\n", $s, $line);
 0;
};
my $parse_enum = sub {
  my $s = shift;
  my $n = 0;
  foreach my $pat ( split /\s*,\s*/, $s ) {
    $n++;
    $parse_pair->($pat);
  }
  # if it's last like something;
  $parse_pair->($s) if ( !$n );
};

# reset current instruction
my $reset = sub {
  $format = $cname = '';
  $alt = $has_op = $op_line = 0;
  @op = @enc = @nenc = @tabs = ();
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
  my @ctabs = @tabs;
  $c[3] = $op_line;
  $c[4] = \@cenc;
  $c[5] = \@cnenc;
  $c[6] = $alt;
  $c[7] = $format;
  $c[8] = \@ctabs;
  if ( defined($opt_m) ) {
    insert_mask($cname, \@c);
   } else {
    insert_ins($cname, \@c);
   }
};
$reset->(); $reset_enum->();
while( $str = <$fh> ) {
  chomp $str;
  $line++;
  if ( !$state ) {
    if ( $str =~ /ENCODING\s+WIDTH\s+(\d+)\s*\;/ ) {
       $g_size = int($1);
       $state = 1;
       next;
    }
    # check enums
    if ( $str =~ /^\s*TABLES/ ) {
      $estate = 0;
      next;
    }
    if ( $str =~ /^\s*REGISTERS/ ) {
      $estate = 1;
      next;
    }
    if ( $str =~ /^\s*ZeroRegister .*\"?RZ\"?\s*=\s*(\d+)\s*;/ )
    {
      $estate = 1;
      $g_rz = int($1);
      next;
    }
    $estate = 1 if ( !$estate && $str =~ /^\s*SpecialRegister / );
    next if ( !$estate );
# printf("e%d %s\n", $estate, $str);
    # 1 - new enum
    if ( 1 == $estate ) {
      if ( $str =~ /^\s*(\w+)\s*$/ ) {
        my %tmp;
        $eref = $g_enums{$1} = \%tmp;
        $estate = 2;
        next;
      }
      next if ( $str =~ /^\s*(\w+)\s*=/ );
      if ( $str =~ /^\s*(\w+)\s+(.*)\s*;?/ ) {
        my %tmp;
        $eref = $g_enums{$1} = \%tmp;
        $parse_enum->($2);
        if ( $str =~ /\;\s*$/ ) {
          $reset_enum->();
        } else {
          $estate = 2;
        }
        next;
      }
    }
    if ( 2 == $estate ) {
      if ( $str =~ /^\s*(.*)\;\s*$/ ) {
       $parse_enum->($1);
       $reset_enum->(); $estate = 1;
      } else {
       $parse_enum->($str);
      }
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
  # parse format
  if ( $state == 2 && $str =~ /FORMAT\s+(?:PREDICATE\s+)?.*Opcode\s*?(.*)$/ ) {
    $format = $1;
    $state = 6 if ( $str !~ /;\s*$/ );
    next;
  }
  if ( 6 == $state ) {
    if ( $str !~ /FORMAT\s+(?:PREDICATE\s+)?.*Opcode/ ) {
      # grab /something()
      if (  $str =~ /^\s*(.*\/\w.*)$/ ) {
        $format .= ' ' . $1;
      } elsif ( $str =~ /^\s*(.*\(\d+\).*)\s*$/ ) {
        $format .= ' ' . $1;
      }
    }
    $state = 2 if ( $str =~ /;\s*$/ );
    next;
  }
  if ( 6 == $state && $str =~ /CONDITIONS/ ) {
    $state = 2;
    next;
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
    for my $s ( split /;/, $str ) {
      # trim leading spaces
      $s =~ s/^\s+//g;
      if ( $s =~ /^\!(\S+)\s*/ ) {
        # put ref from g_mnames into nenc
        if ( !exists $g_mnames{$1} ) {
          printf("%s not exists, line %d op %s\n", $1, $line, $op[0]);
        } else {
          push( @nenc, $g_mnames{$1} );
        }
        next;
      }
      # check for =Opcode
      if ( $s =~ /^(\S+)\s*=\s*Opcode/ ) {
        if ( !exists $g_mnames{$1} ) {
          printf("opcode mask %s not exists, line %d op %s\n", $1, $line, $op[0]);
          $reset->();
        } else {
          $op[2] = $g_mnames{$1};
        }
        next;
      }
      # check for mask = `something
      if ( $s =~ /^(\S+)\s*=\s*\`/ ) {
        if ( !exists $g_mnames{$1} ) {
          printf("quoted encode mask %s not exists, line %d op %s\n", $1, $line, $op[0]);
          # $reset->();
        } else {
          push(@tabs, $s);
        }
        next;
      }
      # check remaining in g_mnames and put to enc
      if ( $s =~ /^(\S+)\s*=/ ) {
        if ( !exists $g_mnames{$1} ) {
          printf("encode mask %s not exists, line %d op %s\n", $1, $line, $op[0]);
          # $reset->();
        } else {
          push(@enc, $s);
        }
        next;
      }
    } # for all ; separated encodings
    next;
  }
}
close $fh;
# check last instr
$ins_op->() if ( $has_op );

dump_enums() if ( defined($opt_e) );

# dump trees
if ( defined($opt_m) ) {
# results
#                sm3 sm4 sm5 sm57 sm72 sm75 sm100 sm101 sm120
# !enc + opcode  205 258 273 275  310  548
# total          279 261 321 363  365  681
# -alternate     135 183 200 194  135  241
#  with encoded = const
# total          330 310 374 411  433  807
# duplicated     113 160 171 173   74  128
#  with encoded =* const - I don't know what this means
# total          330 310 374 415  538 1010    992   927  1016
# duplicated     113 160 171 173   60   96    141   129   156
#  FMZ & PMode + enc = 0bxxx
# total          330 354
# duplicated     113 119
#  `Register@RZ, different names 0
# total          334 358 378 414  538 1010    992   927  1016
# duplicated     113 117 169 176   60   96    141   129   156
#  with enums
# total          340 364 369 405  535 1024   1007   935  1054
# duplicated     107 111 178 185   56   83    133   128   147
#  additional check for ZeroRegister(RZ)
# total          359 383 393 430  570 1064   1036   964  1083
# duplicated      90  92 154 160   34   58    107   102   121
  dump_dup_masks();
  printf("%d duplicates (%d different names), total %d\n", $g_dups, $g_diff_names, scalar keys %g_masks);
} else {
  dump_negtree(\%g_zero);
  printf("--- opcodes tree\n");
  dump_tree(\%g_ops, 0);
  printf("%d different names\n", $g_diff_names);
}