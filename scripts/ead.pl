#!perl -w
# some nvdisasm encoding analysis
# -CBFrm to produce c++
# add -p to dump predicates: https://redplait.blogspot.com/2025/04/nvidia-sass-disassembler-part-6.html
use strict;
use warnings;
use Getopt::Std;
use Carp;
use Data::Dumper;
use v5.10;
use feature qw( switch );
no warnings qw( experimental::smartmatch );

# options
use vars qw/$opt_a $opt_b $opt_B $opt_C $opt_c $opt_e $opt_f $opt_F $opt_g $opt_i $opt_m $opt_N $opt_p $opt_r $opt_t $opt_T $opt_v $opt_w $opt_z/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] md.txt
 Options:
  -a - add alternates
  -b - apply bitsets
  -B - build decision tree for decoding
  -C - suffix
  -c - use format constant to form mask
  -e - dump enums
  -f - dump fully filled masks
  -F - filter by enums
  -g - parse groups
  -i - dump instructions formats
  -m - generate masks
  -N - test single bitmask from cmd-line
  -p - parse predicated
  -r - fill in reverse order
  -t - dump tables
  -T - test file
  -v - verbose
  -w - dump warnings
  -z - remove fully filled tables patterns
EOF
  exit(8);
}

# some hardcoded tabs
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
# enums with only value
my %g_single_enums;
# used enums
my %g_used_enums;
my $g_rz = 0;

# non-existing enums
my %g_bad_enums = (
 REQ => 1,
 RD => 1,
 WR => 1,
);

sub dump_enums
{
  printf("-- Enums\n");
  foreach my $e_name ( sort keys %g_enums ) {
    my $enum = $g_enums{$e_name};
    next if ( !scalar keys %$enum );
    printf("<S> ") if ( exists $g_single_enums{$e_name} );
    printf("%s:\n", $e_name);
    while( my($n, $v) = each %$enum ) {
      printf("  %s\t%d\n", $n, $v);
    }
    printf("\n");
  }
}

sub is_single_enum
{
  my $name = shift;
  return 0 if ( !exists $g_enums{$name} );
  my $k = $g_enums{$name};
  my @keys = keys %$k;
  1 == scalar @keys;
}

# more fast version
sub check_single_enum
{
  my $e = shift;
  return unless exists $g_single_enums{ $e };
  my @v = values %{ $g_enums{ $e } };
  $v[0];
}

# args: instruction, current mask, enum, value
sub get_enc_enum
{
  my($op, $mask, $e, $vname) = @_;
  if ( !exists $g_enums{$e} ) {
    printf("not existing enum %s for mask %s, op %s line %d\n", $e, $mask, $op->[1], $op->[4]);
    return;
  }
  my $er = $g_enums{$e};
  if ( !exists $er->{$vname} ) {
    printf("not existing value %s in enum %s for mask %s, op %s line %d\n", $vname, $e, $mask, $op->[1], $op->[4]);
    return;
  }
  $er->{$vname};
}

# args: ref to currently constructed enum, name of enum, line
sub merge_enum
{
  my($er, $name, $line) = @_;
  if ( !exists $g_enums{$name} ) {
    printf("unknown enum %s to merge on line %d\n", $name, $line);
    return 0;
  }
  my $em = $g_enums{$name};
  keys %$em;
  while( my($n, $v) = each %$em ) {
    $er->{$n} //= $v;
  }
  return 1;
}

# in parallel universe we could make reverse hash value->name
# args: ref to enum hashmap, value to find
sub enum_by_value
{
  my($hr, $val) = @_;
  keys %$hr;
  while( my($n, $v) = each %$hr ) {
    return $n if ( $val == $v );
  }
  undef;
}

sub is_type
{
  my $c = shift;
  return ($c eq 'BITSET') || ($c eq 'UImm') || ($c eq 'SImm') || ($c eq 'SSImm') || ($c eq 'RSImm')
   || ($c eq 'F64Imm') || ($c eq 'F16Imm') || ($c eq 'F32Imm');
}

# global tables hash map, key is name of table, value is another hash map { value -> [ literals list ] }
my %g_tabs;
my %g_used_tabs;

sub dump_tabs
{
  printf("-- Tables\n");
  foreach my $t_name ( sort keys %g_tabs ) {
    my $th = $g_tabs{$t_name};
    next if ( !scalar keys %$th );
    printf("%s:\n", $t_name);
    while( my($v, $lr) = each %$th ) {
      printf("  %d\t", $v);
      if ( 'ARRAY' eq ref $lr ) {
        printf("%s\n", join(" ", @$lr));
      } else {
        printf("%s\n", $lr);
      }
    }
    printf("\n");
  }
}

sub tab_dim
{
  my($tname) = shift;
  return unless exists $g_tabs{$tname};
  my @v = values %{ $g_tabs{$tname} };
  return unless @v;
  my $v1 = $v[0];
  return scalar @$v1 if ( 'ARRAY' eq ref $v1 );
  1;
}

# find key in table $tname for row in array $tref
sub rev_tab_lookup
{
  my($tname, $tref) = @_;
  return unless exists $g_tabs{$tname};
  my $t = $g_tabs{$tname};
  keys %$t;
  while( my($k, $row) = each %$t ) {
    my $same = 1;
    # compare rows - could use Data::Compare
#  printf("%s BAD %s", $tname, $row) if ( 'ARRAY' ne ref $row );
    for ( my $i = 0; $i < scalar @$tref; $i++ ) {
      if ( $row->[$i] != $tref->[$i] ) { $same = 0; last; }
    }
    return $k if $same;
  }
  undef;
}

sub rev_tab1
{
  my($tname, $v) = @_;
  return unless exists $g_tabs{$tname};
  my $t = $g_tabs{$tname};
  keys %$t;
  while( my($k, $row) = each %$t ) {
    return $k if ( $v == $row );
  }
  undef;
}

# args: string to parse, line number for diagnostic msg
sub parse_tab_value
{
  my($s, $line) = @_;
  return parse0b($1) if ( $s =~ /^0b(\w+)/ );
  return hex($1)     if ( $s =~ /^0x(\w+)/i );
  return int($1)     if ( $s =~ /^(\d+)/ );
  # check enum@value
  if ( $s =~ /(\w+)@\"?(\w+)\"?\s*$/ ) {
   return $g_enums{$1}->{$2} if ( exists $g_enums{$1} );
   printf("unknown enum %s for table key, line %d\n", $1, $line);
   return;
  }
  printf("unknown table value %d, line %d\n", $s, $line);
  undef;
}

sub parse_tab_key
{
  my($s, $line) = @_;
  # check in enums
  if ( $s =~ /(\w+)@\"?([^\s\"]+)\"?$/ ) {
    return $g_enums{$1}->{$2} if ( exists $g_enums{$1} );
    printf("unknown enum %s for table key, line %d\n", $1, $line) if ( defined $opt_v );
  }
  return $s if ( $s eq '-' ); # like in GetPseudoXXX
  return hex($1) if ( $s =~ /0x([0-9a-f]+)$/i );
  if ( $s =~ /\'/ ) {
    $s =~ s/\'//g;
    return $s;
  }
  return int($s);
}

sub parse_tab_keys
{
  my($s, $line) = @_;
  # remove trailing spaces
  $s =~ s/\s+$//;
  if ( $s =~ /\s+/ ) {
    # this is compond key - must return ref to array
    my @res;
    foreach my $v ( split /\s+/, $s ) {
      my $next = parse_tab_key($v, $line);
      return if ( !defined $next );
      push @res, $next;
    }
    return unless ( scalar @res );
    return \@res;
  } else {
    # just some literal
    return parse_tab_key($s, $line);
  }
}

# virtual queues logic
# the problem is that VQ_XX has different values, like
# VQ_BAR_EXCH = 26 in sm75 but in sm80:
#  VQ_DMMA = 26
#  VQ_BAR_EXCH = 31
# key is number, value is string name
my %g_vq;
# produce 3 VQ related data:
# 1) enum
# 2) static array with names
# 3) get_vq_name "C" exported function
sub gen_vq
{
  my $fh = shift;
  return unless keys %g_vq;
  # enum
  printf($fh "enum %s_vq {\n", $opt_C);
  my $m = 0;
  foreach my $k ( sort { $a <=> $b } keys %g_vq ) {
    printf($fh " %s = %d,\n", $g_vq{$k}, $k);
    $m = $k if ( $k > $m );
  }
  # string names
  printf($fh "};\nstatic const char *vq_names[] = {\n");
  foreach my $i (0..$m) {
    printf($fh "/* %d */ ", $i);
    if ( exists $g_vq{$i} ) {
      printf($fh "\"%s\",\n", $g_vq{$i});
    } else {
      printf($fh "nullptr,\n");
    }
  }
  # get_vq_name
  print $fh <<EOF;
};
const char *get_vq_name(int idx) {
  if ( idx < 0 || idx >= (sizeof(vq_names) / sizeof(vq_names[0])) ) return nullptr;
  return vq_names[idx];
}
EOF
}

# ENCODING WIDTH
my $g_size;
my $g_min_len;

# for masks we need 2 map - first with name as key, second as mask
# both contains as value array where
# [0] - name
# [1] - mask
# [2] - size of significant bits
# [3] - list for decoding
my(%g_mnames, %g_mmasks);

sub mask_len
{
  my $op = shift;
  my $res = 0;
  my $list = $op->[3];
  for ( my $i = 0; $i < scalar @$list; $i += 2 ) {
    $res += $list->[$i+1];
  }
  return $res;
}

sub scale_len
{
  my $s = shift;
  return 0 unless defined($s);
  given(int($s)) {
    when(1) { return 0;} # wtf?
    when(2) { return 1;}
    when(4) { return 2;}
    when(8) { return 3;}
    when(16) { return 4;}
    when(32) { return 5;}
    default: {
      carp("unknown scale $s");
      return 0;
    }
  }
}

# key - name, value 1 if it incompleted and 0 otherwise
my %g_cached_enums;
# return greatest possibly value for this mask
# like if len = 3 then it is 7: 1 << 3 = 8 then -1 = 7
sub mask_range
{
  my $m = shift;
  my $len = mask_len($m);
  return 1 if ( 1 == $len );
  return (1 << $len) - 1;
}

sub is_incomplete
{
  my($m, $ename) = @_;
  return $g_cached_enums{$ename} if exists $g_cached_enums{$ename};
  # not cached
  my $e = $g_enums{$ename};
  my $res = 0;
  my %tmp;
  $tmp{$_} = 1 for ( values %$e );
  foreach ( 0 .. mask_range($m) ) {
    if ( !exists $tmp{$_} ) {
      $res = 1;
      last;
    }
  }
  $g_cached_enums{$ename} = $res;
  $res;
}

sub dump_incompleted
{
  return if ! keys(%g_cached_enums);
  printf("Incompleted enums:\n");
  while( my($k, $v) = each(%g_cached_enums) ) {
    next if ( !$v );
    printf(" %s\n", $k);
  }
}

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

# calc count of meaningful bits
sub calc_mean_bits
{
  my $m = shift;
  scalar grep { $_ ne '-' } @$m;
}

# for tests only
sub cmp_mask
{
  my($m, $c) = @_;
  if ( length($c) != $g_size ) {
    carp("length of second string must be $g_size");
    return 0;
  }
  for ( my $i = 0; $i < $g_size; $i++ ) {
    my $l = substr($m, $i, 1);
    if ( $l eq '0' ) {
      return 0 if ( substr($c, $i, 1) ne '0' );
    } elsif ( $l eq '1' ) {
      return 0 if ( substr($c, $i, 1) ne '1' );
    }
  }
  1;
}

# the same cmp_mask but second argument is array to ref
sub cmp_maska
{
  my($m, $a) = @_;
  if ( scalar(@$a) != $g_size ) {
    carp("length of array must be $g_size");
    return 0;
  }
  for ( my $i = 0; $i < $g_size; $i++ ) {
    my $l = substr($m, $i, 1);
    if ( $l eq '0' ) {
      return 0 if ( $a->[$i] ne '0' );
    } elsif ( $l eq '1' ) {
      return 0 if ( $a->[$i] ne '1' );
    }
  }
  1;
}

# like cmp_maska but m is array and a is string
sub cmpa_mask
{
  my($m, $a) = @_;
  if ( scalar(@$m) != $g_size ) {
    carp("length of mask array must be $g_size");
    return 0;
  }
  for ( my $i = 0; $i < $g_size; $i++ ) {
    my $l = substr($a, $i, 1);
    if ( $m->[$i] eq '0' ) {
      return 0 if ( $l ne '0' && $l ne '-' );
    } elsif ( $m->[$i] eq '1' ) {
      return 0 if ( $l ne '1' && $l ne '-' );
    }
  }
  1;
}

# args: a - ref to result array, v - int value, mask - ref to value in g_mnames
sub mask_value
{
  my($a, $v, $mask) = @_;
  my $list = $mask->[3];
  if ( defined $opt_r ) {
   # from right to left, least bit will be right
   for ( my $i = scalar(@$list) - 1; $i > 0; $i -= 2 ) {
    my $base = $list->[$i-1] + $list->[$i] - 1;
    for ( my $j = 0; $j < $list->[$i]; $j++ ) {
     my $sym = $v & 1 ? '1' : '0';
     $a->[$base - $j] = $sym;
     $v >>= 1;
    }
   }
  } else {
   # from left to right, least bit will be left
   for( my $i = 0; $i < scalar(@$list); $i += 2 ) {
    my $base = $list->[$i];
    for ( my $j = 0; $j < $list->[$i + 1]; $j++ ) {
     my $sym = $v & 1 ? '1' : '0';
     $a->[$base + $j] = $sym;
     $v >>= 1;
   }
  }
 }
}

# args: a - ref to array, mask - ref to value in g_mnames
sub extract_value
{
  my($a, $mask) = @_;
  my $list = $mask->[3];
  my $name = $mask->[0];
  my $res = 0;
  my $idx = 0;
  if ( defined $opt_r ) {
    # from right to left, least bit will be right
   for ( my $i = scalar(@$list) - 1; $i > 0; $i -= 2 ) {
     my $base = $list->[$i-1] + $list->[$i] - 1;
     for ( my $j = 0; $j < $list->[$i]; $j++, $idx++ ) {
      if ( $a->[$base - $j] eq '1' ) {
        if ( $idx > 63 ) {
          carp("too big index $idx in mask $name");
          return;
        }
        $res |= 1 << $idx;
      }
    }
   }
  } else {
   # from left to right, least bit will be left
   for( my $i = 0; $i < scalar(@$list); $i += 2 ) {
     my $base = $list->[$i];
     for ( my $j = 0; $j < $list->[$i + 1]; $j++ ) {
       if ( [$base + $j] eq '1' ) {
        if ( $idx > 63 ) {
          carp("too big index $idx in mask $name");
          return;
        }
        $res |= 1 << $idx;
       }
     }
   }
  }
  $res;
}

sub bit_array
{
  my $p = shift;
  my @res;
  foreach my $v (@$p) {
   for ( my $idx = 7; $idx >= 0; $idx-- ) {
     push @res, ($v & (1 << $idx)) ? '1' : '0';
   }
  }
  return \@res;
}

sub bit_array_rev
{
  my $p = shift;
  my @res;
  foreach my $v (reverse @$p) {
   for ( my $idx = 7; $idx >= 0; $idx-- ) {
     push @res, ($v & (1 << $idx)) ? '1' : '0';
   }
  }
  return \@res;
}

# encoding for 64bit - just ignore first qword from each 8
sub old64
{
  my $fp = shift;
  my $idx = 0;
  sub {
    my $str;
    if ( !$idx ) {
      return unless ( defined($str = <$fp>) );
      $idx++;
    }
    return unless ( defined($str = <$fp>) );
    $idx = 0 if ( 8 == ++$idx );
    return conv2a($str);
  };
}

# encoding for 88bit is pure madness
# first 64 bit is common for next 3 64bit instructions and actually contains 3 17bit usched infos
sub martian88
{
  my $fp = shift;
  my $idx = 0;
  my $v;
  # lea     eax, [r9+r9*4] - eax = r9 * 5
  # lea     ecx, [r9+rax*4] - ecx = r9 * 21
  # mov     eax, 1FFFFFh - 17bit mask
  my @offsets = ( 0, 21, 42 );
  sub {
    my $str;
    my $i;
    if ( !$idx ) {
      return unless ( defined($str = <$fp>) );
      my @a;
      while ( $str =~ /([0-9a-f]{2})/ig ) {
       push @a, hex($1);
       $i++;
      }
      if ( 8 != $i ) {
        carp("bad control word, len %d", $i);
        return;
      }
      if ( defined $opt_v ) { printf("%2.2X ", $_) for @a; }
      $v = bit_array_rev(\@a);
      printf("\n%s\n", join '', @$v) if ( defined $opt_v );
    }
    return unless ( defined($str = <$fp>) );
    my @l;
    $i = 0;
    while ( $str =~ /([0-9a-f]{2})/ig ) {
      push @l, hex($1);
      $i++;
    }
    if ( 8 != $i ) {
      carp("bad word, len %d", $i);
      return;
    }
    # result array has 88 - 64 - 21 = 3 leading 0
    my @res = ('0') x 3;
    # now add lower 21 bit from v
    push @res, splice( @$v, 1 + 21 * (2 - $idx), 21);
    $idx = 0 if ( 3 == ++$idx );
    my $body = bit_array_rev(\@l);
    push @res, @$body;
    if ( defined $opt_v ) {
      printf("   U                    ");
      printf("%2.2X      ", $_) for reverse @l; printf("\n");
    }
    return \@res;
  };
}

sub conv2a
{
  my $s = shift;
  my @res;
  my $idx = 0;
  while ( $s =~ /([0-9a-f]{2})/ig ) {
    push @res, hex($1);
    $idx++;
  }
  my $len = $g_size / 8;
  carp("bad test string, length must be $len") if ( $len != $idx );
  my @p;
  if ( $len == 8 ) {
    @p = reverse(@res);
  } elsif ( $len == 16 ) {
    @p = reverse splice(@res, 8);
    push(@p, reverse splice(@res));
  } else {
    carp("unknown size $len");
    return;
  }
  if ( defined $opt_v ) {
    printf("%2.2X      ", $_) for @p; printf("\n"); }
  return bit_array(\@p);
}

sub read128
{
  my $fp = shift;
  sub {
    my $str;
    return unless ( defined($str = <$fp>) );
    return conv2a($str);
  }
}

# arg: a - ref to result array, l - letter, mask - ref to value in g_mnames
sub fill_mask
{
  my($a, $l, $mask) = @_;
  my $res = 0;
  my $list = $mask->[3];
  for( my $i = 0; $i < scalar(@$list); $i += 2 ) {
   my $base = $list->[$i];
   for ( my $j = 0; $j < $list->[$i + 1]; $j++ ) {
     if ( '-' eq $a->[$base + $j] ) {
       $a->[$base + $j] = $l;
       $res++;
     }
   }
 }
 return $res;
}

my %g_letters = (
 RegA => 'a',
 RegB => 'b',
 RegC => 'c',
 Dest => 'D',
 Pred => 'p',
 SrcPred => '$',
 aSelect => 's',
 bSelect => 'S',
 VComp => 'V',
 OEWaitOnSb => 'W',
 usched_info => 'u',
);

# args: mask_name, format_name
sub get_letter
{
  my($m, $f) = @_;
  return $g_letters{$m} if ( exists $g_letters{$m} );
  return $g_letters{$f} if ( exists $g_letters{$f} );
  return 'I' if ( $m =~ /Imm/ );
  return 'I' if ( $f =~ /Imm/ );
  return 'V' if ( $m =~ /^VComp/ );
  return 'a' if ( $m =~ /_Ra$/ );
  return 'a' if ( $m =~ /_Ra_offset$/ ); # 72
  return 'c' if ( $m =~ /_Rc$/ );
  return 'b' if ( $m =~ /_Rb$/ );
  return 'd' if ( $m =~ /_Rd$/ );
  return 'p' if ( $m =~ /_Pg$/ );
  return 'f' if ( $m =~ /_srcfmt$/ );
  return 'F' if ( $m =~ /_dstfmt$/ );
  return 'o' if ( $m =~ /_opex$/ );
  undef;
}

# args: ref to op, string mask, optional array ref of missed masks
# return ref to array
sub get_filled_maska
{
  my($op, $str, $missed) = @_;
  my @a = split //, $str;
  my @x;
  foreach my $emask ( @{ $op->[5] } ) {
   if ( $emask =~ /(\w+)\s*=\*?\s*(\w+)/ ) {
       my $l = get_letter($1, $2);
       if ( defined $l ) {
         fill_mask(\@a, $l, $g_mnames{$1});
         next;
       }
    }
    push @x, $g_mnames{$1} if ( $emask =~ /(\S+)\s*=/ );
  }
  foreach ( @x ) {
    my $filled = fill_mask(\@a, 'x', $_);
    push @$missed, $_ if ( $filled && defined($missed) );
  }
  return \@a;
}

# args: ref to op, string mask
# return string
sub get_filled_mask
{
  my $a = get_filled_maska(@_);
  return join('', @$a);
}

# parse args to BITSET
# return (is_ok, size, value)
sub parse_bitset
{
  my $s = shift;
  return 0 if ( $s !~ /(\d+)\/(\w+)/ );
  my $size = int($1);
  my $vs = $2;
  # hex value
  if ( $vs =~ /0x([0-9a-f]+)/i ) {
    return ( 1, $size, hex($1) );
  }
  # 0b value
  if ( $vs =~ /0b(\w+)/ ) {
    return ( 1, $size, parse0b($1) );
  }
  # some decimal number
  if ( $vs =~ /(\d+)/ ) {
    return ( 1, $size, int($1) );
  }
  0;
}

# remove used encodes
# args: op, ref to hash with encoders to exclude
sub remove_encs
{
  my($op, $hr) = @_;
  my @new;
  my $altered = 0;
  foreach my $emask ( @{ $op->[5] } ) {
    if ( $emask =~ /^(\w+)\s*=/ ) {
      if ( exists $hr->{$1} ) {
        $altered++;
        next;
      }
    }
    push @new, $emask;
  }
  $op->[5] = \@new if $altered;
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
  my %rem;
  foreach my $emask ( @{ $op->[5] } ) {
    if ( $emask =~ /^([\w\.]+)\s*=\*?\s*0b(\S+)/ ) { # enc =*? 0bxxx
      my $mask = $g_mnames{$1};
      $rem{$1} = $emask;
      my $v = parse0b($2);
      mask_value(\@res, $v, $mask);
      # add v record for filter
      # if ( defined $op->[12] ) { push @{ $op->[12] }, [ $mask, 'v', $v ]; }
      # else { $op->[12] = [ [ $mask, 'v', $v ] ]; }
    } elsif ( $emask =~ /^([\w\.]+)\s*=\*?\s*(\d+)/ ) {
      my $mask = $g_mnames{$1};
      $rem{$1} = $emask;
      my $v = int($2);
      mask_value(\@res, $v, $mask);
      # add v record for filter
      # if ( defined $op->[12] ) { push @{ $op->[12] }, [ $mask, 'v', $v ]; }
      # else { $op->[12] = [ [ $mask, 'v', $v ] ]; }
    } elsif ( $emask =~ /^([\w\.]+)\s*=\*?\s*0x(\w+)/i ) {
      my $mask = $g_mnames{$1};
      my $v = hex($2);
      $rem{$1} = $emask;
      mask_value(\@res, $v, $mask);
      # add v record for filter
      # if ( defined $op->[12] ) { push @{ $op->[12] }, [ $mask, 'v', $v ]; }
      # else { $op->[12] = [ [ $mask, 'v', $v ] ]; }
    }
  }
  if ( scalar keys %rem ) {
    remove_encs($op, \%rem);
    %rem = ();
  }
  # enc = `const - in op->[9]
  foreach my $q ( @{ $op->[9] } ) {
    if ( $q =~ /^([\w\.]+)\s*=\*?\s*\`(\S+)/ ) {
      my $v = $2;
      if ( $v eq 'Register@RZ' ) {
        $rem{$1} = $q;
        mask_value(\@res, $g_rz, $g_mnames{$1});
        next;
      }
      # check enums
      if ( !exists $g_mnames{$1} ) {
        printf("mask %s not exists in %s line %d\n", $1, $op->[1], $op->[4]);
        next;
      }
      my $mask = $g_mnames{$1};
      if ( $v =~ /^(\S+)@(\w+)/ ) {
        if ( !exists $g_enums{$1} ) {
          printf("cannot find quoted enum %s for %s line %d: %s\n", $1, , $q);
          next;
        }
        my $tab = $g_enums{$1};
        if ( !exists $tab->{$2} ) {
          printf("cannot find quoted enum %s in %s for %s line %d: %s\n", $2, $1, $op->[1], $op->[4], $q);
          next;
        }
        $rem{$mask->[0]} = $q;
        mask_value(\@res, $tab->{$2}, $mask);
      }
    }
  }
  # end of enc = `const loop
  my @pos;
  if ( defined $opt_b ) {
  # check BITSET(size/value):mask
  while( $op->[8] =~ /(?:\'\&\'.*)\s*BITSET\(([^\)]+)\)\:([\w\.]+)/pg ) {
    my $mask = $2;
    my($ok, $size, $v) = parse_bitset($1);
    if ( !$ok ) {
      printf("bad BITSET args %s for %s line %d\n", $1, $op->[1], $op->[4]);
      next;
    }
    # check that mask exists and has size $size
    my $what = check_enc($op->[5], $mask, $mask);
    if ( !defined($what) ) {
      printf("bad BITSET mask %s for %s line %d\n", $mask, $op->[1], $op->[4]);
      next;
    }
    my $ms = mask_len($what);
    if ( $ms != $size ) {
      printf("BITSET size is %X but size of %s is %X for %s line %d\n",
       $size, $what->[0], $ms, $op->[1], $op->[4]);
      next;
    }
    $rem{$what->[0]} = 1;
    my $p = pos($op->[8]);
    push @pos, [ $p - length(${^MATCH}), $p ];
    mask_value(\@res, $v, $what);
  } }
  unless ( defined $opt_F ) {
  # check /enum:alias where enc =* alias
  while( $op->[8] =~ /\/([^\(\)\"\s]+)\:([\w\.]+)/pg ) {
    my $alias = $2;
    my $what;
    my $v = check_single_enum($1);
    if ( defined($v) && defined($what = check_enc_ask($op->[5], $2)) ) {
       $rem{$what->[0]} = 1;
       my $p = pos($op->[8]);
       push @pos, [ $p - length(${^MATCH}), $p ];
       mask_value(\@res, $v, $what);
    }
  }
  # check /Group(Value):alias in format - Value can contain /PRINT suffix
  while ( $op->[8] =~ /\/(\w+)\(\"?([^\"\)\/)]+)\"?(\/PRINT)?\)\:([\w\.]+)/pg ) {
      if ( exists $Tabs{$1} && exists $Tabs{$1}->{$2} ) {
        my $value = $Tabs{$1}->{$2};
        my $what = defined($opt_c) ? check_enc($op->[5], $1, $4) : check_enc_ask($op->[5], $4);
        if ( defined($what) ) {
          my $p = pos($op->[8]);
          # remove if no /PRINT prefix
          $rem{$what->[0]} = 1 if ( !defined($3) );
          push @pos, [ $p - length(${^MATCH}), $p ];
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
      my $what = defined($opt_c) ? check_enc($op->[5], $1, $4) : check_enc_ask($op->[5], $4);
# if ( $op->[1] eq 'F2F' ) { printf("HER %s %s %s\n", $1, $2, $4); }
      if ( defined($what) ) {
        # remove if no /PRINT prefix
        $rem{$what->[0]} = 1 if ( !defined($3) );
        my $p = pos($op->[8]);
        push @pos, [ $p - length(${^MATCH}), $p ];
        mask_value(\@res, $value, $what);
      }
  } }
  # and again check for ZeroRegister(RZ) in format - in worst case just assign it yet one more time
  while ( $op->[8] =~ /\bZeroRegister\(\"?RZ\"?\)\:([\w\.]+)/pg ) {
    my $what = check_enc($op->[5], $1, $1);
    if ( defined $what ) {
      $rem{$what->[0]} = 1;
      my $p = pos($op->[8]);
      push @pos, [ $p - length(${^MATCH}), $p ];
      mask_value(\@res, $g_rz, $what);
    }
  }
  # remove used formats and put new string at index 10
  if ( scalar @pos ) {
   my $cp = $op->[8];
   # sort in back order by offsets
   foreach ( sort { $b->[1] <=> $a->[0] } @pos ) {
     substr($cp, $_->[0], $_->[1] - $_->[0], '');
   }
   # remove empty {}
   $cp =~ s/\{\s*\}//g;
   # and $( )$
   $cp =~ s/\$\(\s*\)\$//g;
   $op->[22] = $cp;
  }
  remove_encs($op, \%rem) if ( scalar keys %rem );
  # process remained encodings
  if ( defined $opt_F ) {
    %rem = ();
    my $patched = 0;
    my $mae = $op->[11];
# printf("%s line %d\n", $op->[1], $op->[4]);
# printf("%s\n", join('', @res));
  foreach my $emask ( @{ $op->[5] } ) {
    next if ( $emask !~ /^([\w\.]+)\s*=(\*?)\s*([\w\.]+)/ ); # wtf? bad encoding?
    my $what = $g_mnames{$1};
    my $ast = $2 ne '';
    my $ename;
    if ( !exists $mae->{$1} ) {
      next unless exists $g_enums{$3};
      $ename = $3;
    } else {
      my $must_be = $mae->{$1};
      $ename = $must_be->[0];
      # printf("enc %s enum(%s) %d in %s\n", $1, $must_be->[0], $must_be->[2], $op->[1]) if ( defined $must_be->[2] );
    }
    my $v = check_single_enum($ename);
    if ( defined($v) ) {
      $rem{$ename} = 1;
      mask_value(\@res, $v, $what);
      $patched++;
    } elsif ( !$ast ) {
      $v = is_incomplete($what, $ename);
      if ( $v ) {
      # put this enum into filter at ->[12]
        if ( defined $op->[12] ) { push @{ $op->[12] }, [ $what, 'e', $g_enums{$ename}, $ename ]; }
         else { $op->[12] = [ [ $what, 'e', $g_enums{$ename}, $ename ] ]; }
      }
    }
  }
  remove_encs($op, \%rem) if ( scalar keys %rem );
# printf("%s %d patched\n", join('', @res), $patched) if ( $patched );
  }
  # and add tables
  foreach my $tmask ( @{ $op->[5] } ) {
    # mask $1 = (optional * $2) table $3 list of vars $4
    if ( $tmask =~ /^([\w\.]+)\s*=(\*)?\s*(\S+)\s*\(\s*([^\)]+)\s*\)/ ) {
      my $what = $g_mnames{$1};
      if ( exists($g_tabs{$3}) ) {
         my $tab_name = $3;
         $g_used_tabs{$3} //= 1;
         # example from sm55_1.txt:
         #   aSelect =* VFormat16(safmt,asel);
         # where /Integer16:safmt & /H1H0(H0):asel both enums
         my @tfilter = ( $what, 't', $g_tabs{$3}, $3 );
         my @e_args;
         my $e_cnt = 0;
         my $list = $4;
         my $me = $op->[16];
         # another example from sm90_1.txt
         #   BITS_4_80_77_mem=*TABLES_mem_2(sem,sco,0);
         # in formats /STRONGONLY:sem /SYSONLY:sco
         # both STRONGONLY & SYSONLY are single value enums
         # so if we can fill whole row with numerical values we could find key in table via rev_tab_lookup
         # and then make new mask $what with that key
         my @num_row;
         foreach my $arg ( split /\s*,\s*/, $list) {
           if ( exists $g_enums{$arg} ) {
             push @e_args, $arg; $e_cnt++;
             push @num_row, check_single_enum($arg);
           } elsif ( exists $me->{$arg} ) {
             push @e_args, $me->{$arg}->[0]; $e_cnt++;
             push @num_row, check_single_enum($me->{$arg}->[0]);
           } elsif ( $arg =~ /^\d+/ ) {
             push @num_row, int($arg);
           } else {
             push @e_args, undef;
             push @num_row, undef;
           }
         }
         my $num_cnt = 0;
         foreach ( @num_row) { $num_cnt++ if defined $_; }
         my $t_dim = tab_dim($tab_name);
         if ( defined($opt_z) && $num_cnt == $t_dim ) {
      printf("CNT %s line %d %s: %d %d tab %s %s\n", $op->[1], $op->[4], $what->[0], $num_cnt, $t_dim, $tab_name, join(' ', @num_row));    
           # try to find key
           my $key = (1 == $t_dim) ? rev_tab1($tab_name, $num_row[0]) : rev_tab_lookup($tab_name, \@num_row);
           if ( defined $key ) {
             printf("Can remove table mask %s, key %X\n", $what->[0], $key);
             mask_value(\@res, $key, $what);
           }
         }
         if ( $e_cnt ) {
           $tfilter[1] = 'T';
           # remove trailng undefs
           while( !defined $e_args[-1] ) { pop @e_args; }
           push @tfilter, $_ for ( @e_args );
         }
         if ( defined $op->[12] ) { push @{ $op->[12] }, \@tfilter; }
         else { $op->[12] = [ \@tfilter ] }
      } elsif ( $3 ne 'IDENTICAL' ) {
        printf("%s at line %d - table %s does not exist for %s\n", $op->[1], $op->[4], $3, $1);
      }
    }
  }
  my $cmin = calc_mean_bits(\@res);
  $g_min_len = $cmin if ( $cmin < $g_min_len );
  $op->[14] = $cmin;
  return join('', @res);
}

# args: list of encoders, instr name, instr line, hash name => [ enum ]
# return ref to hash encoder_name => [ [ enum_name, ref2enum ] ... or undef ]
sub add_tenums
{
  my($enc, $opname, $line, $er) = @_;
  my %res;
  foreach my $tmask ( @$enc ) {
    # mask $1 = table $2
    if ( $tmask =~ /^([\w\.]+)\s*=\*?\s*(\S+)\s*\(([^\)]+)\)/ ) {
      next if ( !exists($g_tabs{$2}) );
      $g_used_tabs{$2} //= 1;
      my @ar;
      my $ename = $1;
      my $cnt = 0;
      foreach my $em ( split /\s*,\s*/, $3 ) {
        if ( exists $er->{$em} ) {
         my $er = $er->{$em};
         push @ar, [ $er->[0], $g_enums{$er->[0]} ]; $cnt++;
        } else { push @ar, undef; }
      }
      next if !$cnt;
      $res{$ename} = \@ar;
    }
  }
  if ( 0 != scalar keys %res ) {
# printf("has te for %s line %d\n", $opname, $line);
    return \%res;
  }
  undef;
}

sub dump_tenums
{
  my $te = shift;
  return if !defined $te;
  foreach my $t ( sort keys %$te ) {
    printf("  te:%s(", $t);
    my $ar = $te->{$t};
    my $res = '';
    foreach my $e ( @$ar ) {
      $res .= defined($e) ? $e->[0] : '';
      $res .= ',';
    }
    chop $res;
    printf("%s)\n", $res);
  }
}

sub dump_filters
{
  my $op = shift;
  return unless ( defined $op->[12] );
  printf("filters:\n");
  my $flist = $op->[12];
  foreach my $f ( @$flist ) {
   if ( $f->[1] eq 'v' ) { printf("  %s %s %d\n", $f->[0]->[0], $f->[1], $f->[2]); }
   elsif ( $f->[1] eq 'T' ) {
     printf("  %s %s %s:", $f->[0]->[0], $f->[1], $f->[3]);
     my $tlen = scalar @$f;
     for ( my $i = 4; $i < $tlen; $i++ ) {
       printf(",") if ( $i != 4 );
       printf("%s", $f->[$i]) if ( defined $f->[$i] );
     }
     printf("\n");
   } else { printf("  %s %s %s\n", $f->[0]->[0], $f->[1], $f->[3]); }
  }
}

# args: mask array, instruction
sub filter_ins
{
  my($a, $op, $verb) = @_;
  return 1 unless ( defined $op->[12] );
  # format of op->[12] elements is array where indexes
  # 0 - mask
  # 1 - letter 'e' for enums, 'v' for values, 't' for tables, 'T' for tables with enums
  # 2 - ref to enum/value
  # 3 - name of enum/table for dumping
  # 4 ... for 'T' - names of enums to check value in them
  my $flist = $op->[12];
  foreach my $f ( @$flist ) {
    my $v = extract_value($a, $f->[0]);
    next if ( !defined $v );
    printf("v %X for mask %s\n", $v, $f->[0]->[0]) if ( defined $verb );
    if ( 'e' eq $f->[1] ) {
      return 0 unless ( defined enum_by_value($f->[2], $v) );
    } elsif ( 'v' eq $f->[1] ) {
      return 0 if ( $v != $f->[2] );
    } else {
      # check tabs in [2]
      my $tr = $f->[2];
      return 0 unless exists( $tr->{$v} );
      return 1 if ( 't' eq $f->[1] );
      # process T with optional enums in table values
      my $row = $tr->{$v};
      if ( 'ARRAY' eq ref($row) ) {
        for ( my $i = 0; $i < scalar @$row; $i++ ) {
          next unless defined($f->[4 + $i]);
          my $e = $g_enums{ $f->[4 + $i] };
          # check that enum in row[$i] exists
          return 0 unless ( defined enum_by_value($e, $row->[$i]) );
        }
      } else {
        my $e = $g_enums{ $f->[4] };
        return 0 unless ( defined enum_by_value($e, $row) );
      }
    }
  }
  1;
}

# return ref to mask if encoding has form mask=*alias
sub check_enc_ask
{
  my($e, $alias) = @_;
  $alias =~ s/\}\s*$//;
  $alias =~ s/\s*$//;
  for my $enc ( @$e ) {
    if ( $enc =~ /^(\S+)\s*=\s*\*\s*(\S+)$/ ) {
      if ( $2 eq $alias ) {
        return $g_mnames{$1} if exists $g_mnames{$1};
        return;
      }
    }
  }
  undef;
}

# return ref to mask from g_mnames by name or alias from list of encoders
sub check_enc
{
  my($e, $name, $alias) = @_;
  $alias =~ s/\}\s*$//;
# printf("check_enc %s %s\n", $name, $alias);
  for my $enc ( @$e ) {
    if ( $enc =~ /^(\S+)\s*=\s*\*?\s*(\S+)\s*/ ) {
      if ( $2 eq $name or $2 eq $alias ) {
         return $g_mnames{$1};
      }
    }
  }
  undef;
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
# [10] - const bank list
# [11] - hash with alias -> enum ref
# [12] - list of filters for this instruction
# [14] - count of meaningful bits
# [15] - string with unused formats
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
my(%g_masks, $g_dec_tree);
my $g_dups = 0;

# args: value, mask, reg to op->[11], ref to kv, format name, optional scale
sub dump_plain_value
{
  my($v, $mask, $mae, $kv, $fn, $sc) = @_;
# printf("dump_plain_value %s\n", $fn) if defined($fn);
  if ( defined($v) ) {
    printf("   %s(", $mask->[0]);
    if ( exists $mae->{$mask->[0]} ) {
      my $e = $mae->{$mask->[0]};
      my $s = enum_by_value($g_enums{$e->[0]}, $v);
      if ( defined $s ) {
        printf("%X) %s\n", $v,$s);
        $kv->{$fn} = [ $v, $s ] if defined($fn);
        return;
      }
    }
    $v *= $sc if ( defined $sc );
    if ( $v ) { printf("%X)\n", $v); }
    else { printf("0)\n"); }
    $kv->{$fn} = $v if defined($fn);
  }
}

# dump IDENTICAL
# args: value, ref to kv, string args in IDENTICAL
sub dump_id_value
{
  my($v, $kv, $args) = @_;
  return 0 if ( !defined($v) );
printf("id_value: %s\n", $args);
  foreach my $a ( split /\s*,\s*/, $args ) {
    $kv->{$a} = $v;
  }
  return 1;
}

sub strip_t
{
  my $s = shift;
  $s =~ s/^\s+//;
  $s =~ s/\s+$//;
  $s;
}

# args - ref to bit array, ref to found instruction, ref to kvalue map
sub dump_values
{
  my($a, $op, $kv) = @_;
  my $enc = $op->[5];
  foreach my $m ( @$enc ) {
    if ( $m =~ /^([\w\.]+)\s*=\*?\s*IDENTICAL\(([^\)]+)\)/ ) {
      my $mask = $g_mnames{$1};
      dump_id_value(extract_value($a, $mask), $kv, $2);
       # mask $1 = table $2 (args) $3
    } elsif ( $m =~ /^([\w\.]+)\s*=\*?\s*(\S+)\s*\((\s*[^\)]+)\s*\)/ ) {
      my $mask = $g_mnames{$1};
      my $v;
      if ( exists($g_tabs{$2}) && defined($v = extract_value($a, $mask)) ) {
        my $tab = $g_tabs{$2};
        if ( exists $tab->{$v} ) {
          my $row = $tab->{$v};
          printf("   %s(%X) %s = ", $mask->[0], $v, $3);
          if ( 'ARRAY' eq ref $row ) {
            my @fa = split /\s*,\s*/, $3;
            my $te = $op->[13];
            if ( defined($te) && exists $te->{$1} ) {
              my $res = '';
              my $ae = $te->{$1};
              for ( my $i = 0; $i < scalar @$row; $i++ ) {
                if ( !defined $ae->[$i] ) { $res .= $row->[$i];
                  $kv->{ $fa[$i] } = $row->[$i] if ( $i < scalar @fa );
                } else {
                  $res .= $ae->[$i]->[0] . '(' . $row->[$i] . ')';
                  $v = enum_by_value($ae->[$i]->[1], $row->[$i]);
                  $kv->{ $fa[$i] } = [ $row->[$i], $v ] if ( $i < scalar @fa && defined($v) );
                  $res .= $v if defined $v;
                }
                $res .= ',';
              }
              chop $res;
              printf("%s\n", $res);
            } else { printf("%s\n", join ",", @$row);
              for ( my $i = 0; $i < scalar @$row; $i++ ) {
                $kv->{ $fa[$i] } = $row->[$i];
              }
            }
          } else { # table with single arg
            printf("%s\n", $row); $kv->{strip_t($3)} = $row; }
        } else {
          printf("   %s value %X does not exists in table %s\n", $mask->[0], $v, $2);
        }
      } else {
        dump_plain_value(extract_value($a, $mask), $mask, $op->[11], $kv);
      }
    } elsif ( $m =~ /^([\w\.]+)\s*=\*?\s*([\w\.\@]+)(?:\s*SCALE\s+(\d+))?$/ ) {
      my $mask = $g_mnames{$1};
      my $scale;
      $scale = int($3) if defined($3);
      dump_plain_value(extract_value($a, $mask), $mask, $op->[11], $kv, $2, $scale);
      # sm86 has strange formats like BITS_16_63_48_Sc=Sb convertFloatType(ofmt == `OFMT@E8M7_V2 || ofmt == `OFMT@BF16_V2, E8M7Imm, ofmt == `OFMT@E6M9_V2, E6M9Imm, F16Imm)
    } elsif ( $m=~ /^([\w\.]+)\s*=\*?\s*([\w\.\@]+)\s+convertFloatType/ ) {
      my $mask = $g_mnames{$1};
      dump_plain_value(extract_value($a, $mask), $mask, $op->[11], $kv, $2);
    } else {
     carp("unknown encoding $m");
   }
  }
  # dump const bank
  if ( defined $op->[10] ) {
    my $cb = $op->[10];
    printf(" -- const bank %s\n", $cb->[0]);
    my $cb_len = scalar @$cb;
    my @fcb;
    if ( $cb->[0] =~ /\(\s*(.*),\s*(.*)\)/ ) {
      push @fcb, $1; push @fcb, $2;
    }
    for ( my $i = 1; $i < $cb_len; $i++ ) {
      my $mask = $g_mnames{$cb->[$i]};
      dump_plain_value(extract_value($a, $mask), $mask, $op->[11], $kv, $fcb[$i-1]);
    }
  }
}

sub lookup_mask
{
  my($op, $ae) = @_;
  return unless defined($op->[11]);
  my $m2e = $op->[11];
  keys %$m2e;
  while( my($m, $e) = each %$m2e ) {
   return $m if ( $e->[0] eq $ae->[0] );
  }
  undef;
}

# args: instruction, ref to kv, ref to enum, ref to array with bits
sub lookup_value
{
  my($op, $kv, $ae, $b) = @_;
  if ( exists $kv->{$ae->[3]} ) { return  $kv->{$ae->[3]}; }
  elsif ( exists $kv->{$ae->[0]} ) { return $kv->{$ae->[0]}; }
  # for some unknown reason we still don't have this value - let's read it from $b
  # but first we need to extract mask
  my $mask = lookup_mask($op, $ae);
  if ( !defined $mask ) {
    printf("format enum: no value for %s format %s\n", $ae->[0], $ae->[3]);
    return;
  }
  my $res = extract_value($b, $g_mnames{$mask});
  # cache readed value in kv
  $kv->{$ae->[0]} = $res if defined($res);
  $res;
}

# dump /enum(value) == value and so should be ignored
# args: ref to array made in cons_ae function, ref to kv
sub ignore_enum
{
  my($op, $ae, $kv, $b) = @_;
  return unless ( defined $ae->[2] ); # no default value
  my $v = lookup_value($op, $kv, $ae, $b);
# printf("ignore_enum %s\n", $ae->[0]);
  if ( !defined $v ) {
    printf("is_ignore_enum: no value for %s format %s\n", $ae->[0], $ae->[3]);
    return;
  }
  my $res = '';
  $res = '.' if $ae->[1];
  if ( 'ARRAY' ne ref $v ) {
  printf("%s %d: single %d vs %d\n", $ae->[0], $ae->[1], $v, $ae->[2]) if defined($opt_v);
    return [ 0, '' ] if ( $v == $ae->[2] ); # default value
    my $ev = enum_by_value($g_enums{$ae->[0]}, $v);
    return [ 1, $res . $ev ] if ( defined ( $ev ) );
    return;
  }
  # value is pair [ int, enum_string ]
printf("%s %d: pair %s vs %d\n", $ae->[0], $ae->[1], $v->[1], $ae->[2]) if defined($opt_v);
  return [ 0, '' ] if ( $v->[0] == $ae->[2] ); # default value
  return [ 1, $res . $v->[1] ];
}

sub dump_formats
{
  my $op = shift;
  return unless defined($op->[15]);
  foreach my $f ( @{$op->[15]} ) {
    if ( $f->[0] eq '$' ) { printf('$' . "\n"); next; }
    printf("%s %s%s%s", $f->[0], $f->[1] || ' ', $f->[2] || ' ', $f->[3] || ' ');
    if ( $f->[0] eq 'P' ) {
      my $ae = $f->[4];
      printf(" %s %s", $ae->[0], $ae->[3]);
    } elsif ( $f->[0] eq 'E' ) {
      my $ae = $f->[4];
      printf(" %s %d %s %s", $ae->[0], $ae->[1], $ae->[2] || '', $ae->[3]);
    } elsif ( $f->[0] eq 'V' ) {
      printf(" %s %s", $f->[4], $f->[5]);
    } elsif ( $f->[0] eq 'D' ) { # DESC: [4][5 /6 + 7], 6 can be optional enum
      my $e4 = $f->[4];
      my $e5 = $f->[5];
      if ( defined $f->[6] ) {
        my $ae = $f->[6];
        printf(" [%s %s %s][%s %s %s%s:%s + %s]", $e4->[0], $e4->[3], defined($e4->[2]) ? $e4->[2] : '',
          $e5->[0], $e5->[3], defined($ae->[1]) ? '/' : '', $ae->[0], $ae->[3], $f->[7]);
      } else {
        printf(" [%s %s %s][%s %s + %s]", $e4->[0], $e4->[3], defined($e4->[2]) ? $e4->[2] : '', $e5->[0], $e5->[3], $f->[7]);
      }
    } elsif ( $f->[0] eq 'C' || $f->[0] eq 'X' ) {
      # const bank can have 2 or 3 op - 3rd is ref to Enum
      printf(" %s: ", $f->[4]); # C:name
      # $f->[5] can be ea
      if ( 'ARRAY' eq ref $f->[5] ) {
        my $ae = $f->[5];
        printf("%s %s", $ae->[0], $ae->[3]);
      } else {
        printf("%s", $f->[5]);
      }
      if ( defined $f->[7] ) {
        my $ae = $f->[7];
        printf(" %s + %s %s", $f->[6], $ae->[0], $ae->[3]);
      } else {
        printf(" %s", $f->[6]);
      }
    } elsif ( $f->[0] eq 'A' or $f->[0] eq '[' ) {
      for ( my $i = 4; $i < scalar @$f; $i++ ) {
        if ( 'ARRAY' eq ref $f->[$i] ) {
          my $ae = $f->[$i];
          printf(" %s %s", $ae->[0], $ae->[3]);
        } elsif ( $f->[$i] eq '+' ) { printf(" +"); }
        else { printf(" %s", $f->[$i]); }
      }
    } elsif ( $f->[0] eq 'T' ) {
      printf(" [%s]", $f->[4]);
    } elsif ( $f->[0] eq 'M1' ) {
      my $ae = $f->[4];
      printf(" [%s %s]", $ae->[0], $ae->[3]);
    } elsif ( $f->[0] eq 'M2') {
      my $ae = $f->[4];
      printf(" [%s %s", $ae->[0], $ae->[3]);
      printf(" + %s", $f->[5] ) if ( defined $f->[5] );
      printf("]");
    }
    printf("\n");
  }
}

sub format_enum
{
  my($ae, $op, $kv, $b) = @_;
  my $sv = '';
  # format enum
  if ( defined $ae->[1] ) {
    my $part = ignore_enum($op, $ae, $kv, $b);
    return $part->[1] if ( defined $part );
  }
  if ( defined $ae->[2] ) {
    my $tmp = ignore_enum($op, $ae, $kv, $b);
    $sv = $tmp->[1] if ( defined $tmp );
  } else {
    my $v = lookup_value($op, $kv, $ae, $b);
# printf("lookup %s returned %s\n", $ae->[0], defined($v) ? $v : 'undef');
    if ( defined($v) ) {
      $sv = '.' if $ae->[1];
      return $sv . $v->[1] if ( 'ARRAY' eq ref $v );
      my $ev = enum_by_value($g_enums{$ae->[0]}, $v);
      if ( defined $ev ) { $sv .= $ev; }
      else { $sv = ''; }
    }
  }
  $sv;
}

sub move_last_commas
{
  my $flist = shift;
  for ( my $i = 0; $i < scalar(@$flist) - 1; $i++ ) {
    my $f = $flist->[$i];
    next if ( $f->[0] eq '$');
    if ( defined($f->[2]) ) { # check if we have suffix
      my $next = $flist->[$i + 1];
      if ( !defined $next->[1] ) {
        $next->[1] = $f->[2];
        $f->[2] = undef;
      }
    }
  }
}

# print iinstruction based on format list in op->[15]
# args: ref to op, ret to kv collected in dump_values
# formats
# [!] - predicate + predicate@not
# [~] - x@invert
# [-] - x@negate
# [||] - x@absolute
# /SomeEnum(Default):alias means .SomeEnum{$value} if $value != SomeEnum{default}
# format item is array where indexes
# 0 - type
# 1 - prefix symbol
# 2 - suffix symbol
# 3 - [x] if presents
# 4 - name of alias to search in kv/ref to enum
sub make_inst
{
  my($op, $kv, $b) = @_;
  my $res = '';
  my $flist = $op->[15];
  return $res unless defined $flist;
  foreach my $f ( @$flist ) {
    my $ae = $f->[4];
    if ( $f->[0] eq 'P' ) { # predicate
      my $pnot;
      my $part = ignore_enum($op, $ae, $kv, $b);
      next if ( !$part->[0] );
      # make alias@not
      if ( defined($f->[3]) && $f->[3] eq '!' ) {
        $pnot = $ae->[3] . '@not';
      }
      my $pres = $f->[1]; # prefix
      if ( defined($pnot) && exists $kv->{$pnot} ) {
        $pres .= '!' if ( $kv->{$pnot} );
      }
      $pres .= $part->[1];
      $res .= $pres . ' ';
      $res .= $f->[2] if ( defined $f->[2] ); # suffix
      next;
    } elsif ( $f->[0] eq 'E' ) { # some enum
      my $ae = $f->[4];
      my $part;
      if ( defined $ae->[1] ) {
        my $part = ignore_enum($op, $ae, $kv, $b);
        if ( defined $part ) {
          next if ( !$part->[0] );
          $res .= $f->[1] if ( defined $f->[1] ); # prefix
          $res .= $part->[1];
          $res .= $f->[2] if ( defined $f->[2] ); # suffix
          next;
        }
        # this enum has non-default value
# printf("%s: no part %d\n", $ae->[0], $ae->[1]);
        my $v = lookup_value($op, $kv, $ae, $b);
        next unless ( defined $v );
        if ( $ae->[1] ) { $res .= '.' }
        else { $res .= ' '; }
        $res .= $f->[1] if ( defined $f->[1] ); # prefix
        # check placeholders
        if ( defined $f->[3] ) {
         if ( $f->[3] eq '!') {
           my $pneg = $ae->[3] . '@not';
           $res .= '!' if ( exists($kv->{$pneg}) && $kv->{$pneg} );
         }
         # check [-]
         elsif ( $f->[3] eq '-') {
           my $pneg = $ae->[3] . '@negate';
           $res .= '-' if ( exists($kv->{$pneg}) && $kv->{$pneg} );
         }
         # check [~]
         elsif ( $f->[3] eq '~') {
           my $pneg = $ae->[3] . '@invert';
           $res .= '~' if ( exists($kv->{$pneg}) && $kv->{$pneg} );
         }
        }
        if ( 'ARRAY' ne ref $v ) {
          my $ev = enum_by_value($g_enums{$ae->[0]}, $v);
          if ( defined $ev ) { $res .= $ev; }
          else {
            printf("cannot find value %d for enum %s\n", $v, $ae->[0]);
            $res .= $v;
          }
        } else {
          $res .= $v->[1];
        }
        $res .= $f->[2] if ( defined $f->[2] ); # suffix
      }
    } elsif ( $f->[0] eq '$' ) { # opcode
      $res .= $op->[1];
      next;
    } elsif ( $f->[0] eq '[' ) {
      $res .= $f->[1] if ( defined $f->[1] ); # prefix
      $res .= ' [';
      for ( my $i = 4; $i < scalar @$f; $i++ ) {
        if ( 'ARRAY' eq ref $f->[$i] ) {
          my $sv = format_enum($f->[$i], $op, $kv, $b);
          $res .= $sv if defined($sv);
        } elsif ( '+' eq $f->[$i] ) { $res .= ' + '; }
        else {
          if ( exists $kv->{$f->[$i]} ) {
            my $v = $kv->{$f->[$i]};
            $res .= sprintf("0x%X", $v);
          } else {
            $res .= sprintf("cannot find value %s", $f->[$i]);
          }
        }
      }
      $res .= ']';
      next;
    } elsif ( $f->[0] eq 'A' ) {
      $res .= $f->[1] if ( defined $f->[1] ); # prefix
      $res .= 'attr[';
      for ( my $i = 4; $i < scalar @$f; $i++ ) {
        if ( 'ARRAY' eq ref $f->[$i] ) {
          my $sv = format_enum($f->[$i], $op, $kv, $b);
          $res .= $sv if defined($sv);
        } elsif ( '+' eq $f->[$i] ) { $res .= ' + '; }
        else {
          if ( exists $kv->{$f->[$i]} ) {
            my $v = $kv->{$f->[$i]};
            $res .= sprintf("0x%X", $v);
          } else {
            $res .= sprintf("cannot find A value %s", $f->[$i]);
          }
        }
      }
      $res .= ']';
      next;
    } elsif ( $f->[0] eq 'D' ) { # dest:[4][5 opt6 + 7], where 4,5 & 6 - enums and 7 - value
      $res .= $f->[1] if ( defined $f->[1] ); # prefix
      $res .= 'desc[';
      # 1st bank
      my $sv = format_enum($f->[4], $op, $kv, $b);
      $res .= $sv if defined($sv);
      $res .= '][';
      # 2nd bank
      $sv = format_enum($f->[5], $op, $kv, $b);
      $res .= $sv if defined($sv);
      if ( defined($f->[6]) ) {
        my $sv = format_enum($f->[6], $op, $kv, $b);
        if ( $sv ) { $res .= $sv; }
      }
      $res .= ' + ';
      if ( exists $kv->{$f->[7]} ) {
        my $v = $kv->{$f->[7]};
        $res .= sprintf("0x%X]", $v);
      } else {
        $res .= sprintf("cannot find bank value %s]", $f->[7]);
      }
      next;
    } elsif ( $f->[0] eq 'T' ) { # TTU
      $res .= $f->[1] if ( defined $f->[1] ); # prefix
      $res .= 'TTU[';
      if ( exists $kv->{$f->[4]} ) {
        my $v = $kv->{$f->[4]};
        $res .= sprintf("0x%X]", $v);
      } else {
        $res .= sprintf("cannot find TTU value %s]", $f->[4]);
      }
      next;
    } elsif ( $f->[0] eq 'M2' ) { # TMEM
      $res .= $f->[1] if ( defined $f->[1] ); # prefix
      $res .= $f->[3]; # name of this TMEM
      $res .= '[';
      my $sv = format_enum($f->[4], $op, $kv, $b);
      $res .= $sv if defined($sv);
      if ( defined $f->[5] ) {
        if ( exists $kv->{$f->[5]} ) {
          my $v = $kv->{$f->[5]};
          $res .= sprintf("+ 0x%X]", $v);
        } else {
          $res .= sprintf("cannot find TMEM value %s]", $f->[5]);
        }
      } else {
        $res .= ']';
      }
    } elsif ( $f->[0] eq 'M1' ) { # U|G MMA
      $res .= $f->[1] if ( defined $f->[1] ); # prefix
      $res .= $f->[3]; # name of this MMA
      $res .= '[';
      my $sv = format_enum($f->[4], $op, $kv, $b);
      $res .= $sv if defined($sv);
      $res .= ']';
      next;
    } elsif ( $f->[0] eq 'C' || $f->[0] eq 'X' ) { # const bank
      my($v, $pfx);
      $pfx = substr($op->[10]->[0], 0, 1) if ( $f->[0] eq 'C' );
      $res .= $f->[1] if ( defined $f->[1] ); # prefix
      if ( defined($f->[3]) and defined($f->[4]) ) {
        # check [~]
        if ( $f->[3] eq '~' ) {
          my $pinv = $f->[4] . '@invert';
          $res .= '~' if ( exists($kv->{$pinv}) && $kv->{$pinv} );
        } elsif ( $f->[3] eq '-' ) {
          my $pneg = $f->[4] . '@negate';
          $res .= '-' if ( exists($kv->{$pneg}) && $kv->{$pneg} );
        }
      }
      $res .= 'C[';
      # sa_bank
      if ( exists $kv->{$f->[5]} ) {
        $v = $kv->{$f->[5]};
        $res .= sprintf("0x%X]", $v);
      } else {
        $res .= sprintf("cannot find bank value %s]", $f->[5]);
      }
      # address can consist from 2 parts
      if ( !defined($f->[7]) ) {
        if ( exists $kv->{$f->[6]} ) {
          $v = $kv->{$f->[6]};
          if ( $f->[0] eq 'C' ) {
            $res .= sprintf("[0x%X]", $pfx eq '2' ? $v * 4 : $v);
          } else {
            $res .= sprintf("[0x%X]", $v);
          }
        } else { $res .= sprintf("[cannot find cvalue %s]", $f->[6]); }
      } else {
        # reg in $f->[7], imm in $f->[6]
        my $sv = format_enum($f->[7], $op, $kv, $b);
        $res .= '[';
        if ( $sv ) {
          $res .= $sv;
          if ( $f->[0] eq 'C') {
            $res .= ' *4 ' if ( $pfx eq '2' );
          }
          $res .= '+';
        }
        # and final imm part
        if ( exists $kv->{$f->[6]} ) {
          $v = $kv->{$f->[6]};
          if ( $f->[0] eq 'C') {
            $res .= sprintf("0x%X]", $pfx eq '2' ? $v * 2 : $v);
          } else {
            $res .= sprintf("0x%X]", $v);
          }
        } else { $res .= sprintf(" cannot find cvalue %s]", $f->[6]); }
        $res .= $f->[2] if ( $f->[2] ); # suffix
      }
    } elsif ( $f->[0] eq 'V' ) { # some value
      if ( exists $kv->{$f->[4]} ) {
        my $v = $kv->{$f->[4]};
        $res .= $f->[1] if ( defined $f->[1] ); # prefix
        $res .= sprintf(" 0x%X", $v);
        $res .= $f->[2] if ( defined $f->[2] ); # suffix
      } else {
        printf("Value %s not exists\n", $f->[4]);
      }
    }
  }
  return $res;
}

# try to find mask from $opt_N
sub make_single_test
{
  my @nb = split(//, $opt_N);
  if ( $g_size != scalar @nb ) {
    carp("bad size of -N arg, should be $g_size");
    return;
  }
  my @fout;
  if ( defined $opt_B ) {
    my @res;
    find_in_dectree($g_dec_tree, \@nb, \@res);
    # dump results
    printf("found %d\n", scalar @res);
    foreach my $r ( @res ) {
      printf("%s\n", $r);
      # cmp with mask
      push @fout, $r if ( cmp_maska($r, \@nb));
    }
  } else {
    foreach my $m ( keys %g_masks ) {
      push @fout, $m if ( cmp_maska($m, \@nb) );
    }
  }
  printf("matched: %d\n", scalar @fout);
  # dump matched masks
  foreach my $r ( @fout ) {
    printf("%s\n", $r);
    my $ops = $g_masks{$r};
    foreach my $co ( @$ops ) {
      printf("%s line %d %s\n", $co->[1], $co->[4], $co->[7] ? 'ALT' : '');
      dump_filters($co);
      next if ( !filter_ins(\@nb, $co, 1) );
      printf("MATCH\n");
    }
  }
}

sub make_test
{
  my $fn = shift;
  my($fh, $b);
  my $cmp_cnt = 0;
  my $filter_cnt = 0;
  my $filtered_cnt = 0;
  my $processed = 0;
  my $readed = 0;
  open($fh, '<', $fn) or die("cannot open $fn, error $!");
  my $cf;
  if ( $g_size == 64 ) { $cf = old64($fh); }
  elsif ($g_size == 88) { $cf = martian88($fh); }
  elsif ($g_size == 128) { $cf = read128($fh); }
  else { carp("unknown width $g_size"); return 0; }
  while( defined($b = $cf->()) ) {
    $readed++;
    printf("%s:", join '', @$b);
    # try to find in all masks - very slow
    my $found = 0;
    my $process = sub {
     my($op, $m) = @_;
     printf("\n") if ( !$found );
     printf("%s - ", $m);
     printf("%s line %d %d bits %s\n", $op->[1], $op->[4], $op->[14], $op->[7] ? 'ALT' : '');
     dump_filters($op);
     # extract all masks values
     dump_mask2enum($op);
     dump_tenums($op->[13]) if defined($op->[13]);
     my %kv;
     dump_values($b, $op, \%kv);
     if ( defined $opt_i ) {
       dump_formats($op);
       if ( keys %kv ) {
         printf("KV:");
         while( my($k, $v) = each %kv ) {
           if ( 'ARRAY' eq ref $v ) {
             printf(" %s:%d(%s)", $k, $v->[0], $v->[1]);
           } else { printf(" %s:%X", $k, $v); }
         }
         printf("\n");
       }
     }
     printf("%s\n", make_inst($op, \%kv, $b));
     $found++;
    };
    if ( defined $opt_B ) {
      my @res;
      my @fout;
      find_in_dectree($g_dec_tree, $b, \@res);
      foreach my $m ( @res ) {
        $cmp_cnt++;
        next if ( !cmp_maska($m, $b) );
        my $ops = $g_masks{$m};
        foreach my $co ( @$ops ) {
          $filter_cnt++;
          next if ( !filter_ins($b, $co) );
          $filtered_cnt++;
          push @fout, [ $co, $m ];
        }
      }
      printf("(%d/%d) ", scalar(@fout), scalar(@res));
      $process->($_->[0], $_->[1]) for @fout;
    } else {
    foreach my $m ( keys %g_masks ) {
      $cmp_cnt++;
      if ( cmp_maska($m, $b) ) {
        my $ops = $g_masks{$m};
        $filter_cnt++;
        next if ( !filter_ins($b, $ops->[0]) );
        $filtered_cnt++;
        $process->($ops->[0], $m);
        # last; # find first mask
      }
    } }
    $processed++ if ( $found );
    printf(" NOTFound\n") if ( !$found );
  }
  close $fh;
  printf("readed %d, processed %d\n", $readed, $processed);
  printf("%d cmp_maska, %d filter_ins, %d filtered\n", $cmp_cnt, $filter_cnt, $filtered_cnt);
}

sub dump_menums
{
 my $m2e = shift;
    # for diffing dump mask->enum in sorted order
    # while( my($m, $e) = each %$m2e ) {
 foreach my $m ( sort keys %$m2e ) {
   my $e = $m2e->{$m};
   if ( defined $e->[2] ) { printf(" %s->%s(%d)", $m, $e->[0], $e->[2]); }
   else { printf(" %s->%s", $m, $e->[0]); }
  }
  printf("\n");
}

sub dump_mask2enum
{
  my $op = shift;
  if ( defined $op->[11] ) {
    printf('mask2enum:');
    dump_menums($op->[11]);
  }
  return unless defined($op->[16]);
  printf('missed:');
  dump_menums($op->[16]);
}

sub in_missed
{
  my($op, $ename) = @_;
  return unless defined($op->[16]);
  my $m2e = $op->[16];
  keys %$m2e;
  while( my($m, $e) = each %$m2e ) {
    return $e if ( $e eq $ename );
  }
  undef;
}

sub dump_dup_masks
{
  foreach my $v ( sort { $a cmp $b } keys %g_masks ) {
    my $ops = $g_masks{$v};
    my $size = scalar @$ops;
    if ( 1 == $size ) {
      printf("%s: %s line %d\n", $v, $ops->[0]->[1], $ops->[0]->[4]);
      next;
    }
    printf("%s: %d items\n", $v, $size);
    # dump duplicated instructions
    my $name1 = $ops->[0]->[1];
    foreach my $op ( @$ops ) {
      if ( defined($opt_f) ) {
        my @x;
        printf("%s\n", get_filled_mask($op, $v, \@x));
        if ( scalar @x ) {
          printf("X:");
          printf(" %s", $_->[0]) for @x;
          printf("\n");
        }
      }
      if ( $name1 ne $op->[1] ) {
        printf(" !!%s line %d %s\n", $op->[1], $op->[4], $op->[8]);
      } else {
        printf("   %s line %d %s\n", $op->[1], $op->[4], $op->[8]);
      }
      printf("   Unused %s\n", $op->[22]) if defined($op->[22]);
      # dump mask to enum mapping
      dump_mask2enum($op);
      dump_tenums($op->[13]) if defined($op->[13]);
      dump_filters($op);
      # dump encodings
      printf("    %s\n", $_) for @{ $op->[5] };
      # dump constant banks
      if ( defined $op->[10] ) {
       printf("    %s\n", $_) for @{ $op->[10] };
      }
    }
  }
}

sub insert_mask
{
  my($cname, $op) = @_;
  # put class name to op
  unshift @$op, $cname;
  # skip altername classes without -a option
  # <()> return if ( $op->[7] && !defined($opt_a) );
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

###
# tree build logic
# hard to google ISDL or LISA algos
# this one is pretty good: https://past.date-conference.com/proceedings-archive/2016/pdf/0066.pdf
###

# leaf node is just list like
#  [ 'L', [ array with masks ] ]
# node with children:
#  [ 'M', index_of_bit, ptr2left_node0, ptr2right_node1, [ array with masks ]
# arrray with masks can occurs if level of current node >= $g_min_len (in other words we have enough meaningful bits)
#  and some mask(s) gave match with current incomplete mask inside cmpa_mask
# args:
#  a - array ref to current mask, cannot be shared by all levels bcs it patched at index_of_bit for left 0, for right 1
#  u - array ref to used masks indexes to ignore, for children add currently found bit so also cannot be shared
#  rem - array of string with masks to place into tree
#  level - nesting level
#  hand 0 if left, 1 if right - for debugging
sub build_node
{
  my($a, $u, $rem, $lvl, $hand) = @_;
  my $cnt = scalar @$rem;
  return if ( !$cnt );
  # 1) build array to count bits
  my @bits;
  my @masks;
  my @node_masks;
  for ( my $i = 0; $i < $g_size; $i++ ) { push @bits, $u->[$i] ? undef : [ 0, 0, 0 ]; }
  foreach my $kmask ( @$rem ) {
    if ( $lvl >= $g_min_len && cmpa_mask($a, $kmask) ) { push @node_masks, $kmask; next; }
    else { push @masks, $kmask; }
    for ( my $i = 0; $i < $g_size; $i++ ) {
      next if ( !defined $bits[$i] );
      my $letter = substr($kmask, $i, 1);
      if ( $letter eq '0' ) { $bits[$i]->[0]++; }
      elsif ( $letter eq '1' ) { $bits[$i]->[1]++; }
      else { $bits[$i]->[2]++; }
    }
  }
  return [ 'L', \@node_masks ] if ( !scalar @masks );
  # 2) we have some remainded masks, calc best bit position - like share of 0 or 1 / total is max
  $cnt = scalar @masks;
  my $curr_idx = -1;
  my $curr_max = 0.0;
  my $i = 0;
  foreach my $b ( @bits ) {
    if ( !defined $b ) { $i++; next; }
    if ( $b->[0] == $cnt || $b->[1] == $cnt || $b->[2] == $cnt ) { $i++; next; }
    # ignore zeros
    if ( !$b->[0] || !$b->[1] ) { $i++; next; }
    my $f0 = $b->[0] * 1.0 / $cnt;
    my $f1 = $b->[1] * 1.0 / $cnt;
    # find min($f0, $f1)
    $f0 = $f1 if ( $f1 < $f0 );
    if ( $f0 > $curr_max ) { $curr_max = $f0; $curr_idx = $i; }
# printf("%d f0 %f f1 %f curr_max %f\n", $i, $f0, $f1, $curr_max);
    $i++;
  }
  if ( -1 == $curr_idx ) {
    # no best bit, we could choice first for 0 or 1
    # but this lead to deep chain of flipping left or right nodes with only child
    # you can check this uncommenting bits dump at end
    # so stop madness and just add all remained masks to leaf node
    # for ( my $i = 0; $i < $g_size; $i++ ) {
    #  my $b = $bits[$i];
    #  next if ( !defined $b );
    #  if ( $b->[0] ) { $curr_idx = $i; last; }
    #  if ( $b->[1] ) { $curr_idx = $i; last; }
    # }
    # if ( -1 == $curr_idx ) {
      if ( defined $opt_v ) {
        printf("%d level %d %d cnt %d curr_max %f rem %d\n", $hand, $lvl, $i, $cnt, $curr_max, scalar @$rem);
        printf("%s MYMASK\n", join '', @$a);
        foreach my $rmask ( @$rem ) {
         printf("%s\n", $rmask);
        }
      }
      # dump bits
#      for ( my $i = 0; $i < $g_size; $i++ ) {
#        my $b = $bits[$i];
#        next if ( !defined $b );
#        printf("%d bits %d %d %d\n", $i, $b->[0], $b->[1], $b->[2]);
#      }
      return [ 'L', $rem ];
  }
  # 3) split to 3 array on bit $curr_idx
  my(@left, @right, @both);
  foreach my $cm ( @masks ) {
    my $letter = substr($cm, $curr_idx, 1);
    if ( $letter eq '0' ) { push @left, $cm; }
    elsif ( $letter eq '1' ) { push @right, $cm; }
    else { push @both, $cm; }
  }
  # 4) make new mask & u
  my @new_a = @$a;
  my @new_u = @$u;
  $new_u[$curr_idx] = 1;
  # 5) left node - @lest || $both, patch new_a[curr_idx] to 0
  my @this_node = ( 'M', $curr_idx, undef, undef );
  if ( scalar(@left) ) {
    push @left, @both if ( scalar @both );
    $new_a[$curr_idx] = '0';
    $this_node[2] = build_node(\@new_a, \@new_u,\@left, $lvl + 1, 0);
  }
  # 6) right node - @right || @both, patch new_a[curr_idx] to 1
  if ( scalar(@right) ) {
    push @right, @both if ( scalar @both );
    $new_a[$curr_idx] = '1';
    $this_node[3] = build_node(\@new_a, \@new_u,\@right, $lvl + 1, 1);
  }
  # that's all folks
  push @this_node, \@node_masks;
  return \@this_node;
}

sub build_tree
{
  # 1) lets exclude bits with the same valus in all instructions
  # indexes are  0  1  X
  my @bits;
  for ( my $i = 0; $i < $g_size; $i++ ) { push @bits, [ 0, 0, 0]; }
  my $cnt = 0;
  while( my($kmask, $op) = each(%g_masks) ) {
    my @ma = split //, $kmask;
    $cnt++;
    for ( my $i = 0; $i < $g_size; $i++ ) {
      if ( $ma[$i] eq '0' ) { $bits[$i]->[0]++; }
      elsif ( $ma[$i] eq '1' ) { $bits[$i]->[1]++; }
      else { $bits[$i]->[2]++; }
    }
  }
  # 2) make initial mask and used
  my @init_a = ( 'X' ) x $g_size;
  my @init_u = ( 0 ) x $g_size;
  for ( my $i = 0; $i < $g_size; $i++ ) {
    my $b = $bits[$i];
    if ( $b->[0] == $cnt ) {
      $init_a[$i] = '0';
      $init_u[$i] = 1;
      next;
    }
    if ( $b->[1] == $cnt ) {
      $init_a[$i] = '1';
      $init_u[$i] = 1;
      next;
    }
    if ( $b->[2] == $cnt ) {
      $init_u[$i] = 1;
      next;
    }
  }
  # and finally build whole tree
  my @all = keys %g_masks;
  my $res = build_node(\@init_a, \@init_u, \@all, 0, 2);
  # 0 - lead nodes, 1 - nodes, 2 - max nesting level
  my @stat = ( 0, 0, 0, 0 );
#  print Dumper($res);
  dump_decision_node($res, 0, 'C', \@stat);
  printf("%d nodes, %d leaves, max level %d total masks %d\n", $stat[0], $stat[1], $stat[2], $stat[3]);
  $res;
}

# main horror - try to find mask array b in decision tree starting from node n, curr is result set
sub find_in_dectree
{
  my($n, $b, $curr) = @_;
  if ( $n->[0] eq 'L' ) {
    push @$curr, @{ $n->[1] } if ( scalar @{ $n->[1] } );
    return 1;
  }
  push @$curr, @{ $n->[4] } if ( scalar @{ $n->[4] } );
  my $bit = $b->[$n->[1]];
  if ( !$bit ) {
    return 0 if ( !defined $n->[2] );
    return find_in_dectree($n->[2], $b, $curr);
  } else {
    return 0 if ( !defined $n->[3] );
    return find_in_dectree($n->[3], $b, $curr);
  }
}

# for debugging of decision tree
sub dump_decision_node
{
  my($n, $lvl, $hand, $st) = @_;
  return if ( !defined $n );
  $st->[2] = $lvl if ( $lvl > $st->[2] );
  if ( $n->[0] eq 'L' ) {
    $st->[0]++;
    $st->[3] += scalar @{ $n->[1] } if defined($n->[1]);
    printf("lvl %d %s masks %d\n", $lvl, $hand, defined($n->[1]) ? scalar @{ $n->[1] } : -1 ) if defined($opt_v);
    return;
  }
  $st->[1]++;
  $st->[3] += scalar @{ $n->[4] } if ( defined($n->[4]) );
  printf("lvl %d %s bit %d masks %d\n", $lvl, $hand, $n->[1], defined($n->[4]) ? scalar @{ $n->[4] } : -1 ) if defined($opt_v);
  dump_decision_node( $n->[2], $lvl + 1, 'L', $st ) if defined $n->[2];
  dump_decision_node( $n->[3], $lvl + 1, 'R', $st ) if defined $n->[3];
}

# C++ generator logic
sub dump_matched
{
  my($fh, $list) = @_;
  return unless defined($list);
  foreach my $m ( @$list ) {
    my $ops = $g_masks{$m};
    foreach my $op ( @$ops ) {
      printf($fh "&%s_%d,", $opt_C, $op->[19]);
    }
  }
}
sub traverse_btree
{
  my($fh, $n, $num) = @_;
  return 'nullptr' unless ( defined($n) );
  if ( $n->[0] eq 'L' ) {
    return 'nullptr' unless defined($n->[1]); # empty leaf node - wtf?
    my $name = 'leaf_' . $$num++;
    printf($fh "static const NV_bt_node %s = { 0x10000, {\n", $name);
    # dump insts in $n->[1]
    dump_matched($fh, $n->[1]);
    printf($fh "} };\n");
    return '&' . $name;
  }
  my $left = traverse_btree( $fh, $n->[2], $num);
  my $right = traverse_btree( $fh, $n->[3], $num);
  my $name = 'node_' . $$num++;
  printf($fh "static const NV_non_leaf %s = { %d, {\n", $name, $g_size - 1 - $n->[1]);
  # dump insts in $n->[4]
  dump_matched($fh, $n->[4]);
  # must invert bit position too
  printf($fh "}, %s, %s };\n", $left, $right);
  return '&' . $name;
}
sub c_mask_name
{
  my $m = shift;
  $m =~ s/\./_/g;
  sprintf("%s_mask_%s", $opt_C, $m);
}
sub gen_masks
{
  my $fh = shift;
  printf($fh "// ---- masks\n");
  foreach my $m ( keys %g_mnames ) {
    my $op = $g_mnames{$m}->[3];
    printf($fh "NV_MASK(%s, %d) = { ", c_mask_name($m), (scalar @$op) / 2);
    for ( my $i = 0; $i < scalar @$op; $i += 2 ) {
      # 1st - offset, 2nd - len
      # we need to invert mask, so new offset will be g_size - (offset + len)
      my $off = $op->[$i];
      my $len = $op->[$i+1];
      printf($fh "{ %d, %d }", $g_size - ($off + $len), $len);
      printf($fh ",");
    }
    printf($fh "};\n");
  }
}
# return (mask_name, mask_size)
sub c_get_mask
{
  my $mname = shift;
  my $mask = $g_mnames{$mname};
  return ( c_mask_name($mname), scalar( @{$mask->[3]} ) / 2 );
}

sub c_enum_name
{
  my $e = shift;
  $e =~ s/\./_/g;
  sprintf("%s_enum_%s", $opt_C, $e);
}
sub gen_enums
{
  my $fh = shift;
  printf($fh "// ---- enums\n");
  foreach my $ename ( keys %g_used_enums ) {
    printf($fh "NV_ENUM(%s) = {\n", c_enum_name($ename));
    my %cenum;
    # make copy
    my $oe = $g_enums{$ename};
    while( my($k, $v) = each %$oe ) {
      $cenum{$v} = $k;
    }
    foreach my $i ( sort keys %cenum ) {
      printf($fh " { %d, \"%s\" },\n", $i, $cenum{$i} );
    }
    printf($fh "};\n");
  }
  printf($fh "\n");
}

sub c_tab_name
{
  sprintf("%s_tab_%s", $opt_C, shift);
}
sub gen_tabs
{
  my $fh = shift;
  printf($fh "// ---- tables\n");
  foreach my $ename ( keys %g_used_tabs ) {
    my $t = $g_tabs{$ename};
    my @cont;
    foreach my $tkey ( sort keys %$t ) {
      my $pfx = sprintf("s_%d_%s", $tkey, $ename);
      push @cont, [ $tkey, $pfx];
      printf($fh "static const unsigned short %s[] = {", $pfx);
      my $row = $t->{$tkey};
      if ( 'ARRAY' eq ref $row ) {
        printf($fh "%d", scalar @$row);
        foreach my $r ( @$row ) {
          printf($fh ", %d", $r);
        }
        printf($fh " };\n");
      } else {
        printf($fh "1, %d };\n", $row);
      }
    }
    printf($fh "NV_TAB(%s) = {\n", c_tab_name($ename));
    printf($fh " {%d, %s},\n", $_->[0], $_->[1]) for @cont;
    printf($fh "};\n");
  }
  printf($fh "\n");
}

# generate key for hash of used enum attrs
sub c_ae
{
  my $ae = shift;
  my $res = $ae->[0];
  $res .= '_t' if ( $ae->[1] );
  $res .= '_' . $ae->[2] if ( defined $ae->[2] ); # default value
  $res .= '_p' if ( $ae->[4] );
  $res;
}
sub c_ae_name
{
  my $ae = shift;
  $ae =~ s/\./_/g;
  sprintf("%s_%s", $opt_C, $ae);
}
sub gen_ae
{
  my($fh, $ae, $name) = @_;
  printf($fh "static const nv_eattr %s = { ", c_ae_name($name));
  printf($fh "%s,", $ae->[1] ? 'true' : 'false');
  printf($fh "%s,", $ae->[4] ? 'true' : 'false');
  printf($fh "%s,", defined $ae->[2] ? 'true' : 'false');
  printf($fh "%d,", $ae->[2] || 0);
  printf($fh "\"%s\",", $ae->[0]);
  printf($fh "&%s };\n", c_enum_name($ae->[0]));
}
sub gen_filter
{
  my($op, $fh) = @_;
  return 'nullptr' unless defined($op->[12]);
  my $name = sprintf("%s_%d_filter", $opt_C, $op->[19]);
  printf($fh "\nstatic int %s(std::function<uint64_t(const std::pair<short, short> *, size_t)>  &fn) {\n", $name);
  # c++ impl of filter_ins
  my $flist = $op->[12];
  foreach my $f ( @$flist ) {
    my($mask, $size) = c_get_mask($f->[0]->[0]);
    printf($fh " { // %s\n", $f->[1]);
    printf($fh " auto v = fn(%s, %d);\n", $mask, $size);
    if ( 'v' eq $f->[1] ) {
      printf($fh " if ( v != %d ) return 0;", $f->[2]);
    } elsif ( 'e' eq $f->[1] ) {
      my $ename = c_enum_name($f->[3]);
      printf($fh " auto ci = %s.find((int)v); if ( ci == %s.end() ) return 0;", $ename, $ename);
    } else {
      my $tr = $f->[2];
      my $tab_name = c_tab_name($f->[3]);
      printf($fh " auto ti = %s.find((int)v); if ( ti == %s.end() ) return 0;", $tab_name, $tab_name);
      if ( 'T' eq $f->[1] ) {
        # extract row from table and check in enums
        for ( my $i = 4; $i < scalar @$f; $i++ ) {
          next unless defined($f->[$i]);
          my $e = c_enum_name($f->[$i]);
          my $i_name = 'i' . ($i - 3);
          printf($fh "\n auto %s = %s.find((int)ti->second[%d]); if ( %s == %s.end()) return 0;", $i_name, $e, $i - 3, $i_name, $e);
        }
      }
    }
    printf($fh " }\n");
  }
  printf($fh " return 1;\n}\n");
  $name;
}
sub inl_extract
{
  my($fh, $m, $n) = @_;
  my($mask, $size) = c_get_mask($m);
  printf($fh "auto v%d = fn(%s, %d);\n", $n, $mask, $size);
}
# as inl_extract without tmp auto vXX
sub inl_extract2
{
  my($fh, $m) = @_;
  my($mask, $size) = c_get_mask($m);
  printf($fh "fn(%s, %d);\n", $mask, $size);
}
sub bank_extract
{
  my($fh, $m, $n) = @_;
  my($mask, $size) = c_get_mask($m);
  printf($fh "auto c%d = fn(%s, %d);\n", $n, $mask, $size);
  return mask_len($g_mnames{$m});
}
# args: op, field name, format for field, ref to float conv ref, array wirh convertFloatType args
sub parse_conv_float
{
  my($op, $fname, $format, $fc, $args) = @_;
  my @spl = split(/\s*\|\|\s*/, $args->[0]);
  my $prev;
  my %vhash;
  foreach my $s ( @spl ) {
    if ( $s !~ /([\w\.]+)\s*==\s*`([\w\.]+)@([\w\.]+)/ ) {
      printf("bad conv_float %s for %s, line %d\n", $s, $op->[1], $op->[4]);
      next;
    }
    # check if we have var in $1
    my $vname = $1;
    if ( !exists $op->[17]->{$vname} ) {
      printf("bad conv_float varname %s for %s, line %d\n", $vname, $op->[1], $op->[4]);
      next;
    }
    if ( defined $prev ) {
      return 0 if ( $prev ne $vname );
    } else {
      $prev = $vname;
    }
    # check value $3 in enum $3
    if ( !exists $g_enums{$2} ) {
      printf("bad conv_float enum %s for %s, line %d\n", $2, $op->[1], $op->[4]);
      next;
    }
    my $e = $g_enums{$2};
    if ( !exists $e->{$3} ) {
      printf("no value %s in enum %s for %s, line %d\n", $3, $2, $op->[1], $op->[4]);
      next;
    }
    next if exists $vhash{$e->{$3}};
    $vhash{$e->{$3}} = 1;
  }
  # ok, check size
  my @vkeys = keys %vhash;
  return 0 unless ( scalar @vkeys );
  if ( scalar(@vkeys) > 2 ) {
      printf("bad conv_float values count %d for %s, line %d\n", scalar @vkeys, $op->[1], $op->[4]);
      return 0;
  }
  # fill array for $fc
  my @res = ( $prev, $format, $vkeys[0] );
  if ( $format eq 'F16Imm') {
    push @res, 0;
  } else {
    push @res, scalar(@vkeys) > 1 ? $vkeys[1]: -1;
  }
  $fc->{$fname} = \@res;
  1;
}
# example from sm3:
# /ICmpAll:icomp but there is no encoder for icomp, instead we have
# IComp = ICmpAll - so we must check if second arg is enum
sub gen_extr
{
  my($op, $fh, $vw, $fc) = @_;
  my $enc = $op->[5];
  return 'nullptr' unless defined($enc);
  my $name = sprintf("%s_%d_extr", $opt_C, $op->[19]);
  printf($fh "\nstatic void %s(std::function<uint64_t(const std::pair<short, short> *, size_t)>  &fn, NV_extracted &res) {\n", $name);
  # c++ impl of dump_values
  my $index = 0;
  foreach my $m ( @$enc ) {
    if ( $m =~ /^([\w\.]+)\s*=\*?\s*IDENTICAL\(([^\)]+)\)/ ) {
      my $ids = $2;
      printf($fh "// %s identical %s\n", $1, $2);
      inl_extract($fh, $1, $index);
      # fill
      foreach my $a ( split /\s*,\s*/, $ids ) {
        printf($fh "res[\"%s\"] = v%d;\n", $a, $index);
      }
      $index++; next;
    } elsif ( $m =~ /^([\w\.]+)\s*=\*?\s*(\S+)\s*\(\s*([^\)]+)\s*\)/ ) {
      if ( exists($g_tabs{$2}) ) {
        printf($fh "// %s table %s %s\n", $1, $2, $3);
        inl_extract($fh, $1, $index);
        my $tab_name = c_tab_name($2);
        printf($fh "auto i%d = %s.find(v%d); if ( i%d != %s.end() ) {\n", $index, $tab_name, $index, $index, $tab_name);
        printf($fh " auto &ctab = i%d->second;\n", $index);
        my @fa = split /\s*,\s*/, $3;
        for ( my $i = 0; $i < @fa; $i++ ) {
          $fa[$i] =~ s/\s+$//;
          next if ( $fa[$i] =~ /^\d+$/ ); # skip constant
          printf($fh " res[\"%s\"] = ctab[%d];\n", $fa[$i], $i+1);
        }
        printf($fh "}\n");
        $index++; next;
      } else {
        printf($fh "// %s bad table %s %s\n", $1, $2, $3);
        printf($fh "res[\"%s\"] = ", $2);
        inl_extract2($fh, $1);
        $index++; next;
      }
    } elsif ( $m =~ /^([\w\.]+)\s*=\*?\s*([\w\.\@]+)(?:\s*SCALE\s+(\d+))?$/ ) {
      if ( defined $3 ) {
        printf($fh "// %s to %s scale %s\n", $1, $2, $3);
      } else {
        printf($fh "// %s to %s\n", $1, $2);
      }
      # mask $1 value $2 - ignore req_bit_set bcs it's always BITSET
      if ( defined $op->[18] && $2 ne 'req_bit_set' && exists $op->[18]->{$2} ) {
        $vw->{$2} = mask_len( $g_mnames{$1} ) + scale_len($3) if exists($g_mnames{$1});
      }
      printf($fh "res[\"%s\"] = ", $2);
      printf($fh "%s * ", $3) if ( defined $3 );
      inl_extract2($fh, $1);
      $index++; next;
    } elsif ( $m=~ /^([\w\.]+)\s*=\*?\s*([\w\.\@]+)\s+convertFloatType\s*\((.*)\s*\)/ ) {
      printf($fh "// convertFloatType %s\n", $3);
      inl_extract($fh, $1, $index);
      my $field = $2;
      printf($fh "res[\"%s\"] = v%d;\n", $2, $index);
      if ( defined $fc ) {
        # I don't know exact format of convertFloatType function - seems that it must have 3 fields:
        # 1 - expression
        # 2 - format name
        # 3 - type F16Imm or F32Imm
        # so first thing to do - split on commas and check size and last arg
        my @fc_args = split(/\s*,\s*/, $3);
        my $fc_len = scalar @fc_args;
        if ( $fc_len < 3 ) {
          printf("bad args for convertFloatType(%s) for %s, line %d\n", $3, $op->[1], $op->[4]);
        } else {
          splice @fc_args, 0, $fc_len - 3 if ( $fc_len > 3 );
          if ( $fc_args[2] eq 'F16Imm' || $fc_args[2] eq 'F32Imm' ) {
            if ( $fc_args[0] =~ /1\s*==\s*1/ ) {
              # it's much simpler just to replace vas field
              my $vas = $op->[18];
              if ( exists $vas->{$field} ) { $vas->{$field}->[0] = $fc_args[2]; }
              else { printf("no vas for %s, instr %s line %d\n", $field, $op->[1], $op->[4]); }
            } else {
              parse_conv_float($op, $field, $fc_args[2], $fc, \@fc_args);
            }
          }
        }
      }
      $index++; next;
    }
  }
  # const bank
  if ( defined $op->[10] ) {
    my $cb = $op->[10];
    printf($fh "// const bank %s\n", $cb->[0]);
    my $cb_len = scalar @$cb;
    my @fcb;
    if ( $cb->[0] =~ /\(\s*(.*),\s*(.*)\)/ ) {
      push @fcb, $1; push @fcb, $2;
    } else {
      printf("bad const bank %s\n", $cb->[0]);
    }
    # scale for Address2
    my $pfx = substr($op->[10]->[0], 0, 1);
    # special case for sm3:
    # BcbankHi,BcbankLo,Bcaddr =  ConstBankAddress2(constBank,immConstOffset)
    # here constBank = BcbankLo | (BcbankHi << size(BcbankLo)
    if ( 4 == $cb_len ) {
      bank_extract($fh, $cb->[1], 0);
      my $lo_size = bank_extract($fh, $cb->[2], 1);
      bank_extract($fh, $cb->[3], 2);
      printf($fh "res[\"%s\"] = c1 | (c0 << %d);\n", $fcb[0], $lo_size);
      if ( $pfx eq '2' ) {
        printf($fh "res[\"%s\"] = 4 * c2;\n", $fcb[1]);
      } else {
        printf($fh "res[\"%s\"] = c2;\n", $fcb[1]);
      }
    } else {
      for ( my $i = 1; $i < $cb_len; $i++ ) {
        inl_extract($fh, $cb->[$i], $index);
        if ( $i == 2 && $pfx eq '2' ) {
          printf($fh "res[\"%s\"] = 4 * v%d;\n", $fcb[$i-1], $index);
        } else {
          printf($fh "res[\"%s\"] = v%d;\n", $fcb[$i-1], $index);
        }
        $index++;
      }
    }
  }
  printf($fh "\n}\n");
  $name;
}
sub gen_base
{
  my $f = shift;
  my $res = '';
  # prefix
  if ( $f->[1] ) { $res .= "'" . $f->[1] . "', "; }
  else { $res .= '0, '}
  # suffix
  if ( $f->[2] ) { $res .= "'" . $f->[2] . "', "; }
  else { $res .= '0, '}
  # mod
  if ( $f->[3] ) { $res .= "'" . $f->[3] . "'"; }
  else { $res .= '0'}
  $res;
}
# M1 & M2 use $f->[3] as format string
sub gen_base2
{
  my $f = shift;
  my $res = '';
  # prefix
  if ( $f->[1] ) { $res .= "'" . $f->[1] . "', "; }
  else { $res .= '0, '}
  # suffix
  if ( $f->[2] ) { $res .= "'" . $f->[2] . "', "; }
  else { $res .= '0, '}
  # mod
  $res .= '0';
  $res;
}
sub quoted_s
{
  my $s = shift;
  return 'nullptr' unless defined($s);
  '"' . $s . '"';
}
sub rend_name_ea
{
  my($f, $idx) = @_;
  my $ea = $f->[$idx];
  return quoted_s($ea->[3]);
}
sub rend_value
{
  my $r = shift;
  'R_value, 0, ' . quoted_s($r);
}
sub rend_value_plus
{
  my($r, $p) = @_;
  my $res = 'R_value, ';
  if ($p) { $res .= "'+', "; }
  else { $res .= '0, ';}
  $res . quoted_s($r);
}
sub rend_enum
{
  my $r = shift;
  'R_enum, 0, ' . quoted_s($r->[3]);
}
# args: $fh, vX index, $f list, start index
sub rend_list
{
  my($fh, $vidx, $f, $idx) = @_;
  printf($fh "{ // rend for %d idx %d\n", $vidx, $idx);
  my $li = 0;
  my $was = 0;
  for my $i ($idx .. scalar @$f) {
    next unless defined $f->[$i];
    if ( 'ARRAY' eq ref $f->[$i] ) { # make enum
      printf($fh " ve_base l%d{ %s };\n", $li, rend_enum($f->[$i]));
    } elsif ( '+' eq $f->[$i] ) { $was = 1; next; }
    else { # make value
      printf($fh " ve_base l%d{ %s };\n", $li, rend_value_plus($f->[$i], $was));
      $was = 0;
    }
    printf($fh " v%d->right.push_back( std::move(l%d));\n", $vidx, $li++);
  }
  printf($fh "}\n");
}
# 7 - reg, 6 - imm
sub rend_C_list
{
  my($fh, $vidx, $f) = @_;
  printf($fh "{ // rend_C for %d len %d\n", $vidx, scalar @$f);
  if ( !defined $f->[7] ) {
   printf($fh " ve_base l%d{ %s };\n", 0, rend_value_plus($f->[6], 0));
   printf($fh " v%d->right.push_back( std::move(l%d));\n", $vidx, 0);
  } else {
   # first enum
   printf($fh " ve_base l%d{ %s };\n", 0, rend_enum($f->[7]));
   printf($fh " v%d->right.push_back( std::move(l%d));\n", $vidx, 0);
   # then value
   printf($fh " ve_base l%d{ %s };\n", 1, rend_value_plus($f->[6], 0));
   printf($fh " v%d->right.push_back( std::move(l%d));\n", $vidx, 1);
  }
  printf($fh "}\n");
}
sub gen_render
{
  my($op, $fh) = @_;
  printf($fh "static void fill_rend_%d(NV_rlist *res) {\n", $op->[19]);
  my $idx = 0;
  foreach my $f ( @{ $op->[15] } ) {
    # total Wittenoom
    if ( $f->[0] eq '$') {
      printf($fh " auto v%d = new render_base(R_opcode, %s);\n", $idx, gen_base($f));
    } elsif ( $f->[0] eq 'P' ) { # +
      printf($fh " auto v%d = new render_named(R_predicate, %s, %s);\n", $idx, gen_base($f), rend_name_ea($f, 4));
    } elsif ( $f->[0] eq 'E' ) { # +
      printf($fh " auto v%d = new render_named(R_enum, %s, %s);\n", $idx, gen_base($f), rend_name_ea($f, 4));
    } elsif ( $f->[0] eq 'V' ) { # +
      printf($fh " auto v%d = new render_named(R_value, %s, %s);\n", $idx, gen_base($f), quoted_s($f->[4]));
    } elsif ( $f->[0] eq '[' ) {
      printf($fh " auto v%d = new render_mem(R_mem, %s, nullptr);\n", $idx, gen_base($f));
      rend_list($fh, $idx, $f, 4);
    } elsif ( $f->[0] eq 'A' ) {
      printf($fh " auto v%d = new render_mem(R_mem, %s, \"attr\");\n", $idx, gen_base($f));
      rend_list($fh, $idx, $f, 4);
    } elsif ( $f->[0] eq 'C') {
      printf($fh " auto v%d = new render_C(R_C, %s, %s, { %s });\n", $idx, gen_base($f), quoted_s($f->[4]), rend_value($f->[5]));
      rend_C_list($fh, $idx, $f);
    } elsif ( $f->[0] eq 'X' ) {
      printf($fh " auto v%d = new render_C(R_CX, %s, %s, { %s });\n", $idx, gen_base($f), quoted_s($f->[4]), rend_enum($f->[5]));
      rend_C_list($fh, $idx, $f);
    } elsif ( $f->[0] eq 'D' ) {
      printf($fh " auto v%d = new render_desc(R_desc, %s, { %s });\n", $idx, gen_base($f), rend_enum($f->[4]));
      rend_list($fh, $idx, $f, 5);
    } elsif ( $f->[0] eq 'T' ) { # +
      printf($fh " auto v%d = new render_TTU(R_TTU, %s, { %s } );\n", $idx, gen_base($f), rend_value($f->[4]));
    } elsif ( $f->[0] eq 'M1' ) { # +
      printf($fh " auto v%d = new render_M1(R_M1, %s, %s, { %s } );\n", $idx, gen_base2($f), quoted_s($f->[3]), rend_enum($f->[4]));
    } elsif ( $f->[0] eq 'M2' ) {
      printf($fh " auto v%d = new render_mem(R_mem, %s, %s);\n", $idx, gen_base2($f), quoted_s($f->[3]));
      rend_list($fh, $idx, $f, 4);
    }
    printf($fh " NVREND_PUSH( v%d )\n", $idx++);
  }
  printf($fh "}\n\n");
}
# predicates logic
# cached functions
my %g_predf;
my $g_pred_n = 0;
# convert enum @ value to int const
sub pred_conv
{
  my($e, $v) = @_;
  if ( !exists $g_enums{$e} ) {
    carp("no enum $e");
    return;
  }
  my $en = $g_enums{$e};
  if ( !exists $en->{$v} ) {
    carp("no $v in enum $e");
    return;
  }
  $en->{$v};
}
# find enum by enc name
sub pred_find_enum
{
  my($op, $e) = @_;
  my $mae = $op->[17];
  foreach my $ae ( values %$mae ) {
    return $ae->[0] if ( $ae->[3] eq $e );
  }
  foreach my $ae ( values %$mae ) {
    return $ae->[3] if ( $ae->[0] eq $e );
  }
  carp("cannot find enum $e");
}
# args: file handle, instruction, function body
sub gen_pred_func
{
  my($fh, $op, $str) = @_;
  return $g_predf{$str} if exists( $g_predf{$str} );
  # make definition
  my $res = 'smp_' . $g_pred_n++;
  printf($fh "static int %s(const NV_extracted &kv) {\n", $res);
  # check if str is just string
  if ( $str =~ /^(\d+)$/ ) {
    $g_predf{$str} = $res;
    printf($fh " return %s;\n}\n", $str);
    return $res;
  }
  # collect args
  my %args;
  while( $str =~ /\b([\w\.]+)\s*[><=\?\!]/g ) {
    $args{$1}++;
  }
  # boring extract logic
  my $copy = $str;
  my $a_idx = 0;
  foreach my $k ( keys %args ) {
    my $kdot = $k;
    $a_idx++;
    if ( $k =~ /\./ ) {
      $k =~ s/\./_/g;
      $copy =~ s/$kdot/$k/g;
    }
    my $iter_name = $k . '_iter';
    if ( defined($op->[18]) && exists $op->[18]->{$k} ) {
     printf($fh " auto %s = kv.find(\"%s\"); if ( %s == kv.end() ) return -%d;\n", $iter_name, $kdot, $iter_name, $a_idx);
    } else {
      my $e_name = pred_find_enum($op, $kdot);
      printf($fh " auto %s = kv.find(\"%s\"); if ( %s == kv.end() ) {\n", $iter_name, $kdot, $iter_name);
      # find value with name
      printf($fh " %s = kv.find(\"%s\"); if ( %s == kv.end() ) return -%d; }\n", $iter_name, $e_name, $iter_name, $a_idx);
    }
    # int value with name $k
    printf($fh " int %s = (int)%s->second;\n", $k, $iter_name);
  }
  $copy =~ s/`(\w+)\@\"?([\w\.]+)\"?/pred_conv($1, $2)/eg;
  printf($fh " return %s;\n", $copy);
  printf($fh "}\n");
  # store processed function in cache
  $g_predf{$str} = $res;
  $res;
}

sub gen_preds
{
  my($fh, $op) = @_;
  return unless defined($op->[21]);
  my $props = $op->[21];
  my $res = 'preds_' . $op->[19];
  my %body;
  while ( my($name, $what) = each %$props ) {
    $body{$name} = gen_pred_func($fh, $op, $what);
  }
  return unless keys(%body); # something went wrong
  printf($fh "NV_PRED(%s) = {\n", $res);
  while ( my($name, $what) = each %body ) {
    printf($fh " // %s -> %s\n", $name, $props->{$name});
    printf($fh " { \"%s\", %s },\n", $name, $what);
  }
  printf($fh "};\n");
  $res;
}
sub gen_instr
{
  my $fh = shift;
  my %cached_ae;
  while( my($m, $list) = each %g_masks) {
    printf($fh "//%s\n", $m);
    foreach my $op ( @$list ) {
      # collect enums
      foreach my $ae ( values %{ $op->[17] } ) {
        my $ename = c_ae($ae);
        next if exists $cached_ae{$ename};
        gen_ae($fh, $ae, $ename);
        $cached_ae{$ename} = 1;
      }
      # predicated
      my $pred_name = gen_preds($fh, $op);
      # filter
      my $op_filter = gen_filter($op, $fh);
      # extractor
      my %vw;
      my %fc;
      my $op_extr = gen_extr($op, $fh, \%vw, \%fc);
      # rows & cols
      my $rname = check_tab_rows($fh, $op);
      my $cname = check_tab_cols($fh, $op);
      # render
      gen_render($op, $fh);
      # vwidth
      my $width_name;
      if ( keys %vw ) {
        $width_name = 'width_' . $op->[19];
        printf($fh "static const NV_width %s = { // %d widths\n", $width_name, scalar keys %vw);
        while( my($v, $w) = each %vw ) {
          printf($fh " {\"%s\", %d },\n", $v, $w);
        }
        printf($fh "};\n");
      }
      # vf_conv
      my $conv_name;
      my @fconv = keys %fc;
      if ( scalar @fconv ) {
        $conv_name = 'conv_' . $op->[19];
        printf($fh "static const NV_conv %s = { // %d conversions\n", $conv_name, scalar @fconv);
        foreach my $c1 ( @fconv ) {
          printf($fh " { \"%s\", {", $c1);
          my $ca = $fc{$c1};
          printf($fh "\"%s\", NV_%s, %d, %d} },\n", $ca->[0], $ca->[1], $ca->[2], $ca->[3]);
        }
        printf($fh "};\n");
      }
      # dump instruction
      printf($fh "static const struct nv_instr %s_%d = {\n", $opt_C, $op->[19]);
      # name mask line n alt meaning_bits
      printf($fh "\"%s\",\n \"%s\",\n \"%s\", %d, %d, %d, %d,\n", $m, $op->[0],$op->[1], $op->[4], $op->[19], $op->[7], $op->[14]);
      # brt properties
      if ( defined $op->[20] ) {
        my $brt = $op->[20];
        printf($fh "%s,", $brt->[0] || '0');
        printf($fh "%s,", $brt->[1] || '0'); # scbd
        printf($fh "%s,", $brt->[2] || '0'); # scbd_type
        for ( my $bi = 3; $bi < 6; $bi++ ) {
          if ( defined $brt->[$bi]) { printf($fh "\"%s\",", $brt->[$bi]); }
          else { printf($fh "nullptr,"); }
        }
        printf($fh "\n");
      } else {
        printf($fh "0, 0, 0, nullptr, nullptr, nullptr,\n");
      }
      # predicates
      if ( defined $pred_name ) { printf($fh "&%s,\n", $pred_name); }
      else { printf($fh " nullptr,\n"); }
      # vf_conv
      if ( defined $conv_name ) { printf($fh "&%s,\n", $conv_name); }
      else { printf($fh " nullptr,\n"); }
      # vwidth
      if ( defined $width_name ) { printf($fh "&%s,\n", $width_name); }
      else { printf($fh " nullptr,\n"); }
      # vas
      if ( defined $op->[18] ) {
        printf($fh " {");
        my $vlist = $op->[18];
        while( my($v, $vf) = each %$vlist ) {
          printf($fh " {\"%s\", { %s, %s}},", $v, 'NV_' . $vf->[0], defined($vf->[1]) ? 'true' : 'false');
        }
        printf($fh "}, // values formats\n");
      } else {
        printf($fh " {}, // no values\n");
      }
      # dump enum attrs
      printf($fh " {");
      foreach my $ae ( values %{ $op->[17] } ) {
        my $ename = c_ae($ae);
        printf($fh "{ \"%s\", &%s },", $ae->[3], c_ae_name($ename));
      }
      printf($fh "}, %s, %s,\n", $op_filter, $op_extr);
      # rows
      if ( defined $rname ) { printf($fh "&%s,\n", $rname); }
      else { printf($fh " nullptr,\n"); }
      # cols
      if ( defined $cname ) { printf($fh "&%s\n", $cname); }
      else { printf($fh " nullptr\n"); }
      printf($fh " };\n");
    }
  }
}

sub gen_C
{
  # enum all instructions
  my $n = 0;
  while( my($kmask, $op) = each(%g_masks) ) {
   foreach my $inst ( @$op ) { $inst->[19] = $n++; }
  }
  # open file
  my $fname = $opt_C . '.cc';
  my $fh;
  open($fh, '>', $fname) or die("Cannot create $fname, error $!");
  # make header
  printf($fh "// Dont edit this file - it was generated %s with option %s", scalar(localtime), $opt_C);
  printf($fh " with predicates") if ( defined $opt_p );
  printf($fh " with groups") if ( defined $opt_g );
  # include
  printf($fh "\n#include \"include/nv_types.h\"\n");
  # virtual queues
  gen_vq($fh);
  # tabs
  gen_c_gtabs($fh) if defined($opt_g);
  # dump masks
  gen_masks($fh);
  # dump used enums
  gen_enums($fh);
  # dump used tabs
  gen_tabs($fh);
  # dump instructions
  gen_instr($fh);
  # ins_render
  printf($fh "NV_one_render ins_render[%d] = {\n", $n);
  printf($fh " { fill_rend_%d },\n", $_) for ( 0 .. $n - 1);
  printf($fh "};\n");
  # dump binary tree in g_dec_tree
  printf($fh "\n// binary tree\n");
  my $num = 0;
  my $root = traverse_btree($fh, $g_dec_tree, \$num);
  # finally gen get_sm
  printf($fh "\nINV_disasm *get_sm() {\n");
  printf($fh " return new NV_disasm<nv%d>(%s, %d, %d); }\n", $g_size, $root, $g_rz, $n);
  close $fh;
}

# group processing logic
my %g_nops; # key - name, value - ref to instruction, can be shared from several names like Op, Op_pipe1, Op_pipe2 etc
my %g_groups; # key - name, value - hashmap with (name, ref to instruction)

# tables are arrays and can contain
# [0] type - true/anti/output
# [1] connector name
# [2] line number - for debugging
# [3] list of columns with size N
# [4] list of rows - first is group_name and then there can be 1 or N values
my @g_gtabs;

# connector sets
my %g_csets; # key - name, value - list of groups

sub process_cset
{
}

# connector conditions
# in old versions like sm3 is pure hell like ANNOTATED etc, so it works only since 5
# we need two hashes - 1st for whole set and 2nd for really used
my %g_ccond; # key - name, value - body
my %g_used_ccond; # key - name, value - body in c++

sub c_ccond_func
{
  my $name = shift;
  return 's_ccond_' . $name;
}

sub try_convert_ccond
{
  my($name, $body) = @_;
  return 1 if exists($g_used_ccond{$name} );
  $body =~ s/_OR_/||/g;
  $body =~ s/_AND_/&&/g;
  # 1) lets collect predicates
  my %preds;
  while( $body =~ /(?:MD_)PRED\(([^\)]+)\)/pg ) {
    $preds{$1}++;
  }
  foreach my $s ( keys %preds ) {
    $body =~ s/(?:MD_)PRED\($s\)/$s/g;
  }
  # 2) call of other connection conditions
  my %calls;
  while( $body =~ /([A-Z0-9]+)/gp ) {
    if ( exists $g_ccond{$1} ) {
      $calls{$1}++;
    } else {
      printf("unknown CCond %s in %s\n", $1, $name);
      return 0;
    }
  }
  # 3) finally collect fields
  my %fields;
  while( $body =~ /\b(\w+)/gp ) {
    next if ( exists $preds{$1} );
    next if ( exists $calls{$1} );
    $fields{$1}++;
  }
  # form result body
  my $res = '';
  $res = ' if ( !i->predicated ) return 0;' . "\n" if ( keys %preds );
  my $idx = 0;
  foreach my $k ( keys %preds ) {
    $res .= sprintf(" auto p%d = i->predicated->find(\"%s\"); if ( p%d == i->predicated->end() ) return 0;\n auto %s = p%d.second(kv);\n",
     $idx, $k, $idx, $k);
    $idx++;
  }
  foreach my $k ( keys %calls ) {
    $res .= sprintf("auto %s = %s(i, kv);\n", $k, c_ccond_func($k));
  }
  foreach my $f ( keys %fields ) {
    $res .= sprintf(" auto fi%s = kv.find(\"%s\"); if ( fi%s == kv.end() ) return 0; auto %s = fi%s.second;\n",
     $f, $f, $f);
  }
  $res .= 'return ' . $body;
  # store body
  $g_used_ccond{$name} = $res;
  return 1;
}


# to link instructions with tabs we need yet two maps - for columns & rows
# key - instr ref, value - [ [ tab, index ], ... ]
my %g_gtcols;
my %g_gtrows;

sub check_tab_cols
{
  my($fh, $i) = @_;
  return unless exists $g_gtcols{$i};
  my $res = 'tab_col_' . $i->[19];
  printf($fh "static const NV_tabrefs %s = {\n", $res);
  my $l = $g_gtcols{$i};
  foreach my $i ( @$l ) {
    printf($fh " { &%s, %d },\n", c_gtab_name($i->[0]), $i->[1]);
  }
  printf($fh "};\n");
  $res;
}

sub check_tab_rows
{
  my($fh, $i) = @_;
  return unless exists $g_gtrows{$i};
  my $res = 'tab_row_' . $i->[19];
  printf($fh "static const NV_tabrefs %s = {\n", $res);
  my $l = $g_gtrows{$i};
  foreach my $i ( @$l ) {
    printf($fh " { &%s, %d },\n", c_gtab_name($i->[0]), $i->[1]);
  }
  printf($fh "};\n");
  $res;
}

# args: name of group, ref to table, index
sub mark_tab_col
{
  my($gname, $t, $cidx) = @_;
  my $res = 0;
  my $g = $g_groups{$gname};
  foreach my $k ( keys %$g ) {
    if ( !exists $g_gtcols{ $k } ) {
      $g_gtcols{ $k } = [ [ $t, $cidx ] ];
    } else {
      my $c = $g_gtcols{ $k };
      my $need = 1;
      foreach my $curr ( @$c ) {
        if ( $curr->[0]->[2] == $t->[2] && $curr->[1] == $cidx ) {
          $need = 0;
          last;
        }
      }
      push @$c, [ $t, $cidx ] if ( $need );
    }
    $res++;
  }
  $res;
}

sub mark_tab_row
{
  my($gname, $t, $ridx) = @_;
  my $res = 0;
  my $g = $g_groups{$gname};
  foreach my $k ( keys %$g ) {
    if ( !exists $g_gtrows{ $k } ) {
      $g_gtrows{ $k } = [ [ $t, $ridx ] ];
    } else {
      my $c = $g_gtrows{ $k };
      my $need = 1;
      foreach my $curr ( @$c ) {
        if ( $curr->[0]->[2] == $t->[2] && $curr->[1] == $ridx ) {
          $need = 0;
          last;
        }
      }
      push @$c, [ $t, $ridx ] if ( $need );
    }
    $res++;
  }
  $res;
}

sub c_gtab_name
{
  my $t = shift;
  return 's_tab_' . $t->[2];
}

# gen one table in NV_tab form
sub gen_c_gtab
{
  my($fh, $t) = @_;
  return unless defined($t);
  my $name = c_gtab_name($t);
  printf($fh "static const NV_tab %s = {\n", $name);
  # name
  printf($fh "\"%s\",\n", $t->[0]);
  # connector
  printf($fh "\"%s\",\n", $t->[1]);
  # col names
  my $cols = $t->[3];
  printf($fh "// columns\n{");
  my $cidx = 0;
  foreach my $c ( @$cols ) {
    mark_tab_col($c, $t, $cidx++);
    printf($fh "\"%s\",", $c);
  }
  printf($fh "},\n");
  # row names
  my $size = scalar(@$t);
  printf($fh "// rows\n{");
  for my $i ( 4 .. $size - 1 ) {
    my $r = $t->[$i];
    mark_tab_row($r->[0], $t, $i-4);
    printf($fh "\"%s\",", $r->[0]);
  }
  printf($fh "},\n{\n");
  # body
  for my $i ( 4 .. $size - 1 ) {
    my $r = $t->[$i];
    printf($fh " {");
    my $rs = scalar @$r;
    foreach my $j ( 1 .. $rs - 1 ) {
      printf($fh ",") if ( $j > 1 );
      if ( defined($r->[$j]) ) {
        printf($fh "%d", $r->[$j]);
      } else {
        printf($fh '-1');
      }
    }
    printf($fh " }, // row %d\n", $i - 4);
  }
  # end
  printf($fh "} };\n");
}

my $m_ccond_fwd = '(const struct nv_instr *i, const NV_extracted &kv)';

sub gen_c_gtabs
{
  my $fh = shift;
  if ( keys %g_used_ccond ) {
    # first pass - form forward declarations bcs they can call each others
    foreach my $cc ( keys %g_used_ccond ) {
      printf($fh "statuc int %s%s;\n", c_ccond_func( $cc ), $m_ccond_fwd);
    }
    # second pass - put defunitions
    foreach my $cc ( keys %g_used_ccond ) {
     printf($fh "statuc int %s%s {\n", c_ccond_func( $cc ), $m_ccond_fwd);
     printf($fh "%s\n}\n", $g_used_ccond{ $cc });
    }
  }
  gen_c_gtab($fh, $_) for @g_gtabs;
}

sub dump_gtab
{
  my $t = shift;
  return unless defined($t);
  my $cols = $t->[3];
  printf("Tab_%s %s line %d %d columns\n", $t->[0], $t->[1], $t->[2], scalar @$cols);
  printf(" col %s\n", $_) for @$cols;
  my $size = scalar(@$t);
  return if ( $size <= 4 );
  printf("%d rows\n", $size - 4);
  for my $i ( 4 .. $size - 1 ) {
    my $r = $t->[$i];
    my $rs = scalar @$r;
    printf(" %s: ", $r->[0]);
    # perl syntax is so mad when you need to carve subarray on array ref
    foreach my $j ( 1 .. $rs - 1 ) {
      if ( defined($r->[$j]) ) {
        print ' ' . $r->[$j];
      } else {
        print ' _';
      }
    }
    printf("\n");
  }
}

sub dump_gtabs
{
  dump_gtab($_) for @g_gtabs;
}

# valid gtab should contain the same amount of values in each row
# 1 either eq amount of colums
# note - row[0] is group name
sub check_gtab
{
  my $t = shift;
  return 0 unless defined($t);
  my $size = scalar(@$t);
  return 0 if ( $size <= 4 );
  # check size of first row
  my $cols = $t->[3];
  my $cs = scalar @$cols;
  my $first = $t->[4];
  my $rs = scalar @$first;
  if ( $rs != 2 && $rs - 1 != $cs ) {
    printf("bad size of first row: %d, cols %d, line %d\n", $rs - 1, $cs, $t->[2]);
    return 0;
  }
  # check remaining rows
  for my $i ( 5 .. $size - 1 ) {
    my $r = $t->[$i];
    my $rsize = scalar @$r;
    if ( $rsize != $rs ) {
      printf("size of row %d: %d, cols %d, line %d\n", $i, $rsize - 1, $cs, $t->[2]);
      return 0;
    }
  }
  1;
}

# parse one table row
sub parse_grow
{
  my($tab, $str, $line) = @_;
  $str =~ s/^\s*//;
  return 1 if ( $str eq '' );
  if ( $str !~ /^(\w+).*:\s*/p ) {
    printf("bad table row %s, line %d\n", substr($str, 0, 64), $line);
    return 0;
  }
  my $name = $1;
  unless ( exists $g_groups{$name} ) {
    printf("unknown group %s at line %d\n", $name, $line);
    return 0;
  }
  $str = ${^POSTMATCH};
  # {d + d}
  if ( $str =~ /\{\s*(\d+)\s*\+\s*(\d+)\s*\}/ ) {
    my $v = $1 + $2;
    push @$tab, [ $name, $v ];
    return 1;
  }
  my @res = ( $name );
  foreach my $v ( split /\s+/, $str ) {
    if ( $v eq '-' ) {
      push @res, undef;
      next;
    }
    if ( $v =~ /^\s*(\d+)\s*$/ ) {
      push @res, int($1);
      next;
    }
    printf("bad row format %s at line %d\n", $str, $line);
    return 0;
  }
  push @$tab, \@res;
  return 1;
}

sub dump_group
{
  my $name = shift;
  unless ( exists $g_groups{$name} ) {
    printf("unknown group %s\n", $name);
    return;
  }
  my $h = $g_groups{$name};
  foreach my $ri ( values %$h ) {
    printf(" %s line %d\n", $ri->[1], $ri->[4]);
  }
}

sub add_ins
{
  my($name, $ins) = @_;
  return unless defined($name);
  unless ( exists $g_nops{$name} ) {
    $g_nops{$name} = [ $ins ];
    return;
  }
  my $v = $g_nops{$name};
  push @$v, $ins;
}

# g1 + g2
sub merge_groups
{
  my($g1, $g2) = @_;
  my %res = %$g1;
  foreach my $i ( keys %$g2 ) {
    $res{$i} = $g2->{$i};
  }
  return \%res;
}

# g1 - g2
sub minus_groups
{
  my($g1, $g2) = @_;
  my %res = %$g1;
  foreach my $i ( keys %$g2 ) {
    delete $res{$i};
  }
  return \%res;
}

sub parse_named_list
{
  my($l, $line) = @_;
  my %res;
  for my $name ( split /\s*,\s*/, $l ) {
    $name =~ s/\s+$//;
    next if ( $name eq '' );
    unless( exists $g_nops{$name} ) {
      printf("unknown instruction name %s, line %d\n", $name, $line);
      next;
    }
    my $v = $g_nops{$name};
    foreach my $i ( @$v ) {
      $res{$i} = $i;
    }
  }
  return \%res;
}

# I am too lazy to implement full-featured LR-parser, so
# bcs expr can be +- {names list} or (expr) we can consider only 3 cases using regexps with /p modifier for tail storing
# thus whole expression evaluates from left to right passing current set and tail to recursive calls of parse_group_tail
# until tail empty
# args: previous set, string, line number
sub parse_group_tail
{
  my($pset, $str, $ln) = @_;
  $str =~ s/^\s+//;
  return $pset if ( $str eq '' ); # condition to stop recursion
  if ( $str =~ /^;\s*/ ) { # tail cannot contain ;
    carp("$str contains ;");
    return $pset;
  }
  # case1: exp +- exp
  if ( $str =~ /^(\+|\-)\s*(\w+)/p ) {
    unless (exists $g_groups{$2}) {
      carp("unknown group $2");
      return $pset;
    }
    return parse_group_tail( $1 eq '+' ? merge_groups($pset, $g_groups{$2}) : minus_groups($pset, $g_groups{$2}),
      ${^POSTMATCH}, $ln);
  }
  # case2: exp +- (exp)
  if ( $str =~ /^(\+|\-)\s*\(\s*(\w+)\s*([^\)]+)\)/p ) {
    my $tail = ${^POSTMATCH};
    my $op = $1;
    unless (exists $g_groups{$2}) {
      carp("unknown group $2");
      return $pset;
    }
    my $tmp = parse_group_tail($g_groups{$2}, $3, $ln);
    return parse_group_tail( $op eq '+' ? merge_groups($pset, $tmp) : minus_groups($pset, $tmp),
      $tail, $ln);
  }
  # case 3: exp +- {names list}
  if ( $str =~ /(\+|\-)\s*\{([^\}]+)\}/p ) {
    my $op = $1;
    my $tail = ${^POSTMATCH};
    my $g = parse_named_list($2, $ln);
    return parse_group_tail( $op eq '+' ? merge_groups($pset, $g) : minus_groups($pset, $g),
      $tail, $ln);
  }
  printf("dont know how to parse tail %s, line %d\n", $str, $ln);
  return $pset;
}

sub read_groups
{
  my($fh, $fname) = @_;
  my($str, $part, $name, $ctab);
  my $state = 0; # 1 - process op set, 2 - table, 3 - table rows
   # 4 - CONNECTOR CONDITION
   # 5 - CONNECTOR SETS
  my $line = 0;
  my $opened = 0; # has non-closed }
  my $reset = sub {
   undef $ctab;
   $state = 0;
  };
  while( $str = <$fh> ) {
    chomp $str;
    $str =~ s/\s*$//;
    $line++;
    next if ( $str eq '' );
    if ( $str =~ /OPERATION SET(?:S?)/ ) {
        $state = 1; next;
    }
    $state = 0 if ( $state > 3 && $str =~ /^TABLE_/ );
    if ( !$state ) {
      if ( $str =~ /CONNECTOR CONDITION(?:S?)/ ) {
        $state = 4; next;
      }
      if ( $str =~ /CONNECTOR SET(?:S?)/ ) {
        $state = 5; next;
      }
      # table can be just table_type(connector): or
      # table_type(connector): first_row
      if ( $str =~ /^TABLE_(\w+)\(([^\)]+)\)\s*:\s*(?:(\w+)(?:\`.*)?)?(?!;)/ ) {
       # probably we should also collect data after ` ?
        if ( defined $3 ) {
          unless(exists $g_groups{$3}) {
            printf("unknown group %s in tab at line %d\n", $3, $line);
            next;
          }
          $ctab = [ $1, $2, $line, [ $3 ] ];
        } else {
          $ctab = [ $1, $2, $line, [] ];
        }
        $state = 2;
        next;
      }
      next if ( !$state );
    }
    if ( 4 == $state ) {
      if ( $str =~ /(\w+)\s*=\s*(.*)\s*;$/ ) {
        $g_ccond{$1} = $2; next;
      } elsif ( $str =~ /CONNECTOR SET(?:S?)/ ) {
        $state = 5; next;
      } else {
        printf("bad CCond %s at line %d\n", substr($str, 0, 64), $line);
        $state = 0;
        next;
      }
    }
    if ( 5 == $state ) {
      if ( $str =~ /(\w+)\s*=\s*(.*)\s*;$/ ) {
        process_cset($1, $2, $line);
        next;
      } else {
        printf("bad CSet %s at line %d\n", substr($str, 0, 64), $line);
        next;
      }
    }
    if ( 2 == $state || 3 == $state ) {
      if ( $str =~ /;$/ || $str =~ /^\w/ ) {
        push @g_gtabs, $ctab if ( defined($ctab) && check_gtab($ctab) );
        $reset->(); next;
      }
      if ( 2 == $state && $str =~ /^\s*(\w+)(?:\`.*)?(?!;)/ ) {
        # probably we should also collect data after ` ?
        unless(exists $g_groups{$1}) {
          printf("unknown group %s in tab column at line %d\n", $1, $line);
          $reset->();
          next;
        }
        my $cols = $ctab->[3];
        push @$cols, $1;
        next;
      }
      if ( 2 == $state && $str =~ /^\s*\{/p ) {
        $state = 3;
        $reset->() if ( !parse_grow($ctab, ${^POSTMATCH}, $line) );
        next;
      }
      if ( 3 == $state ) {
        $reset->() if ( !parse_grow($ctab, $str, $line) );
      }
      next;
    }
    if ( 1 == $state) {
      # seems that all sections start with symbol at column 0
      if ( $str =~ /^\w/ ) {
        $state = 0; next;
      }
    }
# printf("%d: %s\n", $line, $str);
    # group_name = {op_names list separated with commas} |
    #  another group name - then we can just make alias with the same value |
    #  group + expr |
    #  group - expr
    if ( $str =~ /^\s+(\w+)\s*=\s*\{\s*(.*)\s*\}\s*;$/ ) {
      $name = $1;
      $part = parse_named_list($2, $line);
    } elsif ( $str =~ /^\s+(\w+)\s*=\s*\{\s*(.*)\s*\}\s*([^;]+)(;?)$/ ) {
      $name = $1;
      my $tail = $3;
      my $end = defined($4);
      $part = parse_group_tail(parse_named_list($2, $line), $tail, $line);
      next if ( !$end );
    } elsif ( $str =~ /^\s+(\w+)\s*=\s*\{\s*(.*)\s*$/ ) {
      $name = $1;
      $part = parse_named_list($2, $line);
      $opened = 1;
      next;
    } elsif ( $opened && $str =~ /\s*(.*)\s*\}\s*;$/ ) {
      $part = merge_groups($part, parse_named_list($1, $line));
      $opened = 0;
    } elsif ( $opened && $str =~ /\s*(.*)\}\s*([^;]+)(;?)$/ ) {
      $opened = 0;
      my $tail = $2;
      my $end = defined($3);
      $part = merge_groups($part, parse_named_list($1, $line));
      $part = parse_group_tail($part, $tail, $line);
      next if ( !$end );
    } elsif ( $opened && $str =~ /\s*([^;]+)$/ ) {
      $part = merge_groups($part, parse_named_list($1, $line));
      next;
    } elsif ( $str =~ /^\s+(\w+)\s*=\s*(\w+)\s*;$/ ) {
      $name = $1;
      carp("unknown group $2") unless exists $g_groups{$2};
      $part = $g_groups{$2};
    } elsif ( $str =~ /^\s+(\w+)\s*=\s*(\w+)\s*(.*)\s*;/ ) {
      $name = $1;
      carp("unknown group $2") unless exists $g_groups{$2};
      $part = parse_group_tail($g_groups{$2}, $3, $line);
      # name = (grp tail) tail2 ;?
    } elsif ( $str =~ /^\s+(\w+)\s*=\s*\((\w+)\s*([^\)]+)\)\s*([^;]*)\s*(;?)$/ ) {
      $name = $1;
      my $tail = $4;
      my $end = defined($5);
#    printf("%s: end %d tail %s\n", $name, $end, $tail);
      carp("unknown group $2") unless exists $g_groups{$2};
      $part = parse_group_tail($g_groups{$2}, $3, $line);
      $part = parse_group_tail($part, $tail, $line);
      next if ( !$end );
    } else {
      printf("dont know how to parse %s in %s line %d\n", substr($str, 0, 64), $fname, $line);
      next;
    }
    if ( defined $part ) {
      $g_groups{$name} = $part;
      undef $part;
    } else { carp("no result on line $line"); }
    next;
  }
}

### main
my $status = getopts("abBcefFgimprtvwzT:N:C:");
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
# [8] - ref to quoted values
# [9] - list with const banks, [ right, enc1, enc2, ... ]
# [10] - hashmap encoding -> enum
# [11] - list of filters for this instruction
# [12] - map with enums for table-based encoders, key is encoder name
# [13] - count of meaningful bits
# [14] - list of formats
# [15] - href to renaindned enums from [10]
# [16] - original ae hash
# [17] - value names -> type hash
# [18] - idx
# [19] - BRT properties
# [20] - predicates
my($cname, $has_op, $op_line, @op, @enc, @nenc, @quoted, @multi_ops, @cb, @flist, @b_props,
 %ae, $alt, %values, %pipes, %preds, $format);

# table state - estate 3 when we expect table name, 4 - when next string with content
# tref is ref to hash with table content
my($curr_tab, $tref);

# enum state
my($curr_enum, $eref, $e_name);
# 0 - don't parse, 1 - expect start of enum, 2 - continue with next line, 3 - expect start of table, 4 - next line for table
my $estate = 0;

my $reset_tab = sub {
  $g_tabs{$curr_tab} = $tref if ( defined $tref );
  undef $tref;
};

my $reset_enum = sub {
  if ( defined $e_name ) {
    $g_single_enums{$e_name} = 1 if ( is_single_enum($e_name) );
  }
  undef $e_name;
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
  if ( $s =~ /^\"?([\w\.]+)\"?\s*=\s*0b(\w+)$/ ) {
    my $name = $1;
    $curr_enum = parse0b($2);
    $eref->{$1} = $curr_enum++;
    return 1;
  }
  # enum = number
  if ( $s =~ /^\"?([\w\.]+)\"?\s*=\s*(\d+)$/ ) {
    $curr_enum = int($2);
    $eref->{$1} = $curr_enum++;
    return 1;
  }
  # enum $1 (from $2 .. to $3)
  if ( $s =~ /^\"?([\w\.]+)\"?\s*\((\d+)\s*\.\.\s*(\d+)\)$/ ) {
    my $name = $1;
    my $from = int($2);
    my $to = int($3);
    for ( my $i = $from; $i <= $to; $i++ ) {
      my $ename = $name . $i;
      $eref->{$ename} = $curr_enum++;
    }
    return 1;
  }
  # enum $1 (from $2 .. to $3) = (index_from $4 .. index_to $5) - real madness
  if ( $s =~ /^\"?([\w\.]+)\"?\s*\((\d+)\s*\.\.\s*(\d+)\)\s*=\s*\((\d+)\s*\.\.\s*(\d+)\)$/ ) {
    my $name = $1;
    my $from = int($2);
    my $to = int($3);
    my $ifrom = int($4);
    my $ito = int($5);
    if ( $to - $from != $ito - $ifrom ) {
      printf("bad intevals %d-%d and %d-%d\n", $from, $to, $ifrom, $ito);
      return 0;
    }
    $curr_enum = $ifrom;
    for ( my $i = $from; $i <= $to; $i++ ) {
      my $ename = $name . $i;
      $eref->{$ename} = $curr_enum++;
    }
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
  @op = @enc = @nenc = @quoted = @cb = @flist = @b_props = @multi_ops = ();
  %ae = ();
  %values = ();
  %preds = ();
  %pipes = ();
};
# insert copy of current instruction
my $ins_op = sub {
  printf("%s %s %X\n", $cname, $op[0], $op[1]) if ( defined $opt_v );
  if ( !scalar( @enc ) && !scalar( @nenc ) ) {
    printf("%s %s has empty encoding\n", $cname, $op[0]);
    return;
  }
  # fill pairs encoding -> [ enum, optional? ]
  my %cae = %ae;
  my %mae;
  my %missed_ae;
  while( my($a, $e) = each %ae ) {
    $g_used_enums{$e->[0]} //= 1;
    my $what = check_enc(\@enc, $e->[0], $a);
    if ( defined($what) ) { $mae{$what->[0]} = $e; }
    else { $missed_ae{$a} = $e };
  }
  my $miss_size = scalar keys %missed_ae;
  my $tenums = $miss_size ? add_tenums(\@enc, $op[0], $op_line, \%missed_ae): undef;
  # make new instruction
  my @c = @op;
  my @cenc = @enc;
  my @cnenc = @nenc;
  my @cquoted = @quoted;
  my @ccb = @cb;
  my @cflist = @flist;
  move_last_commas(\@cflist);
  $c[3] = $op_line;
  $c[4] = \@cenc;
  $c[5] = \@cnenc;
  $c[6] = $alt;
  $c[7] = $format;
  $c[8] = \@cquoted;
  $c[9] = scalar(@cb) ? \@ccb : undef; # constant bank
  $c[10] = \%mae;
  $c[11] = undef;
  $c[12] = $tenums;
  $c[13] = 0;
  $c[14] = \@cflist;
  $c[15] = $miss_size ? \%missed_ae : undef;
  $c[16] = \%cae;
  if ( keys %values ) {
    my %cvals = %values;
    $c[17] = \%cvals;
  }
  $c[18] = 0;
  if ( scalar(@b_props) ) {
    my @cp = @b_props;
    $c[19] = \@cp;
  } else {
    $c[19] = undef;
  }
  if ( keys %preds ) {
    my %cpreds = %preds;
    $c[20] = \%cpreds;
  } else {
    $c[20] = undef;
  }
  if ( defined($opt_m) ) {
    if ( @multi_ops ) {
      foreach my $pair ( @multi_ops ) {
        my @cc = @c; # shallow copy of original instruction
        $cc[0] = $pair->[0];
        $cc[1] = $pair->[1];
        insert_mask($cname, \@cc);
        if ( defined $opt_g ) { # add all other names for this opcode
          add_ins($pair->[0], \@cc);
          my $v = $pipes{$pair->[1]};
          add_ins($_, \@cc) for @$v;
        }
      }
      # insert last
      insert_mask($cname, \@c);
      if ( defined $opt_g ) { # add all other names for this opcode
        add_ins($op[0], \@c);
        my $v = $pipes{$op[1]};
        add_ins($_, \@c) for @$v;
      }
    } else { # add single version of instruction
      insert_mask($cname, \@c);
      if ( defined($opt_g) ) {
        add_ins($op[0], \@c);
        # and it's pipes
        foreach my $v ( values %pipes ) {
          add_ins($_, \@c) for @$v;
        }
      }
    }
  } else {
    insert_ins($cname, \@c);
  }
};
# consume single enum from string, regex is slightly differs from used in while /g below
my $cons_single = sub {
  my $s = shift;
  if ( $s =~ /(\/?)([\w\.]+)(?:\(\"?([^\)\"]+)\"?(\/PRINT)?\))?\:([\w\.]+)/ ) {
    if ( exists $g_enums{$2} ) {
      my $aref = [ $2, $1 ne '', defined($3) ? $g_enums{$2}->{$3} : undef, $5, defined($4) ];
      $ae{$5} = $aref;
      return $aref;
    }
  }
  undef;
};
# consime single value from string, put in %values key name, value - format
my $cons_value = sub {
  my $s = shift;
  if ( $s =~ /([\w\.]+)(?:\((?:[^\)]+\))?(\*)?\:([\w\.]+))/ ) {
    if ( is_type($1) ) {
      $values{$3} = [ $1, $2 ];
      return $3;
    }
  }
  undef;
};
# parse format in form /? $1 enum $2 optional value $3 alias $4
# format: 0 - letter, 1 - prefix, 2 - suffix, 3 - [x]
my $cons_ae;
$cons_ae = sub {
  my($s, $idx) = @_;
  if ( !$idx ) { # zero index - predicate
    # $1 - @?, $2 - symbol inside [], $3 - enum, $4 - default, $5 - value name
    if ( $s =~ /(\@?)\s*(?:\[(.)\]\s*)?([\w\.]+)(?:\(\"?([^\)\/\"]+)\"?\))?\:([\w\.]+)/ )
    {
      if ( exists $g_enums{$3} ) {
        my $aref = [ $3, 0, defined($4) ? $g_enums{$3}->{$4} : undef, $5 ];
        $ae{$5} = $aref;
        push @flist, [ 'P', $1, undef, $2, $aref ];
      } else {
        carp("unknwon enum $3 for predicate");
      }
      return;
    }
  }
  # collapse $( ... )$
  $s =~ s/\$\((.*)\)\$/$1/;
  # try const bank address, v1
  # $1 - optional comma, $2 - [x], $3 - [||]?, $4 - name after C:, $5 - first [], $6 - second []
  # C: can have [~] - it applied to name after prefix C: like in sample from sm75_1.txt:
  #  [~] CX:Sb[UniformRegister:URb][UImm(16)*:Sb_offset]
  # here we have value Sb@invert
  if ( $s =~ /(\'\,\'\s*)?(?:\[(.)\]\s*)?(\[\|\|\]\s*)?\s*\bC\:([^:\[]+)\s*\[([^\]]+)\]\*?\s*\[([^\]]+)\]/p ) {
        # 0 - type   1 - optional ,              2    3   4
    my @cf = ( 'C', defined($1) ? ',' : undef, undef, $2, $4, $5);
    # see details here: https://perldoc.perl.org/perlretut#Position-information
    my $left = $5;
    my $right = $6;
    $s = ${^POSTMATCH};
    # process left part
    $cons_ae->(${^PREMATCH}, $idx+1);
    # get var from left
    if ( $left =~ /\:(\w+)$/ ) { $cf[5] = $1; $cons_value->($left); }
    # check if right is pair enum + var
    if ( $right =~ /^\s*(.*\:.*)\s*\+\s*(.*\:.*)\s*$/ ) {
      # we have 2 parts
      my $sec = $2;
      my $first = $1;
      $cf[7] = $cons_single->($first);
      if ( $sec =~ /\:(\w+)$/ ) { $cf[6] = $1; $cons_value->($sec); }
    } elsif ( $right =~ /\:(\w+)\s*$/ ) { $cf[6] = $1; $cons_value->($right); }
    # push newly created format
    push @flist, \@cf;
  # try const bank address, v2 with CX prefix - the same as above but don't need to check $op->[10] for version
  # $1 - optional comma, $2 - [x], $3 - [||]?, $4 - name after CX:, $5 - first [], $6 - second []
  } elsif ( $s =~ /(\'\,\'\s*)?(?:\[(.)\]\s*)?(\[\|\|\]\s*)?\s*\bCX\:([^:\[]+)\s*\[([^\]]+)\]\*?\s*\[([^\]]+)\]/p ) {
        # 0 - type   1 - optional ,               2    3    4   5
    my @cf = ( 'X', defined($1) ? ',' : undef, undef, $2, $4, $5);
    # see details here: https://perldoc.perl.org/perlretut#Position-information
    my $spos = $-[0];
    my $slen = $+[0] - $-[0];
    my $left = $5;
    my $right = $6;
    my $reps = '<CX>';
    # get var from left
    $cf[5] = $cons_single->($left);
    # check if right is pair enum + var
    if ( $right =~ /^\s*(.*\:.*)\s*\+\s*(.*\:.*)\s*$/ ) {
      # we have 2 parts
      my $sec = $2;
      my $first = $1;
      $cf[7] = $cons_single->($first);
      # $reps = '<CX ' . $first . ' >' if defined($cf[7]);
      $cf[6] = $cons_value->($sec);
    } else { $cf[6] = $cons_value->($right); }
    # push newly created format
    push @flist, \@cf;
    # remove this part from $s
    substr($s, $spos, $slen, $reps);
  } # DESC: $1 - optional comma, $2 - lest, $3 - right + $4
    # DESC:memoryDescriptor[UniformRegister:Ra_URc][Register:Ra /ONLY64:input_reg_sz_64_dist + SImm(24/0)*:Ra_offset]
    # so 4 is enum in left [], right[ 5 - enum optionally 6 - enum + 7 value]
  elsif ( $s =~ /(\'\,\'\s*)?\s*\bDESC\:\s*(?:[^\:\[]+)?\s*\[([^\]]+)\]\*?\s*\[([^\]]+)\s*\+\s*([^\]]+)\]/p ) {
        # 0 - type   1 - optional ,               2    3    4
    my @cf = ( 'D', defined($1) ? ',' : undef, undef, undef, $2);
    # see details here: https://perldoc.perl.org/perlretut#Position-information
    my $spos = $-[0];
    my $slen = $+[0] - $-[0];
    my $left = $2;
    my $right = $3;
    my $add = $4;
    my $reps = '<DESC>';
    # get enum from left
    $cf[4] = $cons_single->($left);
    # var from add
    $cf[7] = $cons_value->($add);
    # check if right is pair enum + var
    if ( $right =~ /^\s*(.*\:.*)\s+(.*\:.*)\s*$/ ) {
      # we have 2 parts
      my $sec = $2;
      my $first = $1;
      $cf[5] = $cons_single->($first);
      $cf[6] = $cons_single->($sec);
    } else {
      $cf[5] = $cons_single->($right);
    }
    # validate
    if ( defined($cf[4]) && defined($cf[5]) &&  defined($cf[7])) {
      # push newly created format
      push @flist, \@cf;
      # remove this part from $s
      substr($s, $spos, $slen, $reps);
    }
  } elsif ( $s =~ /(\'\,\'\s*)?\s*\b(RF|TMA|[UG]MMA|UMMA[AB])\:\s*(?:[^\:\[]+)?\s*\[([^\]]+)\]/p ) {
    my @cf = ( 'M1', defined($1) ? ',' : undef, undef, $2, $3);
    my $spos = $-[0];
    my $slen = $+[0] - $-[0];
    my $reps = '<' . $2 . '>';
    $cf[4] = $cons_single->($3);
    if ( defined $cf[4] ) {
      push @flist, \@cf;
      # remove this part from $s
      substr($s, $spos, $slen, $reps);
    }
  } elsif ( $s=~ /(\'\,\'\s*)?\s*\b(TMEM[ABCEI])\:\s*(?:[^\:\[]+)?\s*\[([^\]]+)\]/p ) {
    my @cf = ( 'M2', defined($1) ? ',' : undef, undef, $2, $3);
    my $spos = $-[0];
    my $slen = $+[0] - $-[0];
    my $reps = '<' . $2 . '>';
    my $body = $3;
    if ( $body =~ /^\s*(.*\:.*)\s*\+\s*(.*\:.*)\s*$/ ) {
      my $f = $1;
      my $s = $2;
      $cf[4] = $cons_single->($f);
      $cf[5] = $cons_value->($s);
    } else {
      $cf[4] = $cons_single->($body);
    }
    if ( defined $cf[4] ) {
      push @flist, \@cf;
      # remove this part from $s
      substr($s, $spos, $slen, $reps);
    } # TTU $1 - optional comma, $2 - body in []
  } elsif ( $s =~ /(\'\,\'\s*)?\s*\bTTU\:\s*(?:[^\:\[]+)?\s*\[([^\]]+)\]/p ) {
    my @cf = ( 'T', defined($1) ? ',' : undef, undef, undef, $3);
    my $spos = $-[0];
    my $slen = $+[0] - $-[0];
    my $reps = '<TTU>';
    $cf[4] = $cons_value->($2);
    if ( defined $cf[4] ) {
      push @flist, \@cf;
      # remove this part from $s
      substr($s, $spos, $slen, $reps);
    }
  } elsif ( $s =~ /(\'\,\'\s*)?\s*\bA\:\s*(?:[^\:\[]+)?\s*\[([^\]]+)\]/p ) {
    my @cf = ( 'A', defined($1) ? ',' : undef, undef, undef, $2);
    my $spos = $-[0];
    my $slen = $+[0] - $-[0];
    my $reps = '<A>';
    my @a = split(/\s+\+\s+/, $2);
    # 1st can be value or enum
    my $is_e = $cons_single->($a[0]);
    if ( defined($is_e) ) { $cf[4] = $is_e; }
    else { $cf[4] = $cons_value->($a[0]); }
    # optional 2nd is value
    if ( scalar(@a) > 1 ) {
      $cf[5] = '+';
      $cf[6] = $cons_value->($a[1]);
    }
    # add formal list
    if ( defined $cf[4] ) {
      push @flist, \@cf;
      # remove this part from $s
      substr($s, $spos, $slen, $reps);
    }
  } # generic []
  elsif ( $s =~ /(\'\,\'\s*)?\s+\[\s*([^\]\-\!\~\|]+)\s*\]/p ) {
    my @cf = ( '[', defined($1) ? ',' : undef, undef, undef);
    my $second = $2;
    $s = ${^POSTMATCH};
    # process left part
    $cons_ae->(${^PREMATCH}, $idx+1);
    my $is_ok = 1;
    my @a = split(/\s+\+\s+/, $second);
    for ( my $i = 0; $i < scalar @a; $i++ ) {
      push @cf, '+' if ( $i );
      my @curr = split /\s+/, $a[$i];
      if ( 2 == @curr ) {
        # both should be enums
        my $e1 = $cons_single->($curr[0]);
        my $e2 = $cons_single->($curr[1]);
        if ( !defined($e1) || !defined($e2) ) {
          printf("cannot parse mem idx %d, left %s right %s, line %d\n", $i, $curr[0], $curr[1], $line);
          $is_ok = 0;
          last;
        }
        push @cf, $e1;
        push @cf, $e2;
      } else {
        my $is_e = $cons_single->($a[$i]);
        $is_e = $cons_value->($a[$i]) unless defined($is_e);
        if ( !defined $is_e ) {
          printf("cannot parse mem idx %d: %s, line %d\n", $i, $a[$i], $line);
          $is_ok = 0;
          last;
        }
        push @cf, $is_e;
      }
    }
    if ( $is_ok ) {
      push @flist, \@cf;
    }
  } # last one to check what remainded
  elsif ( $s =~ /\b(\w+)\:(?:[^\:\[]+)?\s*\[([^\]]+)\]/ ) {
    printf("UNKNOWN MEM %s [%s], line %d\n", $1, $2, $line) if ( !exists $g_enums{$1} );
  }
  # first 3 is optional comma, $2 - [x], $3 - [||]?
  # next $4 - /?, $5 - enum, $6 - def_value, $7 - optional /PRINT,  $8 - alias $9 - leading char
  while( $s =~ /(?:\'(\,|\?)\'\s*)?(?:\[(.)\]\s*)?(\[\|\|\]\s*)?\s*(\/?)([\w\.]+)(?:\(\"?([^\)\"]+)\"?(\/PRINT)?\))?\*?\:([\w\.]+)\s*(\',\')?/g ) {
    if ( exists $g_enums{$5} ) {
      next if ( exists $g_bad_enums{$5} );
      # key is values in op->[11] hash is
      # 0 - enum name
      # 1 - if / presents
      # 2 - default value if exists
      # 3 - format name
      # 4 - if /PRINT presents
      my $aref = [ $5, $4 ne '', defined($6) ? $g_enums{$5}->{$6} : undef, $8, defined($7) ];
      $ae{$8} = $aref;
      if ( defined($2) and $2 eq '!' and defined($6) ) {
        push @flist, [ 'P', $1, defined($9) ? ',' : undef, $2, $aref ];
      } else {
        push @flist, [ 'E', $1, defined($9) ? ',' : undef, $2, $aref ];
      }
    } elsif ( is_type($5) ) {
      $values{$8} = [ $5, undef ];
      push @flist, [ 'V', $1, defined($9) ? ',' : undef, $2, $8, $5 ];
    } else {
       printf("enum %s does not exists, line %d\n", $5, $line);
    }
  }
};
$reset->(); $reset_enum->(); $reset_tab->();
while( $str = <$fh> ) {
  chomp $str;
  $line++;
  if ( !$state ) {
    if ( $str =~ /ENCODING\s+WIDTH\s+(\d+)\s*\;/ ) {
       $g_min_len = $g_size = int($1);
       $state = 1;
       next;
    }
    if ( !defined($g_size) && $str =~ /^\s+VQ_(\w+)\s*=\s*(\d+)/ ) {
      $g_vq{int($2)} = 'VQ_' . $1;
      next;
    }
    # check tables
    if ( $str =~ /^\s*OPERATION\s+PROPERTIES/ ) {
      $estate = 0;
      next;
    }
    if ( $str =~ /^\s*TABLES\s*$/ ) {
      $estate = 3;
      next;
    }
# printf("e%d %s\n", $estate, $str);
    if ( $estate == 3 ) {
     $str =~ s/\s+//g;
     next if ( $str eq '' );
     $curr_tab = $str;
     $estate = 4;
     next;
    }
    if ( $estate == 4 ) {
      if ( $str =~ /^\s*(.*)\-\>\s*(.*)\s*$/ ) {
       # values in $1, key $2
       my $key = $2;
       my $kv;
       my $v = parse_tab_keys($1, $line);
       if ( defined($v) && defined($kv = parse_tab_value($key, $line)) ) {
         if ( defined $tref ) { $tref->{$kv} //= $v; }
         else { $tref = { $kv => $v }; }
       }
      }
      if ( $str =~ /\;\s*$/ ) {
        $reset_tab->();
        $estate = 3;
        next;
      }
    }
    # check enums
    if ( $str =~ /^\s*REGISTERS/ ) {
      $estate = 1;
      next;
    }
    if ( $str =~ /^\s*ZeroRegister .*\"?RZ\"?\s*=\s*(\d+)\s*;/ )
    {
      $estate = 1;
      $g_rz = int($1);
    }
    $estate = 1 if ( !$estate && $str =~ /^\s*SpecialRegister / );
    next if ( !$estate );
#printf("e%d %s\n", $estate, $str);
    # 1 - new enum
    if ( 1 == $estate ) {
      $str =~ s/^\s*//; $str =~ s/\s*$//;
      if ( $str =~ /^([\w\.]+)$/ ) {
        my %tmp;
        $e_name = $1;
        $eref = $g_enums{$1} = \%tmp;
        $estate = 2;
        next;
      }
      # compound enum like name = e1 + ...;
      if ( $str =~ /^(\w+)\s*=(.*);$/ ) {
        my %tmp;
        $e_name = $1;
        $eref = $g_enums{$1} = \%tmp;
        $str = $2;
        foreach my $cname ( split /\s*\+\s*/, $str ) {
          $cname =~ s/^\s*//;
          merge_enum($eref, $cname, $line);
        }
        $reset_enum->();
        next;
      }
      if ( $str =~ /^([\w\.]+)\s+(.*)\s*;?/ ) {
        my %tmp;
        $e_name = $1;
        $eref = $g_enums{$1} = \%tmp;
        $parse_enum->($2);
        if ( $str =~ /\;$/ ) {
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
  $str =~ s/CC\(CC\)TestCC\/Test\(T\)\:(\w+)/ \/Test\(T\)\:Test /g if ( $state == 2 );
  if ( $state == 2 && $str =~ /FORMAT\s+(?:PREDICATE\s+)?(.*)Opcode\s*?(.*)$/ ) {
    $format = $2;
    $cons_ae->($1, 0);
    push @flist, ['$'];
    $cons_ae->($2, 1);
    $state = 6 if ( $str !~ /;\s*$/ );
    next;
  }
  if ( 6 == $state ) {
    if ( $str !~ /FORMAT\s+(?:PREDICATE\s+)?.*Opcode/ ) {
      $str =~ s/\s*$//;
      # fix TestCC/Test like
      # CC(CC):TestCC/Test(T):fcomp
      $str =~ s/CC\(CC\)TestCC\/Test\(T\)\:(\w+)/ \/Test\(T\)\:Test /g;
      # grab /? $1 enum $2:alias $3 into %ae
      $cons_ae->($str, 2);
      # grab /something()
      if (  $str =~ /^\s*(.*\/\w.*)$/ ) {
        $format .= ' ' . $1;
      } elsif ( $str =~ /^\s*(.*\(\d+\).*)$/ ) {
        $format .= ' ' . $1;
      }
    }
    $state = 2 if ( $str =~ /;$/ );
    next;
  }
  if ( 6 == $state && $str =~ /CONDITIONS/ ) {
    $state = 2;
    next;
  }
  if ( ($state == 2 || $state >= 7) && $str =~ /OPCODES/ ) {
    $state = 3;
    next;
  }
  # parse opcode
  if ( $state == 3 && $str =~ /^\s*(\S+)\s*=\s*(\S+);/ ) {
    my $name = $1;
    my $vs = $2;
    my $value;
    if ( $vs =~ /^(\d+)$/ ) { $value = int($1); }
    else { $value = parse0b($vs); }
    # skip pipe version
    if ( $name =~ /_pipe/ ) {
      if ( defined $pipes{$value} ) {
        my $v = $pipes{$value};
        push @$v, $name;
      } else { $pipes{$value} = [ $name ]; }
      next;
    }
    if ( !defined $op[1] ) {
      $op[0] = $name;
      $op[1] = $value;
    } else { # we have several opcodes for 1 md - put pair [name value] into multi_ops
      push @multi_ops, [ $name, $value ];
    }
  }
  # properties
  if ( $str =~ /^\s*PROPERTIES/ ) {
    $state = 7;
    next;
  }
  if ( 7 == $state ) {
    if ( $str =~ /^\s*BRANCH_TYPE =\s*(\S+)\s*;/ ) {
      $b_props[0] = $1;
      next;
    }
    if ( $str =~ /^\s*MEM_SCBD\s*=\s*(\S+)\s*;/ ) {
      $b_props[1] = $1 if ( $1 ne 'NONE' );
      next;
    }
    if ( $str =~ /^\s*MEM_SCBD_TYPE\s*=\s*(\S+)\s*;/ ) {
      $b_props[2] = $1 if ( $1 ne 'ALL' );
      next;
    }
    if ( $str =~ /^\s*BRANCH_TARGET_INDEX = INDEX\(([^\)]+)\)/ ) {
      $b_props[3] = $1;
      next;
    }
    if ( $str =~ /CC_INDEX = INDEX\(([^\)]+)\)/ ) {
      $b_props[4] = $1;
      next;
    }
    # SIDL_NAME
    if ( $str =~ /SIDL_NAME\s*=\s*`SIDL_NAMES@(\w+)/ ) {
      $b_props[5] = $1;
      next;
    }
  }
  # predicates
  if ( defined($opt_p) && $str =~ /^\s*PREDICATES/ ) {
    $state = 8;
    next;
  }
  if ( 8 == $state ) {
    next if ( $str !~ /^\s*([\w\.]+)\s*=\s*([^;]+)\s*;/ );
    if ( $1 eq 'VIRTUAL_QUEUE' ) {
      my $s = $2;
      $s =~ s/\$VQ/VQ/g;
      $preds{'VQ'} = $s;
      next;
    }
    next if ( $2 eq '0' );
    next if ( $2 eq '(0)' );
    next if ( $2 =~ /\(\s*0\s*\)\s*\*\s*\d+$/ );
    $preds{$1} = $2;
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
      # constbank 1 - encoding list, 2 - remainder
      if ( $s =~ /^(.*)\s*=\s*ConstBankAddress(.*)\s*$/ ) {
        if ( scalar @cb ) {
          printf("duplicated ConstBankAddress for %s on line %d\n", $op[0], $line);
          next;
        }
        push @cb, $2;
        my $cbenc = $1;
        foreach my $em ( split /\s*,\s*/, $cbenc ) {
          # check if encode mask exist
          $em =~ s/\s+//g;
          if ( !exists $g_mnames{$em} ) {
            printf("bank enc %s not exists, line %d op %s\n", $em, $line, $op[0]);
            @cb = ();
            next;
          }
          push @cb, $em;
        }
        next;
      }
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
          push(@quoted, $s);
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

dump_enums() if ( defined $opt_e );
dump_tabs() if ( defined $opt_t );

if ( defined $opt_g ) {
  my $fname = $ARGV[0];
  $fname =~ s/_1\.([^\.]+)$/_2\.\1/;
  open($fh, '<', $fname) or die("cannot open $fname, error $!");
  read_groups($fh, $fname);
  close $fh;
  # dump_group('UPRED_OPS');
  dump_gtabs();
}

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
#  with encoded =* const
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
#  /PRINT suffix - in 5x
# total          359 383 396 433  570  1064
# duplicated      90  92 151 157   34    58
#  without -c option
# total          359 383 405 446  581  1063  1050   978  1099
# duplicated      90  92 142 144   26    52    96    91   108
#  apply single enums where enc =* alias
# total          365 389 420 460  599  1102  1110  1013  1157
# duplicated      84  86 127 128   10    26    28    28    35
#  enum can contains '.'
# total          365 389 420 460  602  1107  1113  1036  1160
# duplicated      84  86 127 128    7    21    25    25    32
#  -F option
# total          365 389 422 464  602  1110  1118  1041  1165
# duplicated      84  86 125 124    7    19    20    20    27
#  compound enums
# total          367 393 435 476  602  1110  1118  1041  1165
# duplicated      82  82 112 112    7    19    20    20    27
# 11 apr - 1 instruction md can have several different opcodes
# total          383 405 436 481  597  1096  1104  1029  1127
# duplicated      92  94 112 109   12    33    32    30    46
  if ( defined($opt_T) || defined($opt_N) ) {
    $g_dec_tree = build_tree() if ( defined($opt_B) );
    make_single_test() if defined($opt_N);
    make_test($opt_T) if defined($opt_T);
  } else {
    if ( defined($opt_B) ) {
      $g_dec_tree = build_tree();
      printf("min mask len %d\n", $g_min_len);
      gen_C() if defined($opt_C);
    } else {
      dump_dup_masks();
      dump_incompleted();
      printf("%d duplicates (%d different names), total %d\n", $g_dups, $g_diff_names, scalar keys %g_masks);
    }
  }
} else {
  dump_negtree(\%g_zero);
  printf("--- opcodes tree\n");
  dump_tree(\%g_ops, 0);
  printf("%d different names\n", $g_diff_names);
}