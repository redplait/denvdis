#!perl -w
# some nvdisasm encoding analysis
use strict;
use warnings;
use Getopt::Std;
use Carp;
use Data::Dumper;

# options
use vars qw/$opt_a $opt_c $opt_e $opt_f $opt_m $opt_r $opt_t $opt_T $opt_v $opt_w/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] md.txt
 Options:
  -a - add alternates
  -c - use format constant to form mask
  -e - dump enums
  -f - dump fully filled masks
  -m - generate masks
  -r - fill in reverse order
  -t - dunp tables
  -T - test bytes
  -v - verbose
  -w - dump warnings
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

sub is_single_enum
{
  my $name = shift;
  return undef if ( !exists $g_enums{$name} );
  my $k = $g_enums{$name};
  my @keys = keys %$k;
  return undef if ( 1 != scalar @keys );
  $k->{$keys[0]};
}

# global tables hash map, key is name of table, value is another hash map { value -> [ literals list ] }
my %g_tabs;

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
   return undef;
  }
  printf("unknown table value %d, line %d\n", $s, $line);
  undef;
}

sub parse_tab_key
{
  my($s, $line) = @_;
  # check in enums
  if ( $s =~ /(\w+)@\"?(\w+)\"?\s*$/ ) {
    return $2 if ( exists $g_enums{$1} );
    printf("unknown enum %s for table key, line %d\n", $1, $line) if ( defined $opt_v );
    return $2;
  }
  $s =~ s/\'//g;
  return $s;
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
      return undef if ( !defined $next );
      push @res, $next;
    }
    return undef if ( !scalar @res );
    return \@res;
  } else {
    # just some literal
    return parse_tab_key($s, $line);
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

sub mask_len
{
  my $op = shift;
  my $res = 0;
  my $list = $op->[3];
  for ( my $i = 0; $i < scalar @$list; $i += 2 ) {
    $res = $list->[$i+1];
  }
  return $res;
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

# the same cmp_mask but second argument is arrey to ref
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
          return undef;
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
          return undef;
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
      return undef unless ( defined($str = <$fp>) );
      my @a;
      while ( $str =~ /([0-9a-f]{2})/ig ) {
       push @a, hex($1);
       $i++;
      }
      if ( 8 != $i ) {
        carp("bad control word, len %d", $i);
        return undef;
      }
      if ( defined $opt_v ) { printf("%2.2X ", $_) for @a; }
      $v = bit_array_rev(\@a);
      printf("\n%s\n", join '', @$v) if ( defined $opt_v );
    }
    return undef unless ( defined($str = <$fp>) );
    my @l;
    $i = 0;
    while ( $str =~ /([0-9a-f]{2})/ig ) {
      push @l, hex($1);
      $i++;
    }
    if ( 8 != $i ) {
      carp("bad word, len %d", $i);
        return undef;
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
    return undef;
  }
  if ( defined $opt_v ) {
    printf("%2.2X      ", $_) for @p; printf("\n"); }
  return bit_array(\@p);
}

sub conv
{
  my $fp = shift;
  sub {
    my $str;
    return undef unless ( defined($str = <$fp>) );
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
 return undef;
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
    if ( $emask =~ /^(\w+)\s*=\*?\s*0b(\S+)/ ) { # enc =*? 0bxxx
      my $mask = $g_mnames{$1};
      $rem{$1} = $emask;
      mask_value(\@res, parse0b($2), $mask);
    } elsif ( $emask =~ /^(\w+)\s*=\*?\s*(\d+)/ ) {
      $rem{$1} = $emask;
      mask_value(\@res, int($2), $g_mnames{$1});
    } elsif ( $emask =~ /^(\w+)\s*=\*?\s*0x(\w+)/i ) {
      $rem{$1} = $emask;
      mask_value(\@res, hex($2), $g_mnames{$1});
    }
  }
  if ( scalar keys %rem ) {
    remove_encs($op, \%rem);
    %rem = ();
  }
  # enc = `const - in op->[9]
  foreach my $q ( @{ $op->[9] } ) {
    if ( $q =~ /^(\w+)\s*=\s*\`(\S+)/ ) {
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
  }
  # check /enum:alias where enc =* alias
  while( $op->[8] =~ /\/([^\(\)\"\s]+)\:([\w\.]+)/pg ) {
    my $alias = $2;
    my $what;
    my $v = is_single_enum($1);
    if ( defined($v) && defined($what = check_enc_ask($op->[5], $2)) ) {
       $rem{$what->[0]} = 1;
       my $p = pos($op->[8]);
       push @pos, [ $p - length(${^MATCH}), $p ];
       mask_value(\@res, $v, $what);
    }
  }
  # check /Group(Value):alias in format - Value can contain /PRINT suffix
  while ( $op->[8] =~ /\/(\w+)\(\"?([^\"\)]+)\"?(\/PRINT)?\)\:([\w\.]+)/pg ) {
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
  }
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
   $op->[11] = $cp;
  }
  remove_encs($op, \%rem) if ( scalar keys %rem );
  return join('', @res);
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
        return undef;
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
# [11] - string with unused formats
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

# args - ref to array, ref to found instruction
sub dump_values
{
  my($a, $op) = @_;
  my $enc = $op->[5];
  foreach my $m ( @$enc ) {
    if ( $m =~ /^(\w+)/ ) {
     my $mask = $g_mnames{$1};
     my $v = extract_value($a, $mask);
     if ( defined($v) ) {
       printf("   %s(", $mask->[0]);
       if ( $v ) { printf("%X)\n", $v); }
       else { printf("0)\n"); }
    }
   }
  }
  # dump const bank
  if ( defined $op->[10] ) {
    my $cb = $op->[10];
    printf(" -- const bank %s\n", $cb->[0]);
    my $cb_len = scalar @$cb;
    for ( my $i = 1; $i < $cb_len; $i++ ) {
     my $mask = $g_mnames{$cb->[$i]};
     my $v = extract_value($a, $mask);
     if ( defined($v) ) {
       printf("   %s(", $mask->[0]);
       if ( $v ) { printf("%X)\n", $v); }
       else { printf("0)\n"); }
    }
   }
  }
}

sub make_test
{
  my $fn = shift;
  my($fh, $b);
  open($fh, '<', $fn) or die("cannot open $fn, error $!");
  my $cf = ($g_size == 88) ? martian88($fh) : conv($fh);
  while( defined($b = $cf->()) ) {
    printf("%s:", join '', @$b);
    # try to find in all masks - very slow
    my $found = 0;
    foreach my $m ( keys %g_masks ) {
      if ( cmp_maska($m, $b) ) {
        printf("\n") if ( !$found );
        printf("%s - ", $m);
        my $ops = $g_masks{$m};
        printf("%s %d items\n", $ops->[0]->[1], scalar(@$ops));
        # extract all masks values
        dump_values($b, $ops->[0]);
        $found++;
        # last; # find first mask
      }
    }
    printf(" NOTFound\n") if ( !$found );
  }
  close $fh;
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
      printf("   Unused %s\n", $op->[11]) if defined($op->[11]);
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
my $status = getopts("acefmrtvwT:");
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
# [9] - list with const banks, [ right, enc1, enc2, ... ] 
my($cname, $has_op, $op_line, @op, @enc, @nenc, @tabs, @cb, $alt, $format);

# table state - estate 3 when we expect table name, 4 - when next string with content
# tref is ref to hash with table content
my($curr_tab, $tref);

# enum state
my($curr_enum, $eref);
# 0 - don't parse, 1 - expect start of enum, 2 - continue with next line
my $estate = 0;

my $reset_tab = sub {
  $g_tabs{$curr_tab} = $tref if ( defined $tref );
  undef $tref;
};

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
  # enum $1 (from $2 .. to $3)
  if ( $s =~ /^\"?([\w\.]+)\"?\s*\((\d+)\.\.(\d+)\)\s*$/ ) {
    my $name = $1;
    my $from = int($2);
    my $to = int($3);
    for ( my $i = $from; $i <= $to; $i++ ) {
      my $ename = $name . $i;
      $eref->{$1} = $curr_enum++;
    }
    return 1;
  }
  # enum $1 (from $2 .. to $3) = (index_from $4 .. index_to $5) - real madness
  if ( $s =~ /^\"?([\w\.]+)\"?\s*\((\d+)\.\.(\d+)\)\s*=\s*\((\d+)\.\.(\d+)\)\s*$/ ) {
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
      $eref->{$1} = $curr_enum++;
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
  @op = @enc = @nenc = @tabs = @cb = ();
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
  my @ccb = @cb;
  $c[3] = $op_line;
  $c[4] = \@cenc;
  $c[5] = \@cnenc;
  $c[6] = $alt;
  $c[7] = $format;
  $c[8] = \@ctabs;
  $c[9] = scalar(@cb) ? \@ccb : undef;
  if ( defined($opt_m) ) {
    insert_mask($cname, \@c);
   } else {
    insert_ins($cname, \@c);
   }
};
$reset->(); $reset_enum->(); $reset_tab->();
while( $str = <$fh> ) {
  chomp $str;
  $line++;
  if ( !$state ) {
    if ( $str =~ /ENCODING\s+WIDTH\s+(\d+)\s*\;/ ) {
       $g_size = int($1);
       $state = 1;
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
         if ( defined $tref ) { $tref->{$kv} = $v; }
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
      next;
    }
    $estate = 1 if ( !$estate && $str =~ /^\s*SpecialRegister / );
    next if ( !$estate );
# printf("e%d %s\n", $estate, $str);
    # 1 - new enum
    if ( 1 == $estate ) {
      if ( $str =~ /^\s*([\w\.]+)\s*$/ ) {
        my %tmp;
        $eref = $g_enums{$1} = \%tmp;
        $estate = 2;
        next;
      }
      next if ( $str =~ /^\s*(\w+)\s*=/ );
      if ( $str =~ /^\s*([\w\.]+)\s+(.*)\s*;?/ ) {
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

dump_enums() if ( defined $opt_e );
dump_tabs() if ( defined $opt_t );

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
  if ( defined $opt_T ) {
    make_test($opt_T);
  } else {
    dump_dup_masks();
    printf("%d duplicates (%d different names), total %d\n", $g_dups, $g_diff_names, scalar keys %g_masks);
  }
} else {
  dump_negtree(\%g_zero);
  printf("--- opcodes tree\n");
  dump_tree(\%g_ops, 0);
  printf("%d different names\n", $g_diff_names);
}