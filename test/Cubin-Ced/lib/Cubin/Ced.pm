package Cubin::Ced;

use 5.030000;
use strict;
use warnings;

require Exporter;
use AutoLoader qw(AUTOLOAD);

our @ISA = qw(Exporter);

# Items to export into callers namespace by default. Note: do not export
# names by default without a very good reason. Use EXPORT_OK instead.
# Do not simply export all your public functions/methods/constants.

# This allows declaration	use Cubin::Ced ':all';
# If you do not need this, moving things directly into @EXPORT or @EXPORT_OK
# will save memory.
our %EXPORT_TAGS = ( 'all' => [ qw(
	
) ] );

use Elf::Reader;

sub exs
{
  my $elf = shift;
  my $secs = $elf->secs();
  return unless defined($secs);
  # 2 - type must be SHT_PROGBITS, 9 - size
  my @res = grep { $_->[2] == SHT_PROGBITS && $_->[1] =~ /^\.text/ && $_->[9] } @$secs;
  return wantarray ? @res : \@res;
}

our %PTypes = (
 0 => "INTEGER",
 1 => "SIGNED_INTEGER",
 2 => "UNSIGNED_INTEGER",
 3 => "FLOAT",
 4 => "DOUBLE",
 5 => "GENERIC_ADDRESS",
 6 => "SHARED_ADDRESS",
 7 => "LOCAL_ADDRESS",
 8 => "TRAM_ADDRESS",
 9 => "LOGICAL_ATTR_ADDRESS",
 10 => "PHYSICAL_ATTR_ADDRESS",
 11 => "GENERIC",
 13 => "CONSTANT_ADDRESS",
 14 => "VILD_INDEX",
 15 => "VOTE_INDEX",
 16 => "STP_INDEX",
 17 => "PIXLD_INDEX",
 18 => "PATCH_OFFSET_ADDRESS",
 19 => "RAW_ISBE_ACCESS",
 20 => "GLOBAL_ADDRESS",
 21 => "TEX",
 22 => "GS_STATE",
 23 => "SURFACE_COORDINATES",
 24 => "FP16SIMD",
 25 => "BINDLESS_CONSTANT_ADDRESS",
 26 => "VERTEX_HANDLE",
 27 => "MEMORY_DESCRIPTOR",
 28 => "FP8SIMD",
 29 => 'TMEM_ADDRESS',
 30 => 'FLOAT128',
);

sub PType_name($)
{
  my $p = shift;
  return $PTypes{$p} if exists($PTypes{$p});
  undef;
}

our %RTypes = (
 1 => 'R_value',
 2 => 'R_enum',
 3 => 'R_predicate',
 4 => 'R_opcode',
 5 => 'R_C',
 6 => 'R_CX',
 7 => 'R_TTU',
 8 => 'R_M1',
 9 => 'R_desc',
 10 => 'R_mem',
);

sub RType_name($)
{
  my $p = shift;
  return $RTypes{$p} if exists($RTypes{$p});
  undef;
}

our %ITypes = (
 1 => 'ABC_REG',
 2 => 'ABC_BCST',
 3 => 'ABC_CCST',
 4 => 'ABC_B20I',
);

sub IType_name($)
{
  my $p = shift;
  return $ITypes{$p} if exists($ITypes{$p});
  undef;
}

our %PRs = (
 0 => 'IDEST',
 1 => 'IDEST2',
 2 => 'ISRC_A',
 3 => 'ISRC_B',
 4 => 'ISRC_C',
 5 => 'ISRC_E',
 6 => 'ISRC_H',
 7 => 'ISRC_I',
);

sub PR_name($)
{
  my $p = shift;
  return $PRs{$p} if exists($PRs{$p});
  undef;
}

our %Brts = (
 1 => 'BRT_CALL',
 2 => 'BRT_RETURN',
 3 => 'BRT_BRANCH',
 4 => 'BRT_BRANCHOUT',
);

sub brt_name($)
{
  my $b = shift;
  return $Brts{$b} if exists($Brts{$b});
  undef;
}

our %scbd = (
 1 => 'SOURCE_RD',
 2 => 'SOURCE_WR',
 3 => 'SINK',
 4 => 'SOURCE_SINK_RD',
 5 => 'SOURCE_SINK_WR',
 6 => 'NON_BARRIER_INT_INST',
);

our %scbd_type = (
 1 => 'BARRIER_INST',
 2 => 'MEM_INST',
 3 => 'BB_ENDING_INST',
);

sub scbd_name($)
{
  my $b = shift;
  return $scbd{$b} if exists($scbd{$b});
  undef;
}

sub scbd_type_name($)
{
  my $b = shift;
  return $scbd_type{$b} if exists($scbd_type{$b});
  undef;
}

our @reg_sfx = ( 'd', 'd2', 'a', 'b', 'c', 'e', 'h', 'i' );

# args: type, is universal
sub rkey($$) {
  my($t, $ur) = @_;
  return if ( $t < 0 || $t > ISRC_I() );
  sprintf("%sR%s", $ur ? 'U' : '', $reg_sfx[$t]);
}

# arg: type
sub reuse_attr($) {
  my $t = shift;
  return if ( $t < 0 || $t > ISRC_I() );
  sprintf("reuse_src_%s", $reg_sfx[$t]);
}

# registry tracking history mask helpers
sub rh_write { $_[0] & 0x8000; }
sub rh_upred { $_[0] & 0x4000; }
sub rh_reuse { $_[0] & (1 << 9); }
sub rh_comp  { $_[0] & (1 << 8); }
sub rh_inlist { $_[0] & (1 << 7); }
sub rh_pred {
  my $v = ($_[0] >> 11 ) & 7;
  return unless $v;
  $v - 1;
}
sub rh_ops {
  my $v = $_[0] & 0xf;
  return unless $v;
  $v - 1;
}
sub rh_widx { ($_[0] >> 4) & 7; }

# lcols/lwors returns just array of LatIndex objects
# to gtoup them by table we need map where key is tab and value is [ array of LatIndex objects ]
sub l2map
{
  my $l = shift;
  return unless defined($l);
  my %res;
  foreach ( @$l ) {
    my $t = $_->tab();
    if ( exists $res{$t} ) { push @{ $res{$t} }, $_; }
    else { $res{$t} = [ $_ ]; }
  }
  \%res;
}

# map reloc type -> field offset
our %rel_off_map;

# arg: reloc type
sub rel2foff
{
  my $b = shift;
  return unless defined($b);
  return $rel_off_map{$b} if exists($rel_off_map{$b});
  undef;
}

our @EXPORT_OK = ( @{ $EXPORT_TAGS{'all'} } );

our @EXPORT = qw(
 brt_name
 exs
 l2map
 IType_name
 PR_name
 PTypes
 PType_name
 RTypes
 RType_name
 reg_sfx
 rel_off_map
 rkey
 reuse_attr
 scbd_name
 scbd_type_name
 rel2foff
 rh_write
 rh_upred
 rh_pred
 rh_reuse
 rh_comp
 rh_inlist
 rh_ops
 rh_widx
);

our $VERSION = '0.01';

require XSLoader;
XSLoader::load('Cubin::Ced', $VERSION);

%rel_off_map = (
 R_CUDA_ABS32_26(), 26,
 R_CUDA_TEX_HEADER_INDEX(), 0,
 R_CUDA_SAMP_HEADER_INDEX(), 20,
 R_CUDA_ABS32_LO_26(), 26,
 R_CUDA_ABS32_HI_26(), 26,
 R_CUDA_ABS32_23(), 23,
 R_CUDA_ABS32_LO_23(), 23,
 R_CUDA_ABS32_HI_23(), 23,
 R_CUDA_ABS24_26(), 26,
 R_CUDA_ABS24_23(), 23,
 R_CUDA_ABS16_26(), 26,
 R_CUDA_ABS16_23(), 23,
 R_CUDA_TEX_SLOT(), 32,
 R_CUDA_SAMP_SLOT(), 40,
 R_CUDA_SURF_SLOT(), 26,
 R_CUDA_TEX_BINDLESSOFF13_32(), 32,
 R_CUDA_TEX_BINDLESSOFF13_47(), 47,
 R_CUDA_CONST_FIELD19_28(), 26, # 26, 1 & 28, 18
 R_CUDA_CONST_FIELD19_23(), 23,
 R_CUDA_TEX_SLOT9_49(), 49,
 R_CUDA_6_31(), 31,
 R_CUDA_2_47(), 47,
 R_CUDA_TEX_BINDLESSOFF13_41(), 41,
 R_CUDA_TEX_BINDLESSOFF13_45(), 45,
 R_CUDA_FUNC_DESC32_23(), 23,
 R_CUDA_FUNC_DESC32_LO_23(), 23,
 R_CUDA_FUNC_DESC32_HI_23(), 23,
 R_CUDA_FUNC_DESC_32(), 0,
 R_CUDA_FUNC_DESC_64(), 0,
 R_CUDA_CONST_FIELD21_26(), 26,
 R_CUDA_QUERY_DESC21_37(), 37,
 R_CUDA_CONST_FIELD19_26(), 26,
 R_CUDA_CONST_FIELD21_23(), 23,
 R_CUDA_PCREL_IMM24_26(), 26,
 R_CUDA_PCREL_IMM24_23(), 23,
 R_CUDA_ABS32_20(), 20,
 R_CUDA_ABS32_LO_20(), 20,
 R_CUDA_ABS32_HI_20(), 20,
 R_CUDA_ABS24_20(), 20,
 R_CUDA_ABS16_20(), 20,
 R_CUDA_FUNC_DESC32_20(), 20,
 R_CUDA_FUNC_DESC32_LO_20(), 20,
 R_CUDA_FUNC_DESC32_HI_20(), 20,
 R_CUDA_CONST_FIELD19_20(), 20,
 R_CUDA_BINDLESSOFF13_36(), 36,
 R_CUDA_SURF_HEADER_INDEX(), 0,
 R_CUDA_CONST_FIELD21_20(), 20,
 R_CUDA_ABS32_32(), 32,
 R_CUDA_ABS32_LO_32(), 32,
 R_CUDA_ABS32_HI_32(), 32,
 R_CUDA_ABS47_34(), 34,
 R_CUDA_ABS16_32(), 32,
 R_CUDA_ABS24_32(), 32,
 R_CUDA_FUNC_DESC32_32(), 32,
 R_CUDA_FUNC_DESC32_LO_32(), 32,
 R_CUDA_FUNC_DESC32_HI_32(), 32,
 R_CUDA_CONST_FIELD19_40(), 40,
 R_CUDA_BINDLESSOFF14_40(), 40,
 R_CUDA_CONST_FIELD21_38(), 38,
 R_CUDA_YIELD_OPCODE9_0(), 0,
 R_CUDA_YIELD_CLEAR_PRED4_87(), 87,
 R_CUDA_32_LO(), 0,
 R_CUDA_32_HI(), 0,
 R_CUDA_UNUSED_CLEAR32(), 0,
 R_CUDA_UNUSED_CLEAR64(), 0,
 R_CUDA_ABS24_40(), 40,
 R_CUDA_ABS55_16_34(), 16, # 16, 8 & 34, 17
 R_CUDA_ABS20_44(), 44,
 R_CUDA_UNIFIED32_LO_32(), 32,
 R_CUDA_UNIFIED32_HI_32(), 32,
 R_CUDA_ABS56_16_34(), 16, # 16, 8 & 34, 48
 R_CUDA_CONST_FIELD22_37(), 37
);

# Preloaded methods go here.

# Autoload methods go after =cut, and are processed by the autosplit program.

1;
__END__
# Below is stub documentation for your module. You'd better edit it!

=head1 NAME

Cubin::Ced - Perl extension for CUBIN inline patching

C++ sources for SASS disassembler and asm parser located in ../ directory

To peek directory with sm modules setup env var SM_DIR

=head1 SYNOPSIS

  use Cubin::Ced;
  use Elf::Reader;
  # first create Elf reader
  my $e = Elf::Reader->new("cudatest.6.sm_61.cubin");
  # set SM_DIR
  $ENV{'SM_DIR'} = '/path/to/dir/with/smxx.so';
  # load symbols, find section with code/attributes etc
  my $cub = Cubin::Ced->new($e)
  # now you can use methods from Cubin::Ced

=head1 DESCRIPTION

There are 4 kind of methods

=over

=item 1) navigation

=item 2) patching

=item 3) getting details of currently processed instruction

=item 4) gather details about currently loaded SM

=back

=head3 Navigation

As you could assume you first need to setup right section containing code and offset inside it before you can fetch/patch something
Cubin can have several sections with code and each such section can contains several functions. So you have two method to select section/function
 
 set_s acepts string as section name or integer - section index
 set_f - string for function name
both return true in case of success

Then you need to peek offset inside section/function. I could seek to first available but disassembling of single instruction is
relative expensive operation and I don't know ahead if you really need instruction at start of section/function.
So to seek to right place you should use method
 $cub->off(offset)

To move to next instruction use 'next' method, if you patch current instruction it will be flushed automatically

you also can gather boundaries of early selected section/function with couple or methods:

 start - returns lowest offset

 end - returns end offset

Also you can use method next to move and disassemly next instruction (if those offset >= start && < end)

To get current/previous/next instruction offset use methods 'get_off', 'prev_off' & 'next_off'. On old SMs to get start
address of block use method 'block_off'

=head3 Patch methods

You can patch whole instruction with methods
  replace('text of SASS')
 to fully replace body of instruction or if you want to NOP some - use $cub->nop method

Note that you must set new offset after using of this couple of methods

Also you can patch only some fields with methods:

=over

=item * patch_pred(is_not, pred_reg_number) to patch initial predicate

=item * patch_cb(cb_field1, cb_field2) to patch Const Bank

=item * patch_tab(tab_idx, value) to patch set of values in some table

=item * patch(field_name, value) to patch single field. Note - some field can be part of table and there could be 
 no record for new value, so this table can be marked as pending until you patch some other fields of the same table

=back

You can check if you still have pending tables with $cub->ptabs method

=head3 Fetching instruction details methods

=over

=item * ins_name - name of instruction

=item * ins_class

=item * ins_false - check if instruction has predicate !@PT (or !@UPT)

=item * ins_target

=item * ins_brt

=item * ins_cc

=item * ins_sidl

=item * ins_line - line number for this particular form in MD file, useful for debugging only

=item * ins_alt - if instruction is just alternate form of more general instruction

=item * ins_mask - mask for this instruction form, all bits for fields replaced with X

=item * ins_min_wait - to extract MIN_WAIT_NEEDED property of current instruction

=item * mask - full mask of instruction, like what nvd -N option does

=item * ins_text - disaasemled string

=item * has_comp - check if current instructions rebder has compound item, return it's type is presents

=item * pred_name - returns name of instruction predicate field @PXX

=item * has_pred - returns 1 if instruction predicate @PXX and not PT

=item * ins_pred - returns ref to hash with predicates, see details L<https://redplait.blogspot.com/2025/04/nvidia-sass-disassembler-part-6.html>

=item * grep_pred($key_name) - try to find predicate with $key_name

=item * ins_prop - returns ref to hash with properties, see details L<https://redplait.blogspot.com/2025/07/sass-instructions-properties.html>

=item * ins_cb - if instruction has Const Bank - returns ref to array where
  a[0] - name of first CB field
  a[1] - name of second CB field
  a[2] - scale (if presents)

=item * ins_cbank - if instruction has Const Bank in form c[I][X] where I is number - returns ref to array [I, X] if X is also number
 or just [I] otherwise. Supports wantarray

=item * efields - returns ref to hash of enum-based fields, key is field name, value is array where
  a[0] - is_ignore
  a[1] - print
  a[2] - has default value
  a[3] - default value if a[2] is non-zero

=item * vfields - returns ref to hash of imm value fields, key is field name, value is kind or ref to array where
  a[0] - kind of imm value - like NV_UImm/NV_F32Imm/etc and
  a[1] - bit-size of imm value

=item * ins_reuse - returns mask of reuse_src_X attributes presented in current instruction. Bit index is ISRC_X - ISRC_A

=item * ins_reuse2 - returns mask of reuse_src_X attributes in this instruction form. Bit index is ISRC_X - ISRC_A

=item * ins_keep - returns mask of keep_X attributes presented in current instruction. Bit index is ISRC_X - ISRC_A

=item * ins_keep2 - returns mask of keep_X attributes in this instruction form. Bit index is ISRC_X - ISRC_A

=item * kv - returns hash of all fields, key is field name and value is, well, field value

=item * grep(regexp) - returns names of KV matched with regexp. support wantarray

=item * get(name) - returns from KV value for field 'name'

=item * get_enum(field_name) returns ref to dictionary with possible values of some enum

=item * has_lut - check if instruction has LUT operation

=item * ctrl - for old 64 bit width SM returns ctrl - see "Understanding the GPU Microarchitecture to Achieve Bare-Metal Performance Tuning" paper

=item * opcode - for old 64 bit width SM returns so called opcode for block of 7 instructions  - WTF is this

=item * cword - for 88 bit width SM returns control word - see maxas L<https://github.com/NervanaSystems/maxas/wiki/Control-Codes for details>

=item * print_cword - render CWord in form similar to official nvdisasm

=back

=head4 render method

returns tied array of rendering items - you can fetch them with simple $r->[index], format is array where indexes

=over

=item 0 - type, enum R_xxx

=item 1 - prefix

=item 2 - suffix

=item 3 - mod (like !~-)

=item 4 - this item has abs attribute

=item 5 - ref to array with trailing enums names

=item 6 - name (or undef if no name presents)

=item 7 - left part of complex render like R_desc

=item 8 - right part of complex render like R_desc

=back

left & right are ref to array where indexes

=over

=item 0 - type, enum R_xxx

=item 1 - prefix

=item 2 - argument name if presents

=back

To extract only rendering item(s) of specific type you can use 'grep' method - it supports wantarray

=head4 Methods to extract fields grouped in tables

=over

=item * tab_count - returns count of tables

=item * tab(table_index) - returns table with table_index - ref to array [ fields_names. hash ]
 hash is just dictionary with key of possible values and value is array ref with integer values for each field. If there is single
 field then values are single integer

=item * has_tfield(field_name) - tries to find field in tables, returns table index or undef

=item * tab_fields($idx) - returns array of fields names for table with index $idx (must be in 0 .. tab_count), supports wantarray
 In essecne it returns first value from I<tab> method

=item * tab_map($idx) - return second value from I<tab> method

=back

=head4 Registers tracking

See L<https://redplait.blogspot.com/2025/07/sass-instructions-registers-tracking.html>

Holds in separate object Cubin::Ced::RegTrack. There are 4 kind of registers in SASS:

=over

=item * GPRs

=item * Predicates

=item * Universal registers: L<https://redplait.blogspot.com/2025/07/sass-instructions-uniform-registers.html>

=item * Universal predicates (UPxx)

=back

Main method in Ced is $ced->track($track_db). Also you should call 'finalize' before get totals

RegTrack can give you all 4 set of registers with (u)rs/(u)ps for registers/predicates - it retuns ref to hash where key is register number

To get details track use following properties

=over

=item r($reg_idx) to get regular registers

=item ur($reg_idx) to get universal registers

=item p($reg_idx) to get predicates

=item up($reg_idx) to get univeral predicates

=back

They return array of refs to array where indexes

=over

=item 0 - instruction offset

=item 1 - mask

=item 2 - modify this register

=item 3 - if those instruction has condition predicate

=item 4 - type of register if presents

=back

To get list of used const banks use 'cbs' method - it support wantarray and return list of array where indexes

=over

=item 0 - instruction offset

=item 1 - cb index

=item 2 - cb offset

=item 3 - kind

=back

You can also extract only snapshot data for currently processed instruction with methods

=over

=item mask/mask2 for reuse

=item keep/keep2

=item snap_clear to clear snapshot

=item snap_empty to check if snapshot is empty

=item snap - main method, returns [ gprs, predicates ]

=back

=head3 Gathering details about currently loaded SM

=over

=item * width of instruction

=item * block_mask - for width 64 or 88 this is mask of instruction block size. You can get address of block like $off & ~$block_mask

=item * sm_num

=item * sm_name

=item * lut(index) - decoded string of LUT operation, see details L<https://redplait.blogspot.com/2025/07/sass-instructions-lut-operations.html>

=item * stat - return statistics of IO operations in form [flush_count, read_count, is_dirty]. Supports wantarray

=item * instrs - return names of all instructions in this SM, support regex as first optional arg

=back


=head2 EXPORT

PTypes & PType_name - names of types

RTypes & RType_name - names of render types (R_xx)

rkey($type, $is_universal) - returns name of key name for some register like 'URc'

reuse_attr($type) - returns name of reuse key like 'reuse_src_c'

=head1 SEE ALSO

C++ version of Ced: L<https://redplait.blogspot.com/2025/07/ced-sed-like-cubin-editor.html>

Cubin::Attrs module: L<https://github.com/redplait/dwarfdump/tree/main/perl/Cubin-Attrs>

base module Elf::Reader: L<https://github.com/redplait/dwarfdump/tree/main/perl/Elf-Reader>

=head1 AUTHOR

redp, E<lt>redp@E<gt>

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2025 by redp

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.30.0 or,
at your option, any later version of Perl 5 you may have available.


=cut
