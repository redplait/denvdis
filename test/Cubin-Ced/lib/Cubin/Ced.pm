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
);

sub PType_name
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

sub RType_name
{
  my $p = shift;
  return $RTypes{$p} if exists($RTypes{$p});
  undef;
}

our @EXPORT_OK = ( @{ $EXPORT_TAGS{'all'} } );

our @EXPORT = qw(
 exs
 PTypes
 PType_name
 RTypes
 RType_name
);

our $VERSION = '0.01';

require XSLoader;
XSLoader::load('Cubin::Ced', $VERSION);

# Preloaded methods go here.

# Autoload methods go after =cut, and are processed by the autosplit program.

1;
__END__
# Below is stub documentation for your module. You'd better edit it!

=head1 NAME

Cubin::Ced - Perl extension for CUBIN inline patching

C++ sources for SASS disassembler and asm parser located in ../ directory

=head1 SYNOPSIS

  use Cubin::Ced;
  use Elf::Reader;
  # first create Elf reader
  my $e = Elf::Reader->new("cudatest.6.sm_61.cubin");
  # load symbols, find section with code etc
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

you also can gather boundaries of early selected section/function with couple or methods:

 start - returns lowest offset

 end - returns end offset

Also you can use method next to move and disassemly next instruction (if those offset >= start && < end)

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

=item * ins_name

=item * ins_class

=item * ins_false - check if instruction has predicate !@PT (or !@UPT)

=item * ins_target

=item * ins_brt

=item * ins_cc

=item * ins_sidl

=item * ins_line - line number for this particular form in MD file, useful for debugging only

=item * ins_alt - if instruction is just alternate form of more general instruction

=item * ins_mask - mask for this instruction form, all bits for fields replaced with X

=item * mask - full mask of instruction, like what nvd -N option does

=item * ins_text - disaasemled string

=item * ins_pred - returns ref to hash with predicates, see details https://redplait.blogspot.com/2025/04/nvidia-sass-disassembler-part-6.html

=item * ins_prop - returns ref to hash with properties, see details https://redplait.blogspot.com/2025/07/sass-instructions-properties.html

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

=item * kv - returns hash of all fields, key is field name and value is, well, field value

=item * get_enum(field_name) returns ref to dictionary with possible values of some enum

=item * has_lut - check if instruction has LUT operation

=back

render method returns tied array of rendering items - you can fetch them with simple $r->[index], format is array where indexes

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

Methods to extract fields grouped in tables

=over

=item * tab_count - returns count of tables

=item * tab(table_index) - returns table with table_index - ref to array [ fields_names. hash ]
 hash is just dictionary with key of possible values and value is array with values for each field. If there is single field then
 key is just single value

=back

=head3 Gathering details about currently loaded SM

=over

=item * width of instruction

=item * sm_num

=item * sm_name

=item * lut(index) - decoded string of LUT operation, see details https://redplait.blogspot.com/2025/07/sass-instructions-lut-operations.html

=item * stat - return statistics of IO operations in form [flush_count, read_count, is_dirty]. Supports wantarray

=back


=head2 EXPORT

None by default.


=head1 SEE ALSO

C++ version of Ced: https://redplait.blogspot.com/2025/07/ced-sed-like-cubin-editor.html

=head1 AUTHOR

redp, E<lt>redp@E<gt>

=head1 COPYRIGHT AND LICENSE

Copyright (C) 2025 by redp

This library is free software; you can redistribute it and/or modify
it under the same terms as Perl itself, either Perl version 5.30.0 or,
at your option, any later version of Perl 5 you may have available.


=cut
