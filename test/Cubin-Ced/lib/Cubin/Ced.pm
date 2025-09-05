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

our @EXPORT_OK = ( @{ $EXPORT_TAGS{'all'} } );

our @EXPORT = qw(
	
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

There are 3 kind of methods

=over

=item 1) navigation

=item 2) patching

=item 3) getting details of currently processed instruction

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
