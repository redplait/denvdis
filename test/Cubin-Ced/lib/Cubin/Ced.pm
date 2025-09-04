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
 1) navigation
 2) patching
 3) getting details of currently processed instruction

=head3 Navigation
As you could assume you first need to setup right section containing code and offset inside it before you can fetch/patch something
Cubin

=head3 Patch methods

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
