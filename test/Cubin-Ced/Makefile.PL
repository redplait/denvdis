use 5.030000;
use ExtUtils::MakeMaker;
use Config;
use LWP::Simple;

# download couple of files
my $e_inc = 'elf.inc';
if ( !-f $e_inc ) {
  my $url = 'https://github.com/redplait/dwarfdump/blob/main/perl/elf.inc';
  my $status = getstore($url, $e_inc);
  die("cannot loaf $e_inc") if ( !is_success($status) );
}
my $t_cubin = 'cudatest.6.sm_61.cubin';
if ( !-f $t_cubin ) {
  my $url = 'https://github.com/cloudcores/CuAssembler/raw/refs/heads/master/TestData/CuTest/cudatest.6.sm_61.cubin';
  my $status = getstore($url, $t_cubin);
  die("cannot loaf $t_cubin") if ( !is_success($status) );
}

# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    NAME              => 'Cubin::Ced',
    VERSION_FROM      => 'lib/Cubin/Ced.pm', # finds $VERSION, requires EU::MM from perl >= 5.5
    PREREQ_PM         => { 'Elf::Reader' => 0.01 }, # e.g., Module::Name => 1.1
    ABSTRACT_FROM     => 'lib/Cubin/Ced.pm', # retrieve abstract from module
    AUTHOR            => 'redp <redp@nail.ru>',
    #LICENSE           => 'perl',
    #Value must be from legacy list of licenses here
    #http://search.cpan.org/perldoc?Module%3A%3ABuild%3A%3AAPI
    XSOPT             => '-C++',
    CC                => 'g++ -std=c++20',
    LIBS              => ["-Wl,--as-needed -L$Config{'sitearch'}/auto/Elf/Reader -l:Reader.so -L.. -lced -lstdc++"], # e.g., '-lm'
    DEFINE            => '', # e.g., '-DHAVE_SOMETHING'
    INC               => '-I. -I.. -I../../scripts -I../../../FP16/include/ -I/home/redp/disc/ELFIO', # e.g., '-I. -I/usr/include/other'
	# Un-comment this if you add C files to link with later:
    # OBJECT            => '$(O_FILES)', # link all the C files too
);
