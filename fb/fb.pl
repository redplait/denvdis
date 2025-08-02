#!perl -w
# script to show/extract/replace files within fat binary
use strict;
use warnings;
use Getopt::Std;
use Digest::MD5;
# couple of hand-made packages
# from https://github.com/redplait/dwarfdump/tree/main/perl/Elf-Reader
use Elf::Reader;
# from https://github.com/redplait/dwarfdump/tree/main/perl/Elf-FatBinary
use Elf::FatBinary;

# options
use vars qw/$opt_a $opt_i $opt_r $opt_v/;

sub calc_md5
{
  my $fn = shift;
  my $fh;
  open($fh, '<', $fn) or die("cannot open $fn, error $!");
  my $ctx = Digest::MD5->new;
  $ctx->addfile($fh);
  my $res = $ctx->hexdigest;
  close $fh;
  $res;
}

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] fat_binary
 Options:
  -a - arch
  -i - index of file
  -r - file to replace by index
  -v - verbose mode
EOF
  exit(8);
}

sub enum_files
{
  my $fb = shift;
  my @res;
  for ( my $i = 0; $i < $fb->count(); $i++ ) {
    my $hr = $fb->[$i];
    next if ( defined $opt_a && $hr->{'arch'} != $opt_a );
    push @res, [ $i, $hr ] if ( defined $opt_a );
    if ( defined $opt_v ) {
      printf("[%d] kind %d arch %X flags %X size %X", $i, $hr->{'kind'}, $hr->{'arch'}, $hr->{'flags'}, $hr->{'size'});
      my $n_o = $hr->{'name_off'};
      my $n_l = $hr->{'name_len'};
      printf(" name_off %X name_len %X", $n_o, $n_l) if ( defined($n_o) || defined($n_l) );
      printf(" C %X", $hr->{'decsize'}) if ( exists $hr->{'decsize'} );
      printf("\n");
    }
  }
  \@res;
}

# args - fat binary object, filename
# index to replace in opt_i, filename in opt_r
sub replace
{
  my($fb, $fn) = @_;
  if ( ! -f $opt_r ) {
     printf("file %s does not exists\n", $opt_r);
    return 0;
  }
  print calc_md5($fn) . "\n";
  if ( $fb->replace($opt_i, $opt_r) ) {
    print calc_md5($fn) . "\n";
    return 1;
  }
  0;
}

# list of [ index, $hr ]
sub extract
{
  my($fb, $l) = @_;
  my $res = 0;
  foreach my $i (@$l) {
    # form file name
    my $fn = $i->[0] . '_' . $i->[1]->{'arch'};
    $fn .= $i->[1]->{'kind'} == 1 ? '.ptx' : '.cubin';
    printf("file %s exists\n", $fn) if ( -f $fn );
    printf("extract %d to %s\n", $i->[0], $fn) if ( defined $opt_v );
    $res++ if $fb->extract($i->[0], $fn);
  }
  $res;
}

# main
my $status = getopts("va:i:r:");
usage() if ( !$status );
if ( -1 == $#ARGV ) {
  printf("where is arg?\n");
  exit(5);
}
$opt_a = hex($opt_a) if ( defined $opt_a );
$opt_i = int($opt_i) if ( defined $opt_i );
# open elf
my $e = Elf::Reader->new($ARGV[0]);
my $fb = Elf::FatBinary->new($e, $ARGV[0]);
if ( !$fb->read() ) {
  printf("cannot read %s\n", $ARGV[0]);
  return 2;
}
# lets check what we should do
if ( defined $opt_i ) {
  if ( defined $opt_r ) { # replace
    replace($fb, $ARGV[0]);
  } else {
    my $hr = $fb->[$opt_i];
    unless ( defined($hr) ) {
      printf("invalid idx %d\n", $opt_i);
      return 3;
    }
    my $item = [ $opt_i, $hr ];
    extract($fb, [ $item ]);
  }
} elsif ( defined($opt_a) || defined($opt_v) ) {
  my $l = enum_files($fb);
  extract($fb, $l) unless defined($opt_v);
} else {
  printf("unknown options combination\n");
}
