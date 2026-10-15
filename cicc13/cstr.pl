#!perl -w
# lame script to compare ctor strings from couple of files
use strict;
use warnings;

sub parse1
{
  my($fname, $db) = @_;
  my($fh, $str);
  open($fh, '<', $fname) or die("Cannot open %fname, $!");
  while( $str = <$fh> ) {
    chomp $str;
    next if ( $str =~ /^\[.*\].*:$/ );
    if ( $str =~ /^ (.*)/ ) {
      $db->{$1}++;
    }
  }
  close $fh;
}

sub parse_cmp
{
  my($fname, $db) = @_;
  my($fh, $str, $idx, %nf);
  open($fh, '<', $fname) or die("Cannot open %fname, $!");
  # for stat - amount of ctors, hash for not found
  my $ct_cnt = 0;
  while( $str = <$fh> ) {
    chomp $str;
    if ( $str =~ /^\[(\d+)\].*:$/ ) {
      $idx = int($1);
      ++$ct_cnt;
      next;
    }
    if ( $str =~ /^ (.*)/ ) {
      next if exists( $db->{$1} );
      $nf{$idx}++;
      printf("[%d] %s\n", $idx, $1);
    }
  }
  close $fh;
  # dump stat
  printf("---- total ctrs %d\n", $ct_cnt);
  my $nf_size = scalar keys %nf;
  printf("Not found %d (%f)\n", $nf_size, 1.0 * $nf_size / $ct_cnt) if $ct_cnt;
}

# main
if ( scalar(@ARGV) != 2 ) {
  printf("format: file1 file2\n");
  exit(6);
}
my %db;
parse1($ARGV[0], \%db);
parse_cmp($ARGV[1], \%db);