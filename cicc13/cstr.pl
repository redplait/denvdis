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
  my($fh, $str, $idx);
  open($fh, '<', $fname) or die("Cannot open %fname, $!");
  while( $str = <$fh> ) {
    chomp $str;
    if ( $str =~ /^\[(\d+)\].*:$/ ) {
      $idx = int($1);
      next;
    }
    if ( $str =~ /^ (.*)/ ) {
      next if exists( $db->{$1} );
      printf("[%d] %s\n", $idx, $1);
    }
  }
  close $fh;
}

# main
if ( scalar(@ARGV) != 2 ) {
  printf("format: file1 file2\n");
  exit(6);
}
my %db;
parse1($ARGV[0], \%db);
parse_cmp($ARGV[1], \%db);