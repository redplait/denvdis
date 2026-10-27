#!perl -w
# script to compare ptx_intr with 13.4 version
use strict;
use warnings;

sub parse
{
  my($fname, $db) = @_;
  my($fh, $str);
  open($fh, '<', $fname) or die("Cannot open %fname, $!");
  while( $str = <$fh> ) {
    chomp $str;
    my @tmp = split /\s+/, $str;
    $db->{$tmp[1]} = $tmp[0];
  }
  close $fh;
}

sub cmp_intr {
  my($odb, $ndb) = @_;
  my(%visited, $name);
  my $latch = 0;
  my $n_cnt = 0;
  foreach $name ( keys %$ndb ) {
    $visited{$name}++;
    next if ( exists $odb->{$name} );
    printf("--- new:\n") if ( !++$latch );
    printf(" %s\n", $name);
    $n_cnt++;
  }
  printf("total %d\n", $n_cnt) if ( $n_cnt );
  # check removed
  $n_cnt = $latch = 0;
  foreach $name ( keys %$odb ) {
    next if ( exists $visited{$name} );
    printf("--- removed:\n") if ( !++$latch );
    printf(" %s\n", $name);
    $n_cnt++;
  }
  printf("total %d\n", $n_cnt) if ( $n_cnt );
}

# main
if ( scalar(@ARGV) != 2 ) {
  printf("format: file1 file2\n");
  exit(6);
}
my(%odb, %ndb);
parse($ARGV[0], \%odb);
parse($ARGV[1], \%ndb);
cmp_intr(\%odb, \%ndb);

