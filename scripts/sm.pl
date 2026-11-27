#!perl -w
# extract all instructions from pile of smXX.so
use strict;
use warnings;
use Cubin::SM;
use Carp;
use Data::Dumper;
use Getopt::Std;

# options
use vars qw/$opt_p/;

sub usage()
{
  print STDERR<<EOF;
Usage: $0 [options] latency.file
 Options:
  -p generate pm
EOF
  exit(8);
}

# main
my $state = getopts("p");
usage() if ( !$state );
my $sm = Cubin::SM->new();
my %ins;
foreach my $fn ( @ARGV ) {
  if ( !$sm->load($fn) ) {
    carp("cannot load $fn");
    next;
  }
  $ins{$_} = 1 foreach @$sm;
}
# check what we must produce
if ( defined $opt_p ) {
 # produce footer
 print<<'EOF';
package Cubin::SM::Data;

use 5.030000;
use strict;
use warnings;

require Exporter;
use AutoLoader qw(AUTOLOAD);

our @ISA = qw(Exporter);
EOF
 # assogn indices and fill back_array
 my @ba;
 my $idx;
 foreach my $i ( sort { $a cmp $b } keys %ins ) {
   $ins{$i} = $idx++;
   push @ba, $i;
 }
 # dump ins & back_array
 printf('our %sins = (' . "\n", '%');
 foreach my $i ( sort { $a cmp $b } keys %ins ) {
   printf("'%s' => %d,\n", $i, $ins{$i});
 }
 printf(");\n" . 'our @ba = (' . "\n");
 printf("'%s',\n", $_) for @ba;
 printf(");\n");
 # produce trailer
 print<<'EOT';
our @EXPORT = qw(ins ba);
1;
__END__
EOT
} else {
  foreach my $i ( sort { $a cmp $b } keys %ins ) {
    printf("%s\n", $i);
  }
}
