#!perl -w
use strict;
use warnings;

my($str, %attrs);
my $state = 0;
while( $str = <> ) {
  chomp $str;
  if ( $str =~ /^setp/ ) { # we need only string next after setp
    $state = 1;
    next;
  }
  if ( $state ) {
    if ( $str !~ /^(.*) ([a-f0-9]+)$/i ) {
      printf("bad str %s\n", $str);
    } else {
      my $what = $1;
      my $idx = hex($2);
      if ( exists $attrs{$idx} ) {
        my $ar = $attrs{$idx};
        push @$ar, $what;
      } else { # add new array
        $attrs{$idx} = [ $what ];
      }
    }
    $state = 0;
    next;
  }
}

# dump results
foreach my $a ( sort { $a <=> $b } keys %attrs ) {
  printf("%s/0x%X:", $a, $a);
  my $ar = $attrs{$a};
  printf(" %s", $_) for @$ar;
  printf("\n");
}