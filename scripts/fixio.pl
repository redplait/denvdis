#!perl -w
# dirty hack to add labels for EIATTR_XXX_INSTR_OFFSET
use strict;
use warnings;

my(%sects, $sidx, $labels, $sname, $str, $state);
$sidx = $state = 0;
while( $str = <> )
{
   #D printf("%d %d %s", $state, $sidx, $str);
   print $str if ( $state != 4 and $state != 2 );
   # try to find section
   if ( $str =~ /\.section\s+([^,]+),/ ) {
      print $str if ( 2 == $state );
      $sidx++;
      $state = 0;
      $sname = $1;
      if ( $sname =~ /^\.nv\.info.(.+)$/ ) { # this is nv.info
        $sname = '.text.' . $1;
        $state = 1;
        next;
      }
      if ( $sname =~ /^\.text/ ) {
        next unless exists $sects{$sname};
        $labels = $sects{$sname};
        $state = 2;
        next;
      }
      next; # skip all remained sections
  }
  # 1 - try to find nvinfo : EIATTR_XXX_INSTR_OFFSET
  if ( $state == 1 or $state == 3 ) {
    if ( $str =~ /nvinfo\s*:\s*EIATTR.*INSTR_OFFSET/ ) {
      $state = 3;
      next;
    }
    next if ( $state == 1 );
  }
  # 3 - ..[\d]..
  if ( $state == 3 ) {
     $state = 4 if ( $str =~ /\.\[\d+\]\./ );
     next;
  }
  # print. 4 - expect label or string with offset
  if ( 4 == $state ) {
    if ( $str =~ /^\..*:/ ) {
      print $str;
      next;
    } elsif ( $str =~ /(.*\.word\s+)0x(\S+)/ ) {
      if ( !exists $sects{$sname} ) {
        my %l;
        $l{'_'} = $sidx;
        $sects{$sname} = \%l;
      }
      $labels = $sects{$sname};
      my $off = hex($2);
      my $label = sprintf(".L_%d_%X", $sidx, $off);
      $labels->{$off} = $label;
      # print modified string
      printf("%s (%s - .S_%d)\n", $1, $label, $sidx);
      $state = 3;
      next;
    }
    $state = 3;
    print $str;
    next;
  }
  # 2 - we inside text section
  if ( 2 == $state ) {
    if ( $str !~ /\/\*(.+)\*\/(\s+.*;.*)$/ ) {
      print $str; next;
    }
    my $off = hex($1);
    if ( !$off ) { # insert .S label at start of section
      printf(".S_%d:\n", $labels->{'_'});
    }
    if ( exists $labels->{$off} ) { # insert label for some offset
      printf("%s:\n", $labels->{$off});
    }
    print $str;
  }
}