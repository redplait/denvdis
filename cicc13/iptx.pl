#!perl -w
# try intersect ptx.txt from cicc with ptx_ops2.txt extracted from ptxas
use strict;
use warnings;

# instr names from ptx_ops2.txt
my %g_ins;

sub read_ops2
{
  my $fname = shift;
  my @mask = ( 0 ) x16;
  my($fh, $str, $m, $rest, $iname);
  open($fh, '<', $fname) or die("Cannot open $fname, error $!");
  while( $str = <$fh> ) {
    chomp $str;
    next if ( $str !~ /^\d+\s+(.*)$/ );
    $str = $1;
    # opcode name and tail starts at 48
    $iname = (split /\t/, substr($str, 48))[0];
    $g_ins{$iname}++; # insert into g_ins
    # make mask
    $m = substr($str, 0, 47);
    my @tmp = map { hex $_; } split /\s+/, $m;
    # or with total mask
    foreach my $mi ( 0 .. 15 ) {
      $mask[$mi] |= $tmp[$mi] if $tmp[$mi];
    }
  }
  close $fh;
  # dump resulting mask
  foreach my $mi ( @mask ) {
    printf("%2.2X ", $mi);
  }
  printf("\n");
}

# read ptx.txt and check in g_ins every instruction
sub apply_ptx
{
  my $fname = shift;
  my($fh, $str);
  open($fh, '<', $fname) or die("Cannot open $fname, error $!");
  my $ln = 0;
  # some stat
  my $found = 0;
  my $bad = 0;
OUTER:
  while( $str = <$fh> ) {
    chomp $str;
    $ln++;
    # filter some trash
    next if ( $str eq ';' || $str eq '' );
    next if ( $str =~ /\{/ );
    $str =~ s/^\s+//;
    my $body = ( split /\s+/, $str )[0];
    next unless $body;
    $body =~ s/\s+$//;
    $body =~ s/;$//;
    next if ( $body =~ /\.reg/ );
    next if ( $body eq ',' or $body eq '}' or $body eq ';' or $body eq '//' or $body eq '@' or $body eq '#' or $body eq '' );
    next if ( $body =~ /^\}/ );
    next if ( $body =~ /^\]/ );
    next if ( $body =~ /^\.row/ );
    next if ( $body =~ /^\.col/ );
    next if ( $body =~ /^\.v2/ );
    next if ( $body =~ /^\.v4/ );
    next if ( $body =~ /^\.pred/ );
    next if ( $body =~ /^\.param/ );
    # split on '.'
    my $need_dot = 0;
    if ( $body =~ /^\./ ) {
      $need_dot = 1;
      $body = substr($body, 1);
    }
    my @chain = split /\./, $body;
    my $name = $chain[0];
    $name = '.' . $name if ( $need_dot );
    if ( exists $g_ins{$name} ) {
      $found++;
      next;
    }
    # try all
    foreach my $i ( 1 .. scalar(@chain) - 1 ) {
      $name = $name . '.' . $chain[$i];
      if ( exists $g_ins{$name} ) {
        $found++;
        next OUTER;
      }
    }
    # finally dump
    printf("line %d: %s\n", $ln, $str);
    $bad++;
  }
  close $fh;
  printf("found %d bad %d\n", $found, $bad);
}

# main
read_ops2('ptx_ops2.txt');
apply_ptx('ptx.txt');
