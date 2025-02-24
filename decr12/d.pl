#!perl -w
# decrypt string from ptxas v12

sub decrypt
{
  my $s = shift;
  chomp $s;
  my @res;
  foreach ( split //, $s ) {
    my $l = ord($_);
    if ( $l < 0x41 ) {
      push @res, $_;
      next;
    }
    my $mask = $l & 0xDF;
    my $si = $mask - 0x41;
    if ( $si <= 0xc ) {
      push @res, chr($l + 0xd);
    } else {
      $si = $mask - 0x4e;
      if ( $si < 0x0d ) {
        push @res, chr($l - 0xd);
      } else {
        push @res, chr($l);
      }
    }
  }
  return join '', @res;
}

while(<>) {
  printf("%s\n", decrypt($_));
}