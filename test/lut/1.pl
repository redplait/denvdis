#!perl -w
# dirty hack to generate expressions for sympy
sub concat_expr
{
  my($r, $e) = @_;
  return $r . '| (' . $e . ')' if ( length $r );
  '(' . $e . ')';
}

# stolen from https://forums.developer.nvidia.com/t/what-does-lop3-lut-mean-how-is-it-executed/227472/7
sub gen_expr
{
  my $r = shift;
  my $res;
  $res = concat_expr($res, '~a & ~b & ~c') if ($r & 0x01);
  $res = concat_expr($res, '~a & ~b &  c') if ($r & 0x02);
  $res = concat_expr($res, '~a &  b & ~c') if ($r & 0x04);
  $res = concat_expr($res, '~a &  b &  c') if ($r & 0x08);
  $res = concat_expr($res, 'a & ~b & ~c') if ($r & 0x10);
  $res = concat_expr($res, 'a & ~b &  c') if ($r & 0x20);
  $res = concat_expr($res, 'a &  b & ~c') if ($r & 0x40);
  $res = concat_expr($res, 'a &  b &  c') if ($r & 0x80);
  $res;
}

print<<HEAD;
from sympy import *
a, b, c = symbols('a b c')
HEAD

for ( my $i = 1; $i < 0xff; $i++ ) {
  my $res = gen_expr($i);
  printf("simplified_expression = simplify_logic(%s, form=\"cnf\")\n", $res);
  printf("print(simplified_expression)\n");
}