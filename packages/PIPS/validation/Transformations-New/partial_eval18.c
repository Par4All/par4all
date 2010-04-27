/* Bug for i++ + ....? No, simply commutativity and associativity are
   not exploited when the expression is not affine. */

void foo(int i)
{
}

int main()
{
  int i=2;
  int j;
  int k = 3;
  i++ + j-j;
  i++ + (j-j);
  i++ + k-k;
  i++ + (k-k);
  return i;
}
