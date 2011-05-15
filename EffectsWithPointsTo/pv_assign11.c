/* pointer values modified in lhs and rhs.
  (inspired from a test case pointed out by SG on 09/14/09) */
int main()
{
  int i;
  int *p, *q;
  int a[10];
  int b[10];

  p = &a[0];
  q = &b[0];
  *p++ = 2 * *q++;

  return(0);
}
