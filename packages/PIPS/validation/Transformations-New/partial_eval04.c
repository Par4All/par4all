/* Extend bug detected by Serge Guelton: no simplification of i = 3*j - 3*j; */

int foo(int i)
{
  int j;
  int n;

  j = n*i - n*i;
  j = n*i - i*n;
  return j;
}
