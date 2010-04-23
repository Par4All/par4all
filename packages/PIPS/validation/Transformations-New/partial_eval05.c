/* Extend bug detected by Serge Guelton: no simplification of i = 3*j - 3*j; */

int foo(int i)
{
  int j;
  int n;

  j = 1 + i - i - 1;

  return j;
}
