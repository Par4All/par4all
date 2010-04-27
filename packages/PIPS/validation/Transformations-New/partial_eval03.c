/* Extend bug detected by Serge Guelton: no simplification of i = 3*j - 3*j; */

int foo(int i)
{
  int j;

  j = 3*i - 2*i - i;
  j = 3*i - 2*i - i + 1;
  j = 1 + 3*i - 2*i - i + 1;
  j = 1 + i;
  j = i + 1;
  return j;
}
