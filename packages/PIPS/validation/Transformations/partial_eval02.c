/* Extend bug detected by Serge Guelton: no simplification of i = 3*j - 3*j; */

int foo(int i)
{
  int j;
  j = 3*i - 3*i;
  return j;
}
