/* Check bug detected by Serge Guelton: no simplification of i = j - j; */

int foo(int i)
{
  int j;
  j = i - i;
  return j;
}
