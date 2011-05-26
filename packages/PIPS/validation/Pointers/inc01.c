/* test case to illustrate how use-def can't be computed without
   points-to */
int inc01(int *pi)
{
  int * q;
  int i =0;
  q = pi;
  i = *q;
  *q = i++;
  return 0;
}
