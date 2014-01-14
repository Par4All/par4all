/* Pointer to a 1-D array */

int pointer20(int (*pa)[5], int i)
{
  int j;
  j = pa[i][0];
  return j;
}
