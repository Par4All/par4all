/* Pointer to a 2-D array */

int pointer21(int (*pa)[5][6], int i, int j)
{
  int k;
  k = pa[i][0][j];
  return k;
}
