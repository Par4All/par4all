/* Easiest case, because the loop is always entered */

int while01()
{
  int i, j, n;

  i = 0;
  j = 1;
  n = 0;
  {
    int n = 10;

    while(j<n) {
      i++;
      j += 2;
    }
  }

  return i+j;
}
