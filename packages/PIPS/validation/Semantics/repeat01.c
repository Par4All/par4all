/* Easiest case, because constants are available */

int repeat01()
{
  int i, j;

  i = 0;
  j = 1;

  {
    int n = 10;

    do {
      i++;
      j += 2;
    } while(j<n);
  }
  return i+j;
}
