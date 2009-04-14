/* Easiest case, because constants are available */

int repeat01()
{
  int i, j, n;

  i = 0;
  j = 1;
  n = 10;

  do {
    i++;
    j += 2;
  } while(j<n);

  return i+j;
}
