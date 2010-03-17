/* Easiest case, because constants are available. Check that the loop is always entered. */

int repeat02()
{
  int i, j;
  int n = 10;

  i = 0;
  j = 1;
  //  {
  //    int n = 10;

    do {
      i++;
      j += 2;
    } while(j>n);
    //}
  return i+j;
}
