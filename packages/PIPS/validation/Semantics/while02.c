/* Just like while01, but the value of n is unknown and the loop may be entered or not */

int while02()
{
  int i, j, n;

  i = 0;
  j = 1;

  while(j<n) {
    i++;
    j += 2;
  }
  return i+j;
}
