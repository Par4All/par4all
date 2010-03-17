/* Just like while01 and while02, but although the value of n is
   unknown, the loop is always entered */

int while03()
{
  int i, j, n;

  i = 0;
  j = 1;

  if(n<=1) exit(1);

  while(j<n) {
    i++;
    j += 2;
  }
  return i+j;
}
