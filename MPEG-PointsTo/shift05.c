// Check analysis of shift operations (found in mpeg2enc)

// make sure that the sign information is preserved when possible

#include <stdio.h>

int shift05(int i, int k)
{
  int j = i;
  int l = -1;
  int m;
  int n;
  int o;

  if(i>0) {
    j = 2<<i;
    o = j;
  }
  else if(i==0)
    j = 2<<i;
  else // i<0
    j = 2<<i;

  m = 2<<i;
  n = (-2)<<i;

  // j=2097152, l=-1, m=2097152, n=-2097152, o=2097152
  printf("j=%d, l=%d, m=%d, n=%d, o=%d\n", j, l, m, n, o);
  return j;
}

main()
{
  int i = 20;
  shift05(i, 2);
}
