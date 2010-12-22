// Check analysis of shift operations (found in mpeg2enc)

#include <stdio.h>

int shift02(int i, int k)
{
  int j = i;

  // Check a subset of particular cases
  if(k==0)
    j = j <<k;
  else if(j<0 && k>0)
    j = j <<k;
  else if(j>=0 && k<0)
    j = j <<k;
  else
    j = j <<k;

  printf("j=%d\n", j);
  return j;
}

main()
{
  int i = 20;
  shift02(i, 2);
}
