// Check analysis of shift operations (found in mpeg2enc)

#include <stdio.h>

int shift03(int i, int k)
{
  int j = i;

  // Check a subset of particular cases
  if(j<0 && 2<= k && k <= 4)
    j = j <<k;
  else if(j>=0 && 2<= k && k <= 4 )
    j = j <<k;
  else
    j = j <<k;

  printf("j=%d\n", j);
  return j;
}

main()
{
  int i = 20;
  shift03(i, 2);
}
