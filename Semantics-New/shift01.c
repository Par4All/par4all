// Check analysis of shift operations (found in mpeg2enc)

#include <stdio.h>

int shift01(int i)
{
  int j = i << 2;
  int n = 1;

  j = j <<n;


  printf("j=%d\n", j);
  return j;
}

main()
{
  int i = 20;
  shift01(i);
}
