// Check side effects in conditions

#include <stdio.h>

void if02()
{
  int i = 0;
  int j = 0;

  if(i++, i>0)
    // This branch is taken by gcc
    j = 1;
  else
    // Should not be reached
    j = 2;

  printf("%d\n", j);
  return;
}

main()
{
  if02();
}

