// Check side effects in conditions

#include <stdio.h>

void if01()
{
  int i = 0;
  int j = 0;

  if(i++)
    // Should not be reached
    j = 1;

  if(i++)
    // Should be reached
    j = 2;

  printf("%d ", j);

  i = 0;

  if(i++>0)
    // Should not be reached
    j = 1;

  if(i++>0)
    // Should be reached
    j = 2;

  printf("%d ", j);

  i = 0;

  if(i++, i>0)
    // Should be reached according to gcc
    j = 10;

  if(i++, i>0)
    // Should be reached
    j = 2*j + 2;

  printf("%d ", j);

  return;
}

main()
{
  if01();
}

