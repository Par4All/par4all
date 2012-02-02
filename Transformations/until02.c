// Check that useless until loops are removed, but that comments are preserved

#include <stdio.h>

void until02()
{
  int i = 1;
  int j = 3;

  i++;
  do {
    int j = 2;
    i += j;
  } while(0);

  printf("%d", i);
}
