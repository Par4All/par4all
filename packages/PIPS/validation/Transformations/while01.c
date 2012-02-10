// Check that useless while loops are removed

#include <stdio.h>

void while01()
{
  int i = 1;
  while(i==1){
    i++;
  }
  printf("%d", i);
}
