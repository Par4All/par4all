// Check that useless while loops are removed

#include <stdio.h>

void while02()
{
  int i = 1;
  while(i<=2){
    i++;
  }
  printf("%d", i);
}
