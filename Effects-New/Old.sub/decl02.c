#include <stdio.h>

int main()
{
  int i;
  int j = 2;
  int k = i;
  int l = j++;
  int m = 4;
  int n = m;
  
  i = 5;
  printf("i = %d, j = %d, k = %d, l = %d, m = %d, n = %d \n", i, j, k, l, m, n);
  return(0);
}
