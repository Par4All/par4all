/* Triply nested loops for Vivien Maisonneuve's PhD */

#include <stdio.h>

int main()
{
  int i=0, j, k, l=0, n=10;

  for(i=0;i<n;i++)
  for(j=0;j<n;j++)
  for(k=0;k<n;k++)
    l++;

  printf("l=%d\n", l);
  return 0;
}
