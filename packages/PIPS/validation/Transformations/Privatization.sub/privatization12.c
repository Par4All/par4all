/* To check that the initialization of k does block the privatization */

#include <stdio.h>

int main(int argc, char* argv[])
{
  int i = 0;
  int j = i++;
  int k = 0;
  for(i=0;i<10;i++)
    for(j=0;j<10;j++)
      k = k+i*j;
  // This would prevent the privatization
  //printf("%d\n",k);
  return 0;
}
