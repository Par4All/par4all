#include<stdlib.h>

int main()
{
  int *pi;
  short *ps ;
  pi = (int*)malloc(4*sizeof(int));
  pi = pi+2;
  ps = (short*)pi;
  ps -= 4;
  free(ps);

  return 0;
}
