/* Check the impact of exit() */
#include<stdlib.h>
int* exit01(int *p)
{
  
  p = (int*) malloc(4*sizeof(int));
  exit(0);
  return p;
}

int main()
{
  int *q, *r;
  q = exit01(r);
  return 0;
}
